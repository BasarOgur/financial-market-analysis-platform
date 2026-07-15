# Architecture Decisions — rag-service

Each entry: the choice, the rejected alternative, and why. Rule 7 in AGENTS.md.

## 1. Raw REST via httpx instead of provider SDKs

**Chose:** one ~25-line class per provider over `httpx` in `shared/llm.py` /
`shared/embeddings.py`.
**Rejected:** installing `google-genai` + `openai` + `anthropic` SDKs.
**Why:** three SDKs for three ~20-line request shapes is dependency weight and
churn (the Gemini SDK is mid-migration to an "interactions" API as of mid-2026).
REST keeps provider swapping symmetrical and the whole surface auditable.
Revisit if we need streaming or provider-side tool calling.

## 2. Services run in place; only `packages/shared` is installed

**Chose:** `rag-service` is not pip-installed; run `python api.py` from its
directory. `packages/shared` is installed editable.
**Rejected:** making each service an installable package.
**Why:** the spec layout (`api.py`, `ingest/`, `retrieval/` at service root)
would install generic top-level module names (`api`, `ingest`, `retrieval`)
into the shared venv — the second service would collide. Anything reusable
belongs in `packages/shared` anyway.

## 3. Chroma as the vector store

**Chose:** `chromadb` PersistentClient, cosine space, embeddings supplied by us.
**Rejected:** FAISS or a hand-rolled numpy cosine store.
**Why:** persistence, metadata filtering, and upsert out of the box; boring and
standard; and it bundles an ONNX MiniLM model we reuse as the keyless `local`
embedding provider (decision 4) — one dependency doing two jobs. A numpy store
would be smaller but re-implements persistence and filtering we'd need
immediately.

## 4. `local` embedding provider = Chroma's bundled ONNX MiniLM

**Chose:** third embedding provider `local` (all-MiniLM-L6-v2, ONNX) for
keyless dev, offline tests of the full path, and a reproducible eval baseline.
**Rejected:** `sentence-transformers`.
**Why:** sentence-transformers drags in torch (~2GB) for the same model;
chromadb is already installed. Trade-off: `shared/embeddings.py` lazily
imports chromadb for this provider only — acceptable coupling, documented in
the class docstring.

## 5. Dense-only retrieval with a documented hybrid/rerank seam

**Chose:** single dense pass in `Retriever.retrieve()`; hybrid (BM25 merge)
and cross-encoder rerank are a documented seam, not code.
**Rejected:** building BM25 + reranking now.
**Why:** YAGNI until the eval says otherwise — and the eval now gives the
concrete trigger: q04 and q11 rank 5 with dense-only. Callers only see
`retrieve(query, top_k)`, so the upgrade is internal to one method.

## 6. Eval gold labels are text spans, not chunk ids

**Chose:** `gold_spans` (normalized substring match) mark a retrieved chunk
as relevant.
**Rejected:** gold chunk ids.
**Why:** chunk ids change whenever chunking parameters change, silently
invalidating the dataset. Spans survive re-chunking as long as they're shorter
than the overlap window. Trade-off: a span appearing in multiple chunks counts
any of them as a hit — acceptable at this corpus size.

## 7. Synthetic fixture corpus (clearly labeled)

**Chose:** six hand-written documents mirroring 10-K/10-Q/call-transcript
structure for two fictional companies plus one more, with internally
consistent figures.
**Rejected:** downloading real EDGAR filings.
**Why:** reproducible (no network, no drift when filings are amended), small
enough to eyeball gold labels, and safe from stale-fact confusion. Real EDGAR
ingestion is an ingest-only change later; frontmatter metadata already models
it. Every fixture carries `note: synthetic fixture`.

## 8. Word-budget chunking, no tokenizer dependency

**Chose:** paragraph packing to a `max_words` budget (220 words ≈ 300 tokens)
with sliding-window overlap for oversized paragraphs.
**Rejected:** token-exact chunking via `tiktoken`/HF tokenizers.
**Why:** the budget only needs to be approximately right for retrieval
quality; a tokenizer is a dependency serving a ±25% precision we don't need.
Revisit if we hit hard context limits with long-context filings.

## 9. Default embedding model `gemini-embedding-001`

**Chose:** originally spec-mandated `text-embedding-004`; on first live run
(2026-07-05) the API returned 404 — Google retired it. Switched the default
to `gemini-embedding-001` (GA, 3072-dim; `gemini-embedding-2` also available).
**Why this validates the design:** the fix was a one-line config default
change, zero call-site changes — exactly the provider/model churn the
config-driven client was built for. Re-ingest (`--reingest`) after any model
switch; dimensions differ.

## 10. Faithfulness via LLM judge; abstention via exact phrase

**Chose:** LLM-as-judge (JSON verdict) for faithfulness of generated answers;
abstention correctness checked deterministically against the exact
`ABSTAIN_PHRASE` the system prompt mandates.
**Rejected:** n-gram/keyword overlap scoring for faithfulness; LLM judging of
abstention.
**Why:** overlap metrics can't catch a wrong number attributed to the right
sentence — the failure mode that matters in finance. Abstention, by contrast,
is a string contract we control, so a deterministic check is strictly more
reliable (and free). Judge weakness (same-family model bias) is documented in
`eval/judge.py`: point `FMA_LLM_MODEL` at a stronger model when judging.

## 11. Retry/backoff on rate-limited LLM and embedding calls

**Chose:** `shared/llm.py` and `shared/embeddings.py` each retry `429`/`503`
responses with increasing backoff (5-30s for embeddings, 15-240s for
generation) before raising.
**Rejected:** failing immediately on any non-200 response.
**Why:** Google's free tier returns `503` transiently under load and `429`
when a per-minute burst limit is hit — both recoverable by waiting. What
backoff can't fix: the free tier also caps `generate_content` at **20
requests/day per model**. `--answers` needs 28 calls (14 questions x 2:
generate + judge), so a full run currently spans multiple days or needs a
paid tier. Added `eval/run_eval.py --limit N` to run a quota-sized subset;
`eval/results.md` documents the answer-quality section as partial until the
daily cap resets. This is a real constraint, not a bug — retrying harder
can't out-wait a daily quota.
