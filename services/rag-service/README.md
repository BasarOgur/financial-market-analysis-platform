# rag-service

Grounded Q&A over SEC filings and earnings-call transcripts. Retrieves the
relevant chunks, generates an answer **only** from them with `[n]` citations,
and abstains when the documents don't contain the answer.

> Analysis and summarization of disclosed information only — **not investment
> advice**. Enforced in the system prompt, asserted by tests, and stamped on
> every response via the `disclaimer` field.

This is the platform's **reference module**: entry point, contract, eval
harness, and docs here set the bar for every later service (see root
[AGENTS.md](../../AGENTS.md)).

## Run it

```bash
# from repo root, once:
python3 -m venv .venv && source .venv/bin/activate

cd services/rag-service
pip install -r requirements.txt   # run from here (relative -e path); installs packages/shared editable
cp ../../.env.example .env        # set GEMINI_API_KEY for generated answers

# one-shot demo (no server)
python api.py --demo "How fast did Meridian's data center segment grow?"

# serve (ingests fixtures into ./.chroma on first start)
python api.py --port 8000
curl -X POST localhost:8000/v1/query -H 'Content-Type: application/json' \
  -d '{"question": "What revenue guidance did Northwind give for Q4?"}'
```

**No API key?** `FMA_EMBEDDING_PROVIDER=local python api.py` runs retrieval
fully offline (bundled ONNX MiniLM). `POST /v1/query` with
`"generate_answer": false` returns citations; generation returns 503 until an
LLM key is set.

## Configuration

All via env / `.env` (loaded from the directory you run in):

| var | default | notes |
|---|---|---|
| `FMA_LLM_PROVIDER` | `gemini` | `gemini` \| `openai` \| `anthropic` |
| `FMA_LLM_MODEL` | `gemini-2.5-flash` | |
| `FMA_EMBEDDING_PROVIDER` | `gemini` | `gemini` \| `openai` \| `local` (keyless) |
| `FMA_EMBEDDING_MODEL` | `gemini-embedding-001` | re-ingest (`--reingest`) after changing |
| `GEMINI_API_KEY` etc. | — | key for the chosen provider(s) |
| `RAG_DATA_DIR` | `data/fixtures` | corpus directory (frontmatter-tagged `.md`) |
| `RAG_CHROMA_DIR` | `.chroma` | vector store location |

## API contract (what the orchestrator consumes)

Defined in `packages/shared/shared/contracts.py`; this service just serves it.

- `GET /healthz`
- `GET /v1/tool-schema` → `ToolDefinition` (`query_financial_documents`) —
  register directly as an LLM tool.
- `POST /v1/query` → `RagQueryRequest {question, top_k=5, generate_answer=true}`
  → `RagQueryResponse {answer, citations[], model, disclaimer}`.
  `generate_answer=false` skips the LLM (citations only). 503 = no LLM
  configured; 502 = provider error.
- `POST /v1/documents` → multipart file upload (field name `file`; `.pdf`,
  `.md`, or `.txt`, 20MB max) → `IngestStats {documents, chunks}`. Runs the
  file through the same chunk/embed/upsert pipeline as fixture ingest; it's
  queryable immediately, and also written to `RAG_DATA_DIR` so `--reingest`
  keeps it (DECISIONS.md #13). Not part of the two-contract orchestrator
  surface — an operator/admin path, not an LLM tool. If `RAG_UPLOAD_TOKEN` is
  set, requires a matching `X-Upload-Token` header (401 otherwise); unset
  means open, like every other endpoint here. See DECISIONS.md #12.
  chatbot-orchestrator's chat UI proxies to this endpoint via its own
  `POST /v1/documents` (same-origin, no CORS needed) — the "Upload" button
  next to the message box.

## Evaluation

```bash
python -m eval.run_eval             # retrieval metrics (keyless with FMA_EMBEDDING_PROVIDER=local)
python -m eval.run_eval --answers   # + faithfulness (LLM judge) and abstention checks
```

14-question fixture dataset (`eval/dataset.jsonl`): 12 answerable with gold
spans, 2 unanswerable (must abstain). Full report: `eval/results.md`.

### Results — `gemini-embedding-001` retrieval, k=5

| metric | value |
|---|---|
| hit@1 | 0.67 |
| hit@3 | 1.00 |
| hit@5 | 1.00 |
| MRR | 0.83 |

Local ONNX baseline (all-MiniLM-L6-v2, keyless) for comparison: hit@1 0.67,
hit@3 0.83, hit@5 1.00, MRR 0.78. Two questions (q03 gross margin, q05 China
exposure, q11 install guidance) only land at rank 2 with Gemini embeddings —
still within k=5, no hybrid/rerank trigger yet (DECISIONS.md #5).

### Answer quality (partial — free-tier daily quota, see DECISIONS.md #11)

Generator/judge: `gemini-2.5-flash-lite`. 6 of 12 answerable questions judged
faithful before hitting Google's 20-req/day free-tier cap: **6/6 faithful**,
0 wrong abstentions. Abstention correctness (q13/q14) not yet run. Full
report: `eval/results.md`.

## Layout

```
api.py            # standalone entry: server, --demo, --reingest
service.py        # RagService: retrieve -> prompt -> grounded answer (DI for tests/eval)
prompts.py        # system prompt (grounding + no-advice), abstain phrase
ingest/           # frontmatter loader, paragraph-aware chunker, embed+upsert pipeline
retrieval/        # Chroma store glue + dense retriever (hybrid/rerank seam documented)
eval/             # dataset.jsonl, metrics, LLM judge, run_eval, results.md
data/fixtures/    # synthetic 10-K/10-Q/transcript corpus (labeled synthetic)
tests/            # 17 offline tests, injected fakes, no keys
```
