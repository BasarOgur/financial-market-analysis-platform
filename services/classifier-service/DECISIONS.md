# Architecture Decisions — classifier-service

Each entry: the choice, the rejected alternative, and why. Rule 7 in AGENTS.md.

## 1. Trained-classifier path = logistic regression on shared embeddings, not fine-tuning

**Chose:** scikit-learn `LogisticRegression` heads (one multinomial for
sentiment, one-vs-rest per topic) on top of embeddings from
`shared/embeddings.py`.
**Rejected:** fine-tuning a transformer (LoRA on a small model, or a hosted
fine-tune API).
**Why:** with 48 training examples, fine-tuning overfits and costs GPU time /
API money; a linear head on frozen embeddings is the standard small-data
recipe and trains in under a second at startup — no model artifact to
version or ship. It also runs fully keyless via the `local` embedding
provider, which fine-tuning can't. Revisit when the dataset grows past a few
thousand examples; the `Classifier` protocol means the swap is internal to
`model/embed_path.py`.

## 2. Two paths behind one `Classifier` protocol

**Chose:** `EmbeddingClassifier` and `LLMClassifier` both expose
`classify(texts) -> list[Prediction]` plus a `model` string; `service.py`
dispatches on `request.path` via a dict.
**Rejected:** separate endpoints per path, or auto-fallback from LLM to embed.
**Why:** one contract endpoint keeps the orchestrator integration identical to
rag-service; the caller picks the path explicitly. Auto-fallback would
silently change model quality mid-batch — worse than an honest 503.

## 3. Additions to `shared/contracts.py` (no interface changes)

**Chose:** added `NewsClassification`, `ClassifyRequest`, `ClassifyResponse`,
`CLASSIFIER_TOOL`, and the `SENTIMENTS`/`TOPICS` vocabularies. Nothing
existing was touched.
**Rejected:** defining these models inside the service.
**Why:** AGENTS.md rule 5 — the contract is what the orchestrator imports, so
it must live in the shared package like `RagQueryRequest` does. The label
vocabularies live there too because they *are* the contract: `Literal` types
on the response mean an off-vocabulary label is a validation error, not a
silent string.

## 4. scikit-learn as a new service dependency

**Chose:** `scikit-learn>=1.5` in this service's requirements only (not
`packages/shared`).
**Rejected:** hand-rolling logistic regression with numpy.
**Why:** class-weighted multinomial LR with proper convergence is ~200 lines
of numpy nobody should re-audit; sklearn is boring and already ubiquitous.
It stays out of `packages/shared` because no other service needs it —
shared only carries cross-service surface.

## 5. Fixed few-shot examples in the prompt, not retrieved ones

**Chose:** 5 hand-picked examples hard-coded in `prompts.py`, using fictional
companies that do not appear in the fixture dataset.
**Rejected:** dynamically retrieving nearest-neighbor examples per input.
**Why:** dynamic few-shot needs an embedding lookup inside the LLM path,
coupling the two paths and muddying the eval (the "zero/few-shot baseline"
would quietly depend on the trained path's embeddings). Fixed examples keep
the baseline honest. Kept out of the dataset so eval never scores an item
the prompt already contains.

## 6. Synthetic labeled dataset, 72 items, fixed train/test split

**Chose:** hand-written snippets for 8 fictional companies; `split` field in
the JSONL (48 train / 24 test); every topic ≥4 times in test, sentiment
balanced 8/8/8 in test.
**Rejected:** scraping real headlines; random split at runtime.
**Why:** same reasoning as rag-service decision 7 (reproducible, no licensing,
eyeball-able labels) — and a *stored* split means the eval table is stable
run to run instead of drifting with a seed. Balance is enforced by
construction so macro-F1 isn't dominated by one label.

## 7. Hand-rolled precision/recall/F1 instead of `sklearn.metrics`

**Chose:** ~40 lines in `eval/metrics.py` scoring both tasks as label sets
(sentiment = one-element set).
**Rejected:** `sklearn.metrics.classification_report`.
**Why:** counter-intuitive given decision 4, but: one code path scores both
the single-label and multi-label task, the eval stays runnable if someone
strips sklearn later (it's only needed by the embed path), and the tie-out
tests in `test_metrics.py` are trivial to verify by hand. `classification_report`
returns formatted strings we'd have to re-parse for the markdown report anyway.

## 8. LLM output coercion instead of raising

**Chose:** `parse_reply` never raises: no JSON → neutral/`other`; invalid
sentiment → neutral; off-vocabulary topics dropped; empty topics → `other`.
Every coercion is logged.
**Rejected:** raising on malformed output (fail the batch), or re-prompting
until valid.
**Why:** one bad generation shouldn't kill a 50-item batch, and re-prompt
loops multiply cost against free-tier quotas. Coercion-to-default is
*visible* in the eval as errors on those items — the honest outcome. The
logs make the rate inspectable.

## 9. LLM-path eval constrained by free-tier quota (same as rag DECISIONS #11)

The full LLM eval is 24 calls; Google's free tier caps `generate_content` at
20/day per model (and this project shares that quota with rag-service). Two
practical findings from the runs:

- `gemini-2.5-flash` with `max_tokens=128` returned truncated JSON
  (`'{"sentiment": "bull'`) — thinking models spend the budget on reasoning
  tokens first. Raised to 1024 in `model/llm_path.py`.
- `eval/run_eval.py --limit N` exists for quota-sized partial runs.
- Two multi-minute runs died to a transient DNS failure (`[Errno 8] nodename
  nor servname provided`) partway through their backoff waits. Fixed in
  `shared/llm.py`: `httpx.TransportError` now retries on the same backoff
  ladder instead of raising immediately — internal behavior change only, no
  interface change.

This is a real constraint, not a bug. Resolution (2026-07-09, after quota
reset): `--llm --limit 20` completed cleanly on `gemini-2.5-flash-lite` with
zero coercion warnings. Result: few-shot LLM sentiment macro F1 0.95 /
topics 0.91 vs trained head 0.65 / 0.79 on the same subset — the expected
outcome at 48 training examples (see #1 for when the trained path wins:
keyless, offline, cheap at volume, improves with data).
