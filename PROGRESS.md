# Progress

## 2026-07-17 — upload UI + upload persistence fixes

### Done

- **chatbot-orchestrator chat UI** (`static/index.html`): "Upload" button next
  to the message box, `.pdf/.md/.txt` file picker, posts to a new
  `POST /v1/documents` on the orchestrator that proxies straight through to
  rag-service's own upload endpoint (`httpx`, no new dep). Kept same-origin on
  purpose instead of adding CORS to rag-service — see chatbot-orchestrator
  DECISIONS.md #8.
- **rag-service upload persistence**: uploads previously only landed in
  Chroma; a `--reingest` (which rebuilds from `data/fixtures` on disk) would
  silently drop them. Added `ingest/pipeline.py:persist_upload()` — writes the
  uploaded doc as frontmatter-tagged markdown into `RAG_DATA_DIR` right after
  ingesting it, so it survives reingest like any fixture. See rag-service
  DECISIONS.md #13.
- **Bug fix while at it**: `load_documents()` was unconditionally recomputing
  `meta["source"]` from `company`/`doc_type`/`period`/`section`, discarding any
  explicit `source` frontmatter — harmless for fixtures (never set it) but
  would have overwritten an upload's filename with a doc-id-derived value on
  reingest. Now prefers an explicit `source` if present.
- **Test-pollution catch**: the new persistence path wrote real files into
  the repo's `services/rag-service/data/fixtures/` during `pytest` (junk
  `upload-notes-*.md` files) before the `client` fixture was updated to
  monkeypatch `DATA_DIR` to `tmp_path`. Cleaned up and fixed before this was
  committed — flagging in case any earlier run left stray files.
- rag-service: 26/26 tests pass. chatbot-orchestrator: 20/20 tests pass.

## 2026-07-16 — rag-service document upload endpoint

### Done

- **`POST /v1/documents`** on rag-service: multipart file upload (`.pdf`,
  `.md`, `.txt`) → text extraction (`pypdf` for PDF) → same chunk/embed/upsert
  pipeline as fixture ingest → `IngestStats` response. Uploaded docs are
  queryable immediately, no reingest step.
  - `ingest/pipeline.py`: added `extract_text`, `ingest_document`, and a
    shared `_embed_and_upsert` helper factored out of `ingest()`.
  - `Retriever`/`RagService` gained read-only `.collection`/`.embedder` and
    `.retriever` properties so the endpoint can reach the already-open
    Chroma collection without re-plumbing config.
  - New deps: `pypdf` (PDF text extraction), `python-multipart` (FastAPI
    requires it for file uploads).
  - Tests: `tests/test_pipeline.py` (extract_text, ingest_document) +
    `tests/test_api.py` (upload roundtrip incl. retrieval, rejected
    extension). 22/22 rag-service tests pass.
- Resolves the "no document upload" limitation flagged 2026-07-15 below.
  Web chat UI still has no upload button — API-only for now (see PROGRESS
  note in `project_finished_explanation.md` #9).

## 2026-07-15 — Chat UI + .env cleanup

### Done

- **Simple chat web UI**: `services/chatbot-orchestrator/static/index.html`
  (vanilla JS/CSS, no build step, no new dependency), served at `GET /` via
  `FileResponse` (DECISIONS.md #7). Verified live in a real browser: typed a
  question, got a rendered answer with the `get_market_data` tool badge and
  disclaimer footer.
- **`FMA_EMBEDDING_PROVIDER=local` made permanent** in
  `rag-service/.env` and `classifier-service/.env` — no longer needs to be
  set per-command. Retrieval stays local/keyless; LLM generation still uses
  the real Gemini key (the two settings are independent).
- Verified `GEMINI_API_KEY` + `FMA_LLM_MODEL=gemini-2.5-flash-lite` already
  set correctly in all three LLM-using services' `.env` files — no change
  needed there.
- 19/19 orchestrator tests pass (added one for the new `/` route).

### Known limitation (not built, flagged to user)

- ~~**No document upload.**~~ Fixed 2026-07-16 — see `POST /v1/documents`
  above. UI still has no upload button, API-only.

## 2026-07-15 — market-data-service run

### Done

- **`services/market-data-service`**: structured market/fundamentals lookup
  by ticker (price, market cap, P/E, volume) via `yfinance` — free, no API
  key (DECISIONS.md #1).
  - **No LLM anywhere**: pure data retrieval, only two failure modes —
    `TickerNotFound` (404) and `ProviderUnavailable` (503, network/timeout).
  - **Contracts**: added `MarketDataRequest/Response`, `MARKET_DATA_TOOL`,
    and a dedicated `MARKET_DATA_DISCLAIMER` (the shared `DISCLAIMER` text
    talks about "source documents", wrong for raw provider data) to
    `shared/contracts.py` — additive only.
  - **Not-found detection**: yfinance doesn't raise for unknown tickers, it
    returns a near-empty dict; verified this empirically before writing the
    check (DECISIONS.md #4) — missing price field = not found.
  - Client injected via `MarketDataClient` protocol, same DI pattern as
    every other service; 11 offline tests (fake client), all passing.
  - **Eval** (`eval/run_eval.py`, live yfinance, keyless): 7/7 = 1.00 on 5
    known-good + 2 known-bad tickers.
  - **chatbot-orchestrator** wired to discover it too (`FMA_MARKET_DATA_URL`,
    default `:8003`); 18/18 orchestrator tests still pass unchanged.
  - **Live integration demo** (all three services running): "What is
    Apple's current stock price?" → routed to `get_market_data`, correct
    price/market-cap/P-E answer. Regression-checked the existing
    rag-service routing path still works with the third tool present.

### Next up

1. All four planned module-map services are now done. Candidate next steps:
   sentiment/topic time-series aggregation, or a thin UI over the
   orchestrator's `/v1/query`.

## 2026-07-14 — Orchestrator run

### Done

- **`services/chatbot-orchestrator`**: conversational front end. Discovers
  `query_financial_documents` (rag-service) + `classify_financial_news`
  (classifier-service) via `/v1/tool-schema` at startup, routes each message
  to one tool or a direct answer, produces final natural-language answer.
  - **Routing**: prompt-engineered JSON (`{"tool", "arguments",
    "direct_answer"}`), same style as classifier-service's `llm_path`, not
    native per-provider function-calling — keeps `shared/llm.py`
    provider-agnostic (DECISIONS.md #1).
  - **Contracts**: added `OrchestratorQueryRequest/Response` +
    `ORCHESTRATOR_TOOL` to `shared/contracts.py` — additive only.
  - **Degrades gracefully**: missing downstream service → skipped at
    discovery (warning logged), not a crash; missing LLM key → 503 on
    `/v1/query`. No keyless path for routing itself (needs LLM to decide).
  - **Deferred (YAGNI)**: multi-tool/multi-turn chaining — one tool call per
    message in v1 (DECISIONS.md #5). Revisit if eval/usage shows a real need.
  - 18 offline tests (fake tool registry + scripted LLM), all passing.
  - **Eval** (`eval/run_eval.py`, tool-selection accuracy, downstream tools
    stubbed): 10/10 = 1.00 on the 10-message fixture set.
  - **Live integration demo** (rag-service :8000 + classifier-service :8001
    actually running, real Gemini key): 3/3 —
    1. document question → routed to `query_financial_documents`, grounded
       answer with citations.
    2. "Should I buy Meridian stock right now?" → no tool, correctly
       declined investment advice.
    3. headline classification → routed to `classify_financial_news`,
       correct sentiment/topic answer.

### Next up

1. `services/market-data-service` — structured market/fundamentals
   retrieval, third tool for the orchestrator to discover.
2. Real answer-quality eval for rag-service with a Gemini key (still
   pending from the 2026-07-05 run).

## 2026-07-08 — Classifier run

### Done

- **`services/classifier-service`**: financial news classifier — sentiment
  (bullish/bearish/neutral) + multi-label topics (earnings, m&a, regulation,
  macro, product, other). Mirrors rag-service structure (api.py with
  `--demo`/`--batch`, service.py DI, eval harness, README, DECISIONS.md).
  - **Two paths, one interface** (`Classifier` protocol, `path` field on the
    request): `embed` = logistic regression heads trained at startup on
    shared embeddings (keyless-capable via local ONNX); `llm` = few-shot
    JSON classification through the shared LLM client (503 keyless).
  - **Contracts**: added `NewsClassification`/`ClassifyRequest`/
    `ClassifyResponse`/`CLASSIFIER_TOOL` + label vocabularies to
    `shared/contracts.py` — additive only, nothing existing touched
    (classifier DECISIONS.md #3).
  - **Dataset**: 72 synthetic labeled snippets (`data/news.jsonl`), fixed
    48/24 train/test split, every topic ≥4× in test, sentiment balanced.
  - **Eval** (`eval/run_eval.py`, per-label P/R/F1 + macro F1, both paths on
    the same test split): on the shared 20-item subset (`--limit 20`, fits
    the 20-req/day free-tier cap, classifier DECISIONS.md #9) — few-shot LLM
    (gemini-2.5-flash-lite) sentiment macro F1 0.95 / topics 0.91 vs trained
    head 0.65 / 0.79. Full 24-item keyless embed run: 0.67 / 0.77. Expected
    at 48 train examples; trained path earns its keep keyless/offline/at
    volume.
  - 21 offline tests (fake embedder + fake LLM), all passing. Verified:
    keyless `--demo`, server roundtrip (healthz, tool-schema, query, live
    llm-path classify).
- **`shared/llm.py` hardening**: transient network failures
  (`httpx.TransportError` — DNS blips, dropped connections) now retry on the
  existing backoff ladder instead of raising immediately; two multi-minute
  eval runs had died to a single DNS blip. Internal change, no interface
  change. (LLM eval was quota-blocked on 2026-07-08 — both 2.5 models'
  20-req/day free-tier caps exhausted — and completed cleanly after the
  2026-07-09 reset with `--limit 20`.)
- **Toolchain finding**: Python 3.12 `site.py` silently skips `.pth` files
  carrying the macOS `UF_HIDDEN` flag, which the Claude Code sandbox sets on
  files it writes — the editable-install `.pth` for `packages/shared` kept
  going invisible, breaking `import shared`. Durable fix: symlinked the
  `shared` package directly into the venv's site-packages (symlinks bypass
  the hidden-flag check). Re-run `ln -sfn "$(pwd)/packages/shared/shared"
  .venv/lib/python3.12/site-packages/shared` from root if imports break.

### Next up

1. chatbot-orchestrator can now consume both tools (`query_financial_documents`,
   `classify_financial_news`) from `/v1/tool-schema`.

## 2026-07-05 — Foundation run

### Done

- **Monorepo skeleton**: `packages/` (installed, reusable) vs `services/`
  (run-in-place, standalone). Single venv at repo root.
- **`packages/shared`**: provider-agnostic LLM client (gemini | openai |
  anthropic) and embedding client (gemini | openai | local ONNX) over raw
  REST via httpx; env-driven config with tiny `.env` loader; logging setup;
  inter-module contracts (`ToolDefinition`, `RagQueryRequest/Response`,
  shared disclaimer).
- **`services/rag-service`** (reference module):
  - `api.py` standalone entry: FastAPI server, `--demo` one-shot mode,
    `--reingest`; degrades gracefully keyless (retrieval works, generation
    503s with a clear message).
  - `ingest/`: frontmatter-tagged markdown loader, paragraph-aware chunker
    (word-budget packing + sliding-window for oversized paragraphs),
    embed + upsert into persistent Chroma (cosine).
  - `retrieval/`: dense retriever with a documented hybrid/rerank seam.
  - `eval/`: 14-question fixture dataset (12 answerable w/ gold spans, 2
    unanswerable), retrieval metrics (hit@1/3/5, MRR), LLM-judge
    faithfulness + deterministic abstention check, markdown report writer.
  - 6-doc synthetic fixture corpus (2 companies' 10-K excerpts, 10-Q,
    2 earnings-call transcripts), clearly labeled synthetic.
  - 17 offline unit tests (chunker, metrics, retriever, service, API,
    contract), all passing.
- **Eval run (local ONNX embeddings, keyless)**: hit@1 0.67, hit@3 0.83,
  hit@5 1.00, MRR 0.78 — recorded in `services/rag-service/eval/results.md`
  and the service README.
- **Governance**: AGENTS.md (rules 1–14), root README (module map,
  contract), service README + DECISIONS.md (10 recorded decisions).

### Verified end-to-end

- `pytest` 17/17 green (offline, no keys).
- `python api.py --demo ...` keyless: correct top citation.
- Live server round-trip: `/healthz`, `/v1/tool-schema`, `/v1/query`
  (citations-only 200; generation without key 503 as designed).
- `python -m eval.run_eval` keyless produces the numbers above.

### Attempted / deferred (with why)

- **Answer-quality eval numbers (faithfulness/abstention)**: harness is
  implemented and wired (`--answers`) but needs an LLM key; no key was
  available in this environment. Run `GEMINI_API_KEY=... python -m
  eval.run_eval --answers` to fill the answer-quality table.
- **Gemini-embedding retrieval numbers**: same reason; the table currently
  shows the local ONNX baseline. Re-run with `FMA_EMBEDDING_PROVIDER=gemini`.
- **Hybrid search + reranking**: deliberately not built (YAGNI). Eval gives
  the trigger: q04 and q11 land at rank 5 with dense-only retrieval. Seam
  documented in `retrieval/retriever.py` and DECISIONS.md #5.
- **Real EDGAR ingestion**: fixtures are synthetic for reproducibility;
  swapping in real filings is an ingest-only change (DECISIONS.md #7).
- **Other modules** (chatbot-orchestrator, market-data, sentiment): out of
  scope this run per the plan; contract they'll consume is frozen in
  `shared/contracts.py`.

### Next up

1. Run answer eval with a Gemini key; record faithfulness/abstention.
2. chatbot-orchestrator consuming `/v1/tool-schema` (contract already fixed).
3. Hybrid retrieval if eval on a larger corpus confirms dense-only misses.
