# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Monorepo of independently runnable AI services analyzing financial
disclosures (SEC filings, earnings-call transcripts, market data).
Positioning: every service **summarizes/analyzes** disclosed information —
nothing here is investment advice. Enforced in system prompts, asserted by
tests, and stamped on every response via `shared.contracts.DISCLAIMER`.

Full contributor rules live in [AGENTS.md](AGENTS.md) — read it, it governs
this repo and is not repeated here. Current module status: [PROGRESS.md](PROGRESS.md).

## Layout

- `packages/shared` — the only shared code: provider-agnostic LLM/embedding
  clients (`shared/llm.py`, `shared/embeddings.py`), env-driven config
  (`shared/config.py`), logging, and inter-module pydantic contracts
  (`shared/contracts.py`). Installed editable (`pip install -e`); this is
  the one package actually pip-installed in the venv.
- `services/rag-service` — **reference module**. Grounded Q&A over SEC
  filings/transcripts with citations and abstention. Its structure
  (standalone entry point, contract endpoints, eval harness, README +
  DECISIONS.md) is the quality bar every other module matches.
- `services/classifier-service` — news sentiment + multi-label topic
  classifier (trained-embedding path and few-shot-LLM path).
- `services/chatbot-orchestrator` — conversational front end; discovers and
  calls the other services as LLM tools.
- `services/market-data-service` — structured market/fundamentals lookups.

Each service directory: `api.py` (standalone entry point — server + `--demo`
mode), `service.py` (core logic, dependency-injected for tests/eval),
`prompts.py`, `tests/`, `eval/` (`run_eval.py`, `dataset.jsonl`,
`results.md`), `README.md`, `DECISIONS.md`.

## Architecture rules (see AGENTS.md for full detail)

- **Every service runs standalone** — `python api.py` works from inside that
  service's own directory with zero other services running, degrading
  gracefully with no API keys.
- **Two-contract integration surface only**: every service exposes
  `GET /v1/tool-schema` (returns a `ToolDefinition` an orchestrator can
  register as an LLM tool) and `POST /v1/query` (typed pydantic
  request/response). Services never talk to each other any other way.
- **Provider calls only through `packages/shared`** — no vendor SDK or raw
  provider HTTP call inside a service. Providers/models are chosen via
  `FMA_*` env vars, never hardcoded.
- **Services are not pip-installed**, run in place from their own directory
  (keeps generic names like `ingest`, `retrieval`, `api` out of the shared
  venv). Only `packages/shared` is installed editable.
- **DECISIONS.md discipline**: every non-obvious choice recorded there with
  the rejected alternative and why. Update the entry if you deviate from it.

## Common commands

One `.venv` at repo root, shared across services:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Per-service setup/run (same pattern for all four services, from inside the
service directory):

```bash
cd services/<service-name>
pip install -r requirements.txt     # -e ../../packages/shared plus service deps
cp ../../.env.example .env          # set GEMINI_API_KEY (or OPENAI_/ANTHROPIC_) for generation
python api.py --demo "<prompt>"     # one-shot demo, no server
python api.py --port 8000           # serve
pytest                              # offline unit tests, injected fakes, no network/keys
python -m eval.run_eval             # eval harness (keyless with FMA_EMBEDDING_PROVIDER=local)
```

Keyless/offline development: `FMA_EMBEDDING_PROVIDER=local` runs embeddings
fully offline (bundled ONNX MiniLM); generation endpoints return 503 until an
LLM key is configured. Run a single test file/case the normal pytest way
(`pytest tests/test_service.py::test_name`) from inside the service directory.

## Provider selection

Config-driven via env vars (`.env` in the directory you run from, or real env
vars) — never edit code to switch providers.

| var | values | default |
|---|---|---|
| `FMA_LLM_PROVIDER` | `gemini` \| `openai` \| `anthropic` | `gemini` |
| `FMA_LLM_MODEL` | — | `gemini-2.5-flash` |
| `FMA_EMBEDDING_PROVIDER` | `gemini` \| `openai` \| `local` | `gemini` |
| `FMA_EMBEDDING_MODEL` | — | `gemini-embedding-001` |

## Context management (from AGENTS.md, worth repeating)

- After pulling Context7 docs for a library, extract only the needed API
  signatures into DECISIONS.md — treat the raw Context7 result as disposable.
- Run `/compact` after finishing a major step (module scaffold, service
  build, eval harness) — not mid-step. Compact before starting a new module.
