# Financial Market-Analysis AI Platform

A monorepo of independently runnable AI services for analyzing financial
disclosures — SEC filings, earnings-call transcripts, and (later) market data.

> **Positioning:** every service in this platform summarizes and analyzes
> disclosed information. Nothing here is investment advice, and the prompts,
> API responses, and evals are built to enforce that.

## Module map

| Module | Status | Purpose |
|---|---|---|
| `packages/shared` | ✅ done | Provider-agnostic LLM + embedding clients, config, logging, inter-module contracts |
| `services/rag-service` | ✅ done (reference module) | Grounded Q&A over SEC filings / earnings transcripts, with eval harness |
| `services/classifier-service` | ✅ done | News sentiment + multi-label topic classifier; trained-vs-few-shot paths, with eval harness |
| `services/chatbot-orchestrator` | ✅ done | Conversational front end that discovers and calls services as LLM tools |
| `services/market-data-service` | ✅ done | Structured market/fundamentals retrieval |

`rag-service` is the reference module: its structure (standalone entry point,
contract endpoints, eval harness, README + DECISIONS.md) is the quality bar
every later module must match. See [AGENTS.md](AGENTS.md) for the rules and
[PROGRESS.md](PROGRESS.md) for status.

## How modules compose

Every service is standalone (`python api.py` just works) and exposes the same
two-contract surface defined in `packages/shared/shared/contracts.py`:

- `GET /v1/tool-schema` — an LLM-tool-style definition (name, description,
  JSON input schema) the future orchestrator registers directly with its LLM.
- `POST /v1/query` — the typed request/response pair described by that schema.

New services plug in by implementing the same pair — no shared-code changes.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
cd services/rag-service
pip install -r requirements.txt     # run from the service dir; also installs packages/shared (editable)
cp ../../.env.example .env                              # add GEMINI_API_KEY for full answers
python api.py --demo "How fast did Meridian's data center segment grow?"
python api.py                                           # serve on :8000
python -m eval.run_eval                                 # retrieval eval (keyless with FMA_EMBEDDING_PROVIDER=local)
```

No API key? `FMA_EMBEDDING_PROVIDER=local` runs retrieval fully offline;
generation endpoints return 503 until an LLM key is configured.

## Provider selection

LLM and embedding providers are config-driven (`.env` / env vars), no code
changes to switch: `gemini` (default), `openai`, `anthropic` for generation;
`gemini` (default, `text-embedding-004`), `openai`, `local` for embeddings.
