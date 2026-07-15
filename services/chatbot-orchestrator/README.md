# chatbot-orchestrator

Single conversational entry point over the platform's tool services. Discovers
`query_financial_documents` (rag-service) and `classify_financial_news`
(classifier-service) via their `/v1/tool-schema` contracts, routes each user
message to the right tool (or answers directly), and produces one final
natural-language answer.

> Analysis/summarization only — **not investment advice**. Enforced in the
> answer-generation system prompt and asserted in tests, stamped on every
> response via the `disclaimer` field.

See [AGENTS.md](../../AGENTS.md) for the platform-wide rules this follows, and
[DECISIONS.md](DECISIONS.md) for why routing is prompt-engineered JSON rather
than native provider function-calling.

## Run it

```bash
# repo root, once: python3 -m venv .venv && source .venv/bin/activate

# start the two tool services first (separate terminals)
cd services/rag-service && python api.py --port 8000
cd services/classifier-service && python api.py --port 8001

# then the orchestrator
cd services/chatbot-orchestrator
pip install -r requirements.txt   # installs packages/shared editable too
cp ../../.env.example .env        # set GEMINI_API_KEY for routing + answers

# one-shot demo (no server)
python api.py --demo "How fast did Meridian's data center segment grow?"

# serve
python api.py --port 8002
curl -X POST localhost:8002/v1/query -H 'Content-Type: application/json' \
  -d '{"message": "How fast did Meridian'"'"'s data center segment grow?"}'

# or open the chat UI in a browser
open http://localhost:8002/
```

**Downstream service not running?** Its tool is simply not discovered at
startup (logged as a warning) — the orchestrator still starts and routes to
whatever tools it found, or answers directly if none. **No LLM key?** Routing
and answering both need one; `/v1/query` returns 503 until a key is set (there
is no keyless path here since there's no local-embedding equivalent for
routing decisions).

## Configuration

All via env / `.env` (loaded from the directory you run in):

| var | default | notes |
|---|---|---|
| `FMA_RAG_URL` | `http://localhost:8000` | rag-service base URL |
| `FMA_CLASSIFIER_URL` | `http://localhost:8001` | classifier-service base URL |
| `FMA_MARKET_DATA_URL` | `http://localhost:8003` | market-data-service base URL |
| `FMA_LLM_PROVIDER` | `gemini` | `gemini` \| `openai` \| `anthropic` |
| `FMA_LLM_MODEL` | `gemini-2.5-flash` | used for both routing and final answers |
| `GEMINI_API_KEY` etc. | — | key for the chosen provider |

## API contract

Defined in `packages/shared/shared/contracts.py`; this service just serves it.

- `GET /healthz`
- `GET /v1/tool-schema` → `ToolDefinition` (`ask_financial_assistant`) — register as an LLM tool.
- `POST /v1/query` → `OrchestratorQueryRequest {message}` → `OrchestratorQueryResponse {answer, tool_used, tool_result, model, disclaimer}`.
  503 = no LLM configured; 502 = router returned malformed JSON or a provider error.
- `GET /` → simple chat web UI (`static/index.html`), talks to `/v1/query` from the browser.

## Evaluation

```bash
python -m eval.run_eval
```

Measures **tool-selection accuracy only** (the routing decision) over a
10-message fixture dataset (`eval/dataset.jsonl`) — downstream tools are
stubbed so this needs an LLM key but no running rag-service/classifier-service.
Full report: `eval/results.md`. rag-service and classifier-service already
cover retrieval/answer/classification quality in their own evals; this one
only checks that the right tool (or none) gets picked.

## Tests

```bash
pytest
```

Offline, no network, no keys — downstream tool calls are faked
(`tests/conftest.py::FakeToolRegistry`), the LLM is scripted
(`ScriptedLLM`).
