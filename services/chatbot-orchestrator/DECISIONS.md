# Architecture Decisions — chatbot-orchestrator

Each entry: choice, rejected alternative, why. Rule 7 in [AGENTS.md](../../AGENTS.md).

## 1. Prompt-engineered JSON routing instead of native provider function-calling

**Chose:** ask the LLM (via the existing `shared.llm.LLMClient.generate()`) to return a JSON object
`{"tool": ..., "arguments": ..., "direct_answer": ...}`, same style as classifier-service's
`llm_path` few-shot JSON classification.

**Rejected:** each provider's native tool-calling API (Gemini function calling, OpenAI tools,
Anthropic tool_use).

**Why:** the three native APIs have three different request/response shapes for tool
definitions and tool-call results, which would mean adding provider-specific tool-calling
code to `shared/llm.py`'s three ~25-line classes (see rag-service DECISIONS.md #1, same
REST-over-SDK reasoning) instead of the ~20 lines already needed to embed the tool catalogue
in a prompt. The codebase already established the JSON-in-prompt pattern for classifier-service;
reusing it keeps the LLM client provider-agnostic and untouched. Revisit if a provider's tool
calling meaningfully improves argument-schema adherence over prompting.

## 2. Arguments passed straight through to `/v1/query`, no local duplicate schema

**Chose:** the router prompt embeds each tool's real `input_schema` (which is exactly the
downstream service's pydantic model, e.g. `RagQueryRequest.model_json_schema()`); the LLM's
`arguments` JSON is POSTed to `/v1/query` unmodified.

**Rejected:** hand-writing local pydantic models mirroring `RagQueryRequest`/`ClassifyRequest`
in the orchestrator to validate arguments before sending.

**Why:** the schema is already the single source of truth on the serving side (AGENTS.md rule
2 — contract endpoints are the only integration surface). Duplicating it here would drift the
moment either service's request model changes. The downstream service's own FastAPI/pydantic
validation is the enforcement point; a malformed argument just surfaces as a normal HTTP error
from that service, mapped by `ToolRegistry.invoke` into `ToolUnavailable`.

## 3. Tool discovery degrades gracefully; startup never crashes

**Chose:** `discover_tools()` GETs `/v1/tool-schema` from each configured service URL and
skips (logs a warning) any that fail to respond. The orchestrator starts with whatever subset
of tools it found — including zero, in which case the router always answers directly.

**Rejected:** failing orchestrator startup if any downstream service is unreachable.

**Why:** AGENTS.md rule 1 — every service must run standalone, in degraded mode, without
other services present. An orchestrator that refuses to start without both rag-service and
classifier-service up violates that for the one service whose whole job is composing others.

## 4. Two LLM calls (route, then answer) instead of one combined call

**Chose:** a routing call that only decides the tool + arguments, and — once the tool result is
in hand — a second call that turns the raw JSON result into a natural-language answer.

**Rejected:** a single prompt that both picks the tool and (somehow) pre-writes the answer.

**Why:** the answer can only be grounded in the tool's actual result, which doesn't exist until
after the tool runs. Splitting the two concerns also keeps each prompt single-purpose, mirroring
rag-service's separate answer-generation and judge prompts.

## 5. Multi-tool / multi-turn chaining deferred (YAGNI)

**Chose:** v1 handles exactly one tool call per message. If the router picks no tool, or the
chosen tool fails, the orchestrator returns without trying another.

**Rejected:** building a loop that lets the LLM call multiple tools in sequence for one message
(e.g. classify a headline, then look up related filing risk).

**Why:** no eval evidence yet that real questions need more than one tool per turn — same
reasoning as rag-service deferring hybrid retrieval until the eval showed a concrete miss (see
rag-service DECISIONS.md #5). Revisit if the eval dataset or real usage surfaces questions that
a single tool call can't answer.

## 7. Chat UI is one static HTML file served from `/`, not a separate frontend service

**Chose:** `static/index.html` (vanilla JS, inline CSS, no build step, no new dependency)
served via `FileResponse` on `GET /` in `api.py`.

**Rejected:** a separate `services/web-ui` service/process, or a frontend framework/bundler.

**Why:** ponytail rung 6/7 — the whole UI is "post a message, render the JSON reply,"
which is a few dozen lines of plain JS. A second service would mean another port, another
`requirements.txt`, and CORS config for zero benefit at this scale. FastAPI already serves
files; reusing that avoids adding `starlette.staticfiles` or any new package.

## 6. Orchestrator exposes its own `/v1/tool-schema` too

**Chose:** `GET /v1/tool-schema` returns `ORCHESTRATOR_TOOL` (`ask_financial_assistant`), even
though nothing currently calls it as a tool.

**Rejected:** skipping the contract endpoint since the orchestrator has no caller yet.

**Why:** AGENTS.md rule 2 makes the two-endpoint contract mandatory for every service, no
exceptions carved out for "the top of the stack." Keeps the door open for a future
meta-orchestrator or UI to register it the same way it registers rag-service and
classifier-service.
