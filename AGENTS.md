# Agent & Contributor Rules

Rules for anyone (human or agent) working in this repo. `rag-service` is the
reference implementation of all of them.

## Architecture

1. **Every service runs standalone.** `python api.py` from the service
   directory must work with no orchestrator, no other service, and — in
   degraded mode — no API keys.
2. **Contract endpoints are mandatory.** Every service exposes
   `GET /v1/tool-schema` (a `shared.contracts.ToolDefinition`) and
   `POST /v1/query` with typed pydantic request/response models living in
   `packages/shared/shared/contracts.py`. That is the only integration
   surface; nothing else may be assumed by other modules.
3. **Provider calls only through `packages/shared`.** No vendor SDK or raw
   provider HTTP call inside a service. Providers/models are selected via env
   config (`FMA_*` vars), never hardcoded.
4. **Services are not pip-installed.** They run in place from their own
   directory (keeps generic top-level names out of the shared venv). Reusable
   code goes in `packages/shared`, which IS installed (editable).

## Quality bar (what "done" means for a module)

5. **Eval harness before merge.** A module ships with a runnable eval
   (`python -m eval.run_eval`), a fixture dataset, and real numbers in its
   README. Placeholder tests don't count.
6. **Tests run offline.** Unit tests use injected fakes, no network, no keys.
7. **DECISIONS.md discipline.** Every non-obvious choice is recorded with the
   rejected alternative and why. If you deviate from a recorded decision,
   update the record.
8. **README per service:** how to run, config table, API contract, eval
   results table.

## Product constraints

9. **Analysis, not advice.** No output may recommend buying/selling/holding
   or give price targets. System prompts must instruct summarization of
   disclosures instead; responses carry the shared disclaimer
   (`shared.contracts.DISCLAIMER`).
10. **Grounding is mandatory.** Generation answers only from retrieved
    context and abstains with the exact phrase in `prompts.ABSTAIN_PHRASE`
    when the context lacks the answer (the eval asserts this).

## Process

11. **YAGNI.** No auth, UI, multi-user, or speculative abstraction until a
    module needs it. Extension points are documented seams (see
    `retrieval/retriever.py`), not built-out frameworks.
12. **Dependencies are justified.** New runtime dependency = DECISIONS.md
    entry. Prefer stdlib, then already-installed deps.
13. **Library APIs from current docs** (Context7 MCP or official docs), not
    from memory.
14. **Update PROGRESS.md** at the end of any working session: what's done,
    what was attempted and deferred, and why.

## Context management

- After pulling Context7 docs for a library, extract only the needed API
  signatures/patterns into DECISIONS.md, then treat the raw Context7 result
  as disposable.
- Run /compact after finishing each major step (module scaffold, service
  build, eval harness) and before starting the next one — not mid-step.
- Before starting a new /goal for a new module, always /compact first.
