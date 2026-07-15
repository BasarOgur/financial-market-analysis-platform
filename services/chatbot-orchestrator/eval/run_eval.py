"""Eval harness for chatbot-orchestrator. Run from the service root:

    python -m eval.run_eval

Measures tool-selection accuracy only (the routing decision), not full
round-trip quality -- rag-service and classifier-service already have their
own evals for that. Downstream tools are stubbed (no network, no ports
required); only the router LLM call needs a real key.

Writes eval/results.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.config import load_settings
from shared.contracts import CLASSIFIER_TOOL, RAG_TOOL
from shared.llm import LLMError, llm_from_settings
from shared.logging import get_logger, setup_logging

from service import OrchestratorService

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results.md"

log = get_logger("orchestrator.eval")


class _StubTools:
    """Only definitions() is exercised by routing; invoke() is never reached
    because the eval measures the routing decision, not tool execution."""

    def definitions(self):
        return [RAG_TOOL, CLASSIFIER_TOOL]

    def invoke(self, name, arguments):
        raise AssertionError("eval measures routing only; invoke() should not be called")


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in (HERE / "dataset.jsonl").read_text().splitlines() if line.strip()]


def run(limit: int | None = None) -> str:
    settings = load_settings()
    setup_logging(settings.log_level)
    llm = llm_from_settings(settings)  # raises LLMError if unconfigured
    service = OrchestratorService(_StubTools(), llm)

    dataset = load_dataset()
    if limit:
        dataset = dataset[:limit]

    rows = []
    correct = 0
    for ex in dataset:
        decision = service.route(ex["message"])
        picked = decision.get("tool")
        ok = picked == ex["expected_tool"]
        correct += ok
        rows.append({"id": ex["id"], "message": ex["message"], "expected": ex["expected_tool"],
                      "picked": picked, "ok": ok})
        log.info("%s expected=%s picked=%s ok=%s", ex["id"], ex["expected_tool"], picked, ok)

    accuracy = correct / len(dataset)
    detail = "\n".join(
        f"| {r['id']} | {r['message'][:60]} | {r['expected'] or 'none'} | {r['picked'] or 'none'} | {'✅' if r['ok'] else '❌'} |"
        for r in rows
    )
    report = f"""# chatbot-orchestrator eval results

Router model: `{llm.model}`. Tool-selection accuracy over {len(dataset)} messages \
(downstream services stubbed -- this measures routing only).

| metric | value |
|---|---|
| tool-selection accuracy | {correct}/{len(dataset)} = {accuracy:.2f} |

| id | message | expected | picked | ok |
|---|---|---|---|---|
{detail}
"""
    RESULTS_PATH.write_text(report)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N dataset messages")
    args = parser.parse_args()
    try:
        print(run(limit=args.limit))
    except LLMError as exc:
        raise SystemExit(f"orchestrator eval needs a configured LLM: {exc}") from exc
    print(f"written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
