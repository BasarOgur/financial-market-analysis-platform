"""OrchestratorService: routes a user message to a tool (or answers directly),
then produces a final natural-language answer.

Dependencies are injected (llm, tools) so api.py wires real providers from
config while tests and the eval harness inject fakes.
"""

from __future__ import annotations

import json

from shared.contracts import OrchestratorQueryRequest, OrchestratorQueryResponse
from shared.llm import LLMClient
from shared.logging import get_logger

from prompts import ANSWER_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT, answer_prompt, router_prompt
from tools import ToolSource, ToolUnavailable

log = get_logger("orchestrator.service")


class LLMUnavailable(RuntimeError):
    """No LLM configured (api.py maps this to 503)."""


class RoutingError(RuntimeError):
    """The router LLM did not return valid routing JSON (api.py maps this to 502)."""


class OrchestratorService:
    def __init__(self, tools: ToolSource, llm: LLMClient | None = None,
                 llm_error: str | None = None) -> None:
        self._tools = tools
        self._llm = llm
        self._llm_error = llm_error or "no LLM configured"

    def query(self, request: OrchestratorQueryRequest) -> OrchestratorQueryResponse:
        if self._llm is None:
            raise LLMUnavailable(self._llm_error)

        decision = self.route(request.message)
        tool_name = decision.get("tool")

        if not tool_name:
            answer = decision.get("direct_answer") or "I don't have an answer for that."
            return OrchestratorQueryResponse(answer=answer, model=self._llm.model)

        try:
            result = self._tools.invoke(tool_name, decision.get("arguments") or {})
        except ToolUnavailable as exc:
            log.warning("tool %s unavailable: %s", tool_name, exc)
            return OrchestratorQueryResponse(
                answer=f"I couldn't reach the {tool_name} service right now. ({exc})",
                tool_used=tool_name,
                model=self._llm.model,
            )

        final = self._llm.generate(
            answer_prompt(request.message, tool_name, result),
            system=ANSWER_SYSTEM_PROMPT,
            temperature=0.1,
        )
        log.info("routed %r to %s", request.message[:60], tool_name)
        return OrchestratorQueryResponse(
            answer=final, tool_used=tool_name, tool_result=result, model=self._llm.model
        )

    def route(self, message: str) -> dict:
        """Ask the LLM which tool (if any) applies. Public so eval can measure
        tool-selection accuracy without invoking downstream services."""
        if self._llm is None:
            raise LLMUnavailable(self._llm_error)
        raw = self._llm.generate(
            router_prompt(message, self._tools.definitions()),
            system=ROUTER_SYSTEM_PROMPT,
            temperature=0.0,
        )
        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError as exc:
            raise RoutingError(f"router returned non-JSON: {raw[:200]}") from exc


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()
