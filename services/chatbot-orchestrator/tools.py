"""Tool discovery and invocation against downstream services.

Each downstream service exposes GET /v1/tool-schema (-> ToolDefinition) and
POST /v1/query. The orchestrator discovers tools at startup and, at call
time, POSTs the LLM-chosen arguments straight through -- those arguments are
expected to validate against the tool's own input_schema (the service's
pydantic model), so no local duplicate of that schema lives here.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from shared.contracts import ToolDefinition
from shared.logging import get_logger

log = get_logger("orchestrator.tools")

_TIMEOUT = 30.0


class ToolUnavailable(RuntimeError):
    """Raised when the service backing a chosen tool can't be reached."""


class ToolSource(Protocol):
    def definitions(self) -> list[ToolDefinition]: ...
    def invoke(self, name: str, arguments: dict) -> dict: ...


class ToolRegistry:
    """Maps a discovered tool name to its (ToolDefinition, base_url)."""

    def __init__(self, entries: dict[str, tuple[ToolDefinition, str]]) -> None:
        self._entries = entries

    def __bool__(self) -> bool:
        return bool(self._entries)

    def definitions(self) -> list[ToolDefinition]:
        return [definition for definition, _ in self._entries.values()]

    def invoke(self, name: str, arguments: dict) -> dict:
        if name not in self._entries:
            raise ToolUnavailable(f"unknown tool: {name}")
        _, base_url = self._entries[name]
        try:
            resp = httpx.post(f"{base_url}/v1/query", json=arguments, timeout=_TIMEOUT)
        except httpx.TransportError as exc:
            raise ToolUnavailable(f"{name} service unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise ToolUnavailable(f"{name} returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()


def discover_tools(service_urls: dict[str, str]) -> ToolRegistry:
    """GET /v1/tool-schema from each service URL; skip ones that fail (degraded mode)."""
    entries: dict[str, tuple[ToolDefinition, str]] = {}
    for label, base_url in service_urls.items():
        try:
            resp = httpx.get(f"{base_url}/v1/tool-schema", timeout=_TIMEOUT)
            resp.raise_for_status()
            tool = ToolDefinition.model_validate(resp.json())
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("tool discovery failed for %s (%s): %s", label, base_url, exc)
            continue
        entries[tool.name] = (tool, base_url)
        log.info("discovered tool %r from %s", tool.name, label)
    return ToolRegistry(entries)
