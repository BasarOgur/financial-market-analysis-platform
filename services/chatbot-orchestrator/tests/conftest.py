"""Deterministic fakes so tests run offline with no API keys or network."""

from __future__ import annotations

import pytest

from shared.contracts import CLASSIFIER_TOOL, RAG_TOOL


class ScriptedLLM:
    """Returns canned replies in order; records prompts/system for assertions."""

    model = "fake-llm"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=1024):
        self.calls.append({"prompt": prompt, "system": system})
        if not self._replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


class FakeToolRegistry:
    """Duck-types tools.ToolSource without touching the network."""

    def __init__(self, defs, result=None, error=None) -> None:
        self._defs = defs
        self._result = result if result is not None else {"ok": True}
        self._error = error
        self.invoked = None

    def __bool__(self) -> bool:
        return bool(self._defs)

    def definitions(self):
        return self._defs

    def invoke(self, name, arguments):
        self.invoked = (name, arguments)
        if self._error:
            raise self._error
        return self._result


@pytest.fixture
def fake_tools():
    return FakeToolRegistry([RAG_TOOL, CLASSIFIER_TOOL])


@pytest.fixture
def empty_tools():
    return FakeToolRegistry([])
