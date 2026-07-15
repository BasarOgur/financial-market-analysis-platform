import httpx
import pytest

from shared.contracts import RAG_TOOL

from tools import ToolRegistry, ToolUnavailable, discover_tools


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code != 200:
            raise httpx.HTTPStatusError("bad status", request=None, response=self)


def test_discover_tools_registers_reachable_services(monkeypatch):
    def fake_get(url, timeout=None):
        assert url == "http://rag.test/v1/tool-schema"
        return _FakeResponse(json_body=RAG_TOOL.model_dump())

    monkeypatch.setattr(httpx, "get", fake_get)
    registry = discover_tools({"rag": "http://rag.test"})

    assert registry
    assert [d.name for d in registry.definitions()] == [RAG_TOOL.name]


def test_discover_tools_skips_unreachable_service(monkeypatch):
    def fake_get(url, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)
    registry = discover_tools({"rag": "http://rag.test", "classifier": "http://classifier.test"})

    assert not registry
    assert registry.definitions() == []


def test_invoke_posts_arguments_to_backing_service(monkeypatch):
    registry = ToolRegistry({RAG_TOOL.name: (RAG_TOOL, "http://rag.test")})

    def fake_post(url, json=None, timeout=None):
        assert url == "http://rag.test/v1/query"
        assert json == {"question": "hi"}
        return _FakeResponse(json_body={"answer": "hi back"})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = registry.invoke(RAG_TOOL.name, {"question": "hi"})
    assert result == {"answer": "hi back"}


def test_invoke_unknown_tool_raises():
    registry = ToolRegistry({})
    with pytest.raises(ToolUnavailable):
        registry.invoke("nope", {})


def test_invoke_maps_transport_error_to_tool_unavailable(monkeypatch):
    registry = ToolRegistry({RAG_TOOL.name: (RAG_TOOL, "http://rag.test")})

    def fake_post(url, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ToolUnavailable):
        registry.invoke(RAG_TOOL.name, {"question": "hi"})
