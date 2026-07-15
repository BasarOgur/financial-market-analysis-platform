from fastapi.testclient import TestClient

from shared.contracts import ORCHESTRATOR_TOOL

from api import create_app
from service import OrchestratorService
from tests.conftest import ScriptedLLM


def test_healthz(fake_tools):
    llm = ScriptedLLM(replies=[])
    service = OrchestratorService(fake_tools, llm)
    with TestClient(create_app(service)) as c:
        assert c.get("/healthz").json() == {"status": "ok"}


def test_root_serves_chat_ui(fake_tools):
    llm = ScriptedLLM(replies=[])
    service = OrchestratorService(fake_tools, llm)
    with TestClient(create_app(service)) as c:
        resp = c.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<title>Financial Assistant</title>" in resp.text


def test_tool_schema_matches_contract(fake_tools):
    llm = ScriptedLLM(replies=[])
    service = OrchestratorService(fake_tools, llm)
    with TestClient(create_app(service)) as c:
        data = c.get("/v1/tool-schema").json()
        assert data["name"] == ORCHESTRATOR_TOOL.name
        assert set(data["input_schema"]["properties"]) == {"message"}


def test_query_endpoint_roundtrip(fake_tools):
    llm = ScriptedLLM(replies=['{"tool": null, "arguments": {}, "direct_answer": "Hi!"}'])
    service = OrchestratorService(fake_tools, llm)
    with TestClient(create_app(service)) as c:
        resp = c.post("/v1/query", json={"message": "hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Hi!"
        assert body["disclaimer"]


def test_query_validation_rejects_bad_input(fake_tools):
    llm = ScriptedLLM(replies=[])
    service = OrchestratorService(fake_tools, llm)
    with TestClient(create_app(service)) as c:
        assert c.post("/v1/query", json={}).status_code == 422
        assert c.post("/v1/query", json={"message": ""}).status_code == 422


def test_llm_unavailable_maps_to_503(fake_tools):
    service = OrchestratorService(fake_tools, llm=None, llm_error="no key")
    with TestClient(create_app(service)) as c:
        resp = c.post("/v1/query", json={"message": "hello"})
        assert resp.status_code == 503


def test_malformed_router_reply_maps_to_502(fake_tools):
    llm = ScriptedLLM(replies=["not json"])
    service = OrchestratorService(fake_tools, llm)
    with TestClient(create_app(service)) as c:
        resp = c.post("/v1/query", json={"message": "hello"})
        assert resp.status_code == 502
