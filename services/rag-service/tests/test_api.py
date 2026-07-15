import pytest
from fastapi.testclient import TestClient

from shared.contracts import RAG_TOOL

from api import create_app
from retrieval.retriever import Retriever
from service import RagService
from tests.test_retriever import _ingest


@pytest.fixture
def client(eval_collection, fake_embedder, fake_llm):
    _ingest(eval_collection, fake_embedder, {"doc": "revenue grew 18% in fiscal 2025"})
    service = RagService(Retriever(eval_collection, fake_embedder), fake_llm)
    with TestClient(create_app(service)) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_tool_schema_matches_contract(client):
    data = client.get("/v1/tool-schema").json()
    assert data["name"] == RAG_TOOL.name
    assert set(data["input_schema"]["properties"]) == {"question", "top_k", "generate_answer"}
    assert data["input_schema"]["required"] == ["question"]


def test_query_endpoint_roundtrip(client):
    resp = client.post("/v1/query", json={"question": "how much did revenue grow"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] and body["citations"] and body["disclaimer"]


def test_query_validation_rejects_bad_input(client):
    assert client.post("/v1/query", json={"question": "hi", "top_k": 99}).status_code == 422
    assert client.post("/v1/query", json={}).status_code == 422


def test_generation_unavailable_maps_to_503(eval_collection, fake_embedder):
    _ingest(eval_collection, fake_embedder, {"doc": "text"})
    service = RagService(Retriever(eval_collection, fake_embedder), llm=None, llm_error="no key")
    with TestClient(create_app(service)) as c:
        resp = c.post("/v1/query", json={"question": "anything here"})
        assert resp.status_code == 503
        # citation-only still works keyless
        ok = c.post("/v1/query", json={"question": "anything here", "generate_answer": False})
        assert ok.status_code == 200 and ok.json()["answer"] is None
