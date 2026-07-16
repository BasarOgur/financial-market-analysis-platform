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


def test_upload_document_ingests_and_is_retrievable(client):
    resp = client.post(
        "/v1/documents",
        files={"file": ("notes.txt", b"Zenith Robotics grew backlog 40% in Q3.", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"] == 1
    assert body["chunks"] == 1

    q = client.post("/v1/query", json={"question": "Zenith backlog growth", "generate_answer": False})
    assert q.status_code == 200
    assert any(c["chunk_id"].startswith("upload-notes") for c in q.json()["citations"])


def test_upload_document_rejects_unsupported_extension(client):
    resp = client.post(
        "/v1/documents",
        files={"file": ("data.csv", b"a,b\n1,2", "text/csv")},
    )
    assert resp.status_code == 400


def test_upload_different_content_same_filename_does_not_collide(client):
    r1 = client.post("/v1/documents", files={"file": ("notes.txt", b"first document body", "text/plain")})
    r2 = client.post("/v1/documents", files={"file": ("notes.txt", b"second document body", "text/plain")})
    assert r1.status_code == r2.status_code == 200

    q = client.post("/v1/query", json={"question": "document body", "top_k": 5, "generate_answer": False})
    chunk_ids = {c["chunk_id"] for c in q.json()["citations"]}
    upload_ids = {cid for cid in chunk_ids if cid.startswith("upload-notes")}
    assert len(upload_ids) == 2  # both kept, neither silently overwrote the other


def test_upload_requires_token_when_configured(client, monkeypatch):
    import api as api_module

    monkeypatch.setattr(api_module, "UPLOAD_TOKEN", "secret123")
    files = {"file": ("notes.txt", b"some content", "text/plain")}

    assert client.post("/v1/documents", files=files).status_code == 401
    assert client.post(
        "/v1/documents", files=files, headers={"X-Upload-Token": "wrong"}
    ).status_code == 401
    assert client.post(
        "/v1/documents", files=files, headers={"X-Upload-Token": "secret123"}
    ).status_code == 200


def test_generation_unavailable_maps_to_503(eval_collection, fake_embedder):
    _ingest(eval_collection, fake_embedder, {"doc": "text"})
    service = RagService(Retriever(eval_collection, fake_embedder), llm=None, llm_error="no key")
    with TestClient(create_app(service)) as c:
        resp = c.post("/v1/query", json={"question": "anything here"})
        assert resp.status_code == 503
        # citation-only still works keyless
        ok = c.post("/v1/query", json={"question": "anything here", "generate_answer": False})
        assert ok.status_code == 200 and ok.json()["answer"] is None
