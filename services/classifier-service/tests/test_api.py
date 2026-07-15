import pytest
from fastapi.testclient import TestClient

from api import create_app, read_batch
from model import Prediction
from service import ClassifierService


class StubClassifier:
    model = "stub"

    def classify(self, texts):
        return [Prediction(sentiment="bullish", topics=["earnings"]) for _ in texts]


@pytest.fixture
def client():
    service = ClassifierService(StubClassifier(), llm_clf=None, llm_error="no key")
    with TestClient(create_app(service)) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_tool_schema_matches_contract(client):
    data = client.get("/v1/tool-schema").json()
    assert data["name"] == "classify_financial_news"
    props = data["input_schema"]["properties"]
    assert "texts" in props and "path" in props


def test_query_roundtrip(client):
    resp = client.post("/v1/query", json={"texts": ["Acme beat estimates."]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["sentiment"] == "bullish"
    assert body["results"][0]["topics"] == ["earnings"]
    assert body["path"] == "embed"
    assert "not investment advice" in body["disclaimer"].lower()


def test_llm_path_returns_503_when_unconfigured(client):
    resp = client.post("/v1/query", json={"texts": ["x"], "path": "llm"})
    assert resp.status_code == 503


def test_validation_rejects_empty_texts(client):
    assert client.post("/v1/query", json={"texts": []}).status_code == 422


def test_read_batch_accepts_jsonl_and_plain_lines(tmp_path):
    f = tmp_path / "batch.jsonl"
    f.write_text('{"text": "from jsonl"}\nplain snippet line\n\n')
    assert read_batch(str(f)) == ["from jsonl", "plain snippet line"]
