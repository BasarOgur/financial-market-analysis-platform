import pytest
from fastapi.testclient import TestClient

from shared.contracts import MARKET_DATA_TOOL

from api import create_app
from service import MarketDataService


@pytest.fixture
def client(fake_client):
    service = MarketDataService(fake_client)
    with TestClient(create_app(service)) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_tool_schema_matches_contract(client):
    data = client.get("/v1/tool-schema").json()
    assert data["name"] == MARKET_DATA_TOOL.name
    assert set(data["input_schema"]["properties"]) == {"ticker"}
    assert data["input_schema"]["required"] == ["ticker"]


def test_query_endpoint_roundtrip(client):
    resp = client.post("/v1/query", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL" and body["price"] == 314.86 and body["disclaimer"]


def test_unknown_ticker_maps_to_404(client):
    resp = client.post("/v1/query", json={"ticker": "NOTREAL"})
    assert resp.status_code == 404


def test_provider_failure_maps_to_503(client):
    resp = client.post("/v1/query", json={"ticker": "DOWN"})
    assert resp.status_code == 503


def test_query_validation_rejects_bad_input(client):
    assert client.post("/v1/query", json={}).status_code == 422
