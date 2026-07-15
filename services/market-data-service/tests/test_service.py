import pytest

from shared.contracts import MarketDataRequest

from service import MarketDataService, TickerNotFound
from client import ProviderUnavailable


def test_query_maps_fields(fake_client):
    service = MarketDataService(fake_client)
    resp = service.query(MarketDataRequest(ticker="AAPL"))
    assert resp.ticker == "AAPL"
    assert resp.name == "Apple Inc."
    assert resp.price == 314.86
    assert resp.market_cap == 4_624_460_808_192
    assert resp.pe_ratio == 38.16
    assert resp.volume == 36_328_962
    assert resp.disclaimer


def test_lowercase_ticker_normalized(fake_client):
    service = MarketDataService(fake_client)
    resp = service.query(MarketDataRequest(ticker="aapl"))
    assert resp.ticker == "AAPL"


def test_unknown_ticker_raises_not_found(fake_client):
    service = MarketDataService(fake_client)
    with pytest.raises(TickerNotFound):
        service.query(MarketDataRequest(ticker="NOTREAL"))


def test_provider_failure_propagates(fake_client):
    service = MarketDataService(fake_client)
    with pytest.raises(ProviderUnavailable):
        service.query(MarketDataRequest(ticker="DOWN"))
