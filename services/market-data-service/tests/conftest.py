import pytest

from client import MarketDataClient

AAPL_INFO = {
    "symbol": "AAPL",
    "longName": "Apple Inc.",
    "currentPrice": 314.86,
    "currency": "USD",
    "marketCap": 4_624_460_808_192,
    "trailingPE": 38.16,
    "volume": 36_328_962,
}

NOT_FOUND_INFO = {"trailingPegRatio": None}


class FakeClient:
    """Canned responses keyed by ticker; raises ProviderUnavailable for a
    sentinel ticker to simulate a network failure."""

    def __init__(self, responses: dict[str, dict]) -> None:
        self._responses = responses

    def get(self, ticker: str) -> dict:
        from client import ProviderUnavailable

        if ticker == "DOWN":
            raise ProviderUnavailable("simulated network failure")
        return self._responses.get(ticker, NOT_FOUND_INFO)


@pytest.fixture
def fake_client() -> MarketDataClient:
    return FakeClient({"AAPL": AAPL_INFO})
