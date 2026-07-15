"""MarketDataService: fetch + map provider data. No LLM involved -- pure data
retrieval, so the only failure modes are "ticker not found" and "provider
unreachable" (see DECISIONS.md #2).

Client is injected so tests/eval use a fake, never hitting the network.
"""

from __future__ import annotations

from shared.contracts import MarketDataRequest, MarketDataResponse
from shared.logging import get_logger

from client import MarketDataClient

log = get_logger("marketdata.service")


class TickerNotFound(ValueError):
    """Ticker doesn't resolve to a known instrument (api.py maps this to 404)."""


class MarketDataService:
    def __init__(self, client: MarketDataClient) -> None:
        self._client = client

    def query(self, request: MarketDataRequest) -> MarketDataResponse:
        ticker = request.ticker.upper()
        info = self._client.get(ticker)

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            raise TickerNotFound(f"no market data found for ticker {ticker!r}")

        response = MarketDataResponse(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName") or ticker,
            price=price,
            currency=info.get("currency") or "USD",
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            volume=info.get("volume") or info.get("regularMarketVolume"),
        )
        log.info("fetched %s: price=%s", ticker, price)
        return response
