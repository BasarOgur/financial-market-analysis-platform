"""Market-data provider client. yfinance wraps Yahoo Finance's public quote
endpoint -- no API key, free, sufficient for read-only quote lookups.
See DECISIONS.md #1.
"""

from __future__ import annotations

from typing import Protocol

import yfinance as yf


class ProviderUnavailable(RuntimeError):
    """Provider network/timeout/rate-limit failure (api.py maps this to 503)."""


class MarketDataClient(Protocol):
    def get(self, ticker: str) -> dict:
        """Raw provider info dict for a ticker. Callers decide if it's valid/found."""
        ...


class YFinanceClient:
    def get(self, ticker: str) -> dict:
        try:
            return yf.Ticker(ticker).get_info()
        except Exception as exc:  # yfinance raises assorted urllib/requests errors
            raise ProviderUnavailable(f"market-data provider unreachable: {exc}") from exc
