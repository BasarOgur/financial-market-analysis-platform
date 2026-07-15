from unittest.mock import patch

import pytest

from client import ProviderUnavailable, YFinanceClient


def test_get_wraps_provider_exception_as_unavailable():
    client = YFinanceClient()
    with patch("yfinance.Ticker", side_effect=RuntimeError("boom")):
        with pytest.raises(ProviderUnavailable):
            client.get("AAPL")
