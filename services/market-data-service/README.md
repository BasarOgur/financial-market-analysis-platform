# market-data-service

Structured market/fundamentals lookups by ticker (price, market cap, P/E ratio,
volume) via [yfinance](https://github.com/ranaroussi/yfinance) — free, no API
key required.

> Raw market data only — **not investment advice**. Stamped on every response
> via the `disclaimer` field.

See [AGENTS.md](../../AGENTS.md) for the platform-wide rules this follows, and
[DECISIONS.md](DECISIONS.md) for why yfinance was chosen and how not-found vs.
provider-failure is detected.

## Run it

```bash
# repo root, once: python3 -m venv .venv && source .venv/bin/activate

cd services/market-data-service
pip install -r requirements.txt   # installs packages/shared editable too

# one-shot demo (no server, no key needed)
python api.py --demo AAPL

# serve
python api.py --port 8003
curl -X POST localhost:8003/v1/query -H 'Content-Type: application/json' -d '{"ticker": "AAPL"}'
```

No API key is needed at all — Yahoo Finance's quote endpoint is public.
**Unknown ticker?** `404`. **Provider unreachable/timed out?** `503`.

## API contract

Defined in `packages/shared/shared/contracts.py`; this service just serves it.

- `GET /healthz`
- `GET /v1/tool-schema` → `ToolDefinition` (`get_market_data`) — register as an LLM tool.
- `POST /v1/query` → `MarketDataRequest {ticker}` → `MarketDataResponse {ticker, name, price, currency, market_cap, pe_ratio, volume, disclaimer}`.
  404 = ticker not found; 503 = provider unreachable.

## Evaluation

```bash
python -m eval.run_eval
```

Hits the real provider (keyless, needs network) against 5 known-good tickers and
2 known-bad ones, checks found/not-found classification and price sanity. Full
report: `eval/results.md`.

## Tests

```bash
pytest
```

Offline, no network — the provider client is faked (`tests/conftest.py::FakeClient`).
One test in `test_client.py` verifies the real `YFinanceClient` maps a provider
exception to `ProviderUnavailable` (network call itself is mocked).
