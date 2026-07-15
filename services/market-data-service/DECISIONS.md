# Architecture Decisions — market-data-service

Each entry: choice, rejected alternative, why. Rule 7 in [AGENTS.md](../../AGENTS.md).

## 1. yfinance as the market-data provider

**Chose:** `yfinance`, a thin wrapper over Yahoo Finance's public quote endpoint.

**Rejected:** Alpha Vantage, IEX Cloud, or other providers requiring a signup + API key.

**Why:** AGENTS.md rule 1 requires every service to run standalone, keyless. Every
key-based market-data API caps free tier at a handful of requests/day/minute, which
would make this service the one exception that can't demo or eval without a key. Yahoo's
quote endpoint needs none. Revisit if Yahoo's endpoint becomes unreliable or rate-limits
harder than observed here.

## 2. No LLM involved; only two failure modes

**Chose:** `MarketDataService.query()` calls the provider and maps its result straight
into `MarketDataResponse` — no generation step. Only `TickerNotFound` (404) and
`ProviderUnavailable` (503) exist as errors.

**Rejected:** routing "no ticker found" through an LLM to produce a friendlier message,
or wrapping every field in an LLM-written summary.

**Why:** the whole service is a structured data lookup; rag-service/classifier-service
need an LLM to *generate* something, this one doesn't. Adding an LLM step here would be
unrequested complexity for no product value (ponytail: YAGNI) — the orchestrator's own
answer-generation step already turns raw tool JSON into a natural-language reply.

## 3. `get_info()` over `fast_info` despite the extra request

**Chose:** `yf.Ticker(ticker).get_info()`, one HTTP round-trip, returns price, market
cap, volume, and `trailingPE` together.

**Rejected:** `fast_info` (faster, one call) plus a second call for P/E specifically.

**Why:** `fast_info` doesn't expose `trailingPE` at all. Fetching it separately would be
two network round-trips for one logical lookup; `get_info()` gets everything in one.

## 4. Not-found detection: missing price field, not an exception

**Chose:** yfinance's `get_info()` does not raise for an unknown ticker — it logs a 404
internally and returns a near-empty dict (`{"trailingPegRatio": None}`). `service.py`
treats "no `currentPrice`/`regularMarketPrice` field" as `TickerNotFound`.

**Rejected:** catching an exception for not-found (there isn't one to catch).

**Why:** verified empirically (`python -c "yf.Ticker('NOTAREALTICKERXYZ').get_info()"`)
before writing the check — the provider's actual behavior, not assumed behavior, decided
the code path. `ProviderUnavailable` stays reserved for real network/timeout failures
(wrapped in `client.py`'s `try/except Exception`).

## 5. Additions to `shared/contracts.py` are additive only

**Chose:** `MarketDataRequest`, `MarketDataResponse`, `MARKET_DATA_TOOL`, and a
dedicated `MARKET_DATA_DISCLAIMER` (not the shared `DISCLAIMER`).

**Rejected:** reusing `DISCLAIMER` ("Generated from source documents for analysis and
summarization only...").

**Why:** this service doesn't generate anything from source documents — it returns raw
provider data. Reusing the wrong disclaimer text would be a small but real accuracy bug
in every response. Same reasoning as rag-service/classifier-service: contract lives in
`shared/contracts.py`, nothing existing touched.

## 6. Client injected as a `Protocol`, same DI pattern as every other service

**Chose:** `MarketDataClient` protocol with one `get(ticker) -> dict` method;
`YFinanceClient` implements it for real, tests/eval use a `FakeClient` (unit tests) or
the real client (eval, deliberately live per DECISIONS #7 below).

**Rejected:** importing `yfinance` directly inside `service.py`.

**Why:** AGENTS.md rule 6 — keeps unit tests offline/fast; matches the injection pattern
used for `Retriever`/`LLMClient` in rag-service and `Classifier` in classifier-service.

## 7. Eval hits the real provider, not a fixture

**Chose:** `eval/run_eval.py` calls live yfinance against 5 known-good tickers (AAPL,
MSFT, GOOGL, AMZN, TSLA) and 2 known-bad ones, checking found/not-found classification
and that valid prices are positive.

**Rejected:** mocking provider responses in the eval, mirroring the unit tests.

**Why:** the one thing worth measuring here is "does the real keyless provider still
answer correctly" — mocking would just re-test the same mapping logic the unit tests
already cover. No LLM key needed to run it, unlike rag-service/classifier-service's
LLM-path evals.
