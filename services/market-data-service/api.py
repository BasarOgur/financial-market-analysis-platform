"""market-data-service standalone entry point.

Run the API server:
    python api.py [--port 8003]

One-shot demo without a server:
    python api.py --demo AAPL

No API key needed -- yfinance is a free, keyless provider (DECISIONS.md #1).

Contract for the orchestrator:
    GET  /v1/tool-schema  -> shared.contracts.ToolDefinition (register as LLM tool)
    POST /v1/query        -> MarketDataRequest -> MarketDataResponse
    GET  /healthz
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from shared.contracts import MARKET_DATA_TOOL, MarketDataRequest, MarketDataResponse, ToolDefinition
from shared.logging import get_logger, setup_logging

from client import ProviderUnavailable, YFinanceClient
from service import MarketDataService, TickerNotFound

log = get_logger("marketdata.api")


def build_service() -> MarketDataService:
    setup_logging("INFO")
    return MarketDataService(YFinanceClient())


def create_app(service: MarketDataService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.marketdata = service or build_service()
        yield

    app = FastAPI(
        title="market-data-service",
        description="Structured market/fundamentals lookups by ticker. Raw data only -- not investment advice.",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/tool-schema", response_model=ToolDefinition)
    async def tool_schema() -> ToolDefinition:
        return MARKET_DATA_TOOL

    @app.post("/v1/query", response_model=MarketDataResponse)
    async def query(request: MarketDataRequest) -> MarketDataResponse:
        try:
            return app.state.marketdata.query(request)
        except TickerNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ProviderUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--demo", metavar="TICKER", help="fetch one ticker and exit (no server)")
    args = parser.parse_args()

    if args.demo:
        service = build_service()
        try:
            response = service.query(MarketDataRequest(ticker=args.demo))
        except TickerNotFound as exc:
            raise SystemExit(str(exc)) from exc
        except ProviderUnavailable as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(response.model_dump(), indent=2))
        return

    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
