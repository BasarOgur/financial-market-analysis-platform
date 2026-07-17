"""chatbot-orchestrator standalone entry point.

Run the API server (discovers downstream tools at startup):
    python api.py [--port 8002]

One-shot demo without a server:
    python api.py --demo "How fast did Meridian's data center segment grow?"

Requires rag-service (FMA_RAG_URL, default :8000), classifier-service
(FMA_CLASSIFIER_URL, default :8001), and market-data-service (FMA_MARKET_DATA_URL,
default :8003) running to discover their tools. A service that isn't reachable
is skipped, not fatal (degraded mode).

Contract:
    GET  /v1/tool-schema  -> shared.contracts.ToolDefinition (ask_financial_assistant)
    POST /v1/query        -> OrchestratorQueryRequest -> OrchestratorQueryResponse
    GET  /healthz
    GET  /                -> simple chat web UI (static/index.html)
    POST /v1/documents    -> multipart file, proxied to rag-service (chat UI upload button)
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from shared.config import load_settings
from shared.contracts import ORCHESTRATOR_TOOL, OrchestratorQueryRequest, OrchestratorQueryResponse, ToolDefinition
from shared.llm import LLMError, llm_from_settings
from shared.logging import get_logger, setup_logging

from service import LLMUnavailable, OrchestratorService, RoutingError
from tools import discover_tools

log = get_logger("orchestrator.api")

RAG_URL = os.environ.get("FMA_RAG_URL", "http://localhost:8000")
CLASSIFIER_URL = os.environ.get("FMA_CLASSIFIER_URL", "http://localhost:8001")
MARKET_DATA_URL = os.environ.get("FMA_MARKET_DATA_URL", "http://localhost:8003")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # mirrors rag-service's own cap; reject early rather than buffer huge files


def build_service() -> OrchestratorService:
    """Wire real providers from env config. Skips any downstream service that
    isn't reachable rather than failing startup."""
    settings = load_settings()
    setup_logging(settings.log_level)
    tools = discover_tools({"rag": RAG_URL, "classifier": CLASSIFIER_URL, "market_data": MARKET_DATA_URL})
    if not tools:
        log.warning("no downstream tools discovered; orchestrator will only answer directly")
    llm, llm_error = None, None
    try:
        llm = llm_from_settings(settings)
    except LLMError as exc:
        llm_error = str(exc)
        log.warning("LLM unavailable (%s); orchestrator cannot route or answer", exc)
    return OrchestratorService(tools, llm, llm_error)


def create_app(service: OrchestratorService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.orchestrator = service or build_service()
        yield

    app = FastAPI(
        title="chatbot-orchestrator",
        description="Routes user questions to the platform's tool services "
        "(document Q&A, news classification) and returns a final answer. "
        "Analysis/summarization only — not investment advice.",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/")
    async def ui() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/v1/tool-schema", response_model=ToolDefinition)
    async def tool_schema() -> ToolDefinition:
        return ORCHESTRATOR_TOOL

    @app.post("/v1/query", response_model=OrchestratorQueryResponse)
    async def query(request: OrchestratorQueryRequest) -> OrchestratorQueryResponse:
        try:
            return app.state.orchestrator.query(request)
        except LLMUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (LLMError, RoutingError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/documents")
    async def upload_document(
        file: UploadFile = File(...),
        x_upload_token: str | None = Header(default=None),
    ) -> JSONResponse:
        # ponytail: dumb proxy to rag-service, same request/response shape it
        # returns. Keeps the browser same-origin (no CORS) and RAG_URL server-side.
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large")
        headers = {"X-Upload-Token": x_upload_token} if x_upload_token else {}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{RAG_URL}/v1/documents",
                    files={"file": (file.filename, data, file.content_type)},
                    headers=headers,
                )
        except httpx.TransportError as exc:
            raise HTTPException(status_code=502, detail=f"rag-service unreachable: {exc}") from exc
        try:
            content = resp.json()
        except ValueError:
            content = {"detail": resp.text or "rag-service returned a non-JSON response"}
        return JSONResponse(status_code=resp.status_code, content=content)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--demo", metavar="MESSAGE", help="route one message and exit (no server)")
    args = parser.parse_args()

    if args.demo:
        orch = build_service()
        try:
            response = orch.query(OrchestratorQueryRequest(message=args.demo))
        except LLMUnavailable as exc:
            raise SystemExit(f"orchestrator needs a configured LLM: {exc}") from exc
        print(json.dumps(response.model_dump(), indent=2))
        return

    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
