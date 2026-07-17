"""rag-service standalone entry point.

Run the API server (ingests fixture data on first start):
    python api.py [--port 8000] [--reingest]

One-shot demo without a server:
    python api.py --demo "How fast did Meridian's data center segment grow?"

Keyless mode (no API keys; retrieval works, generation returns 503):
    FMA_EMBEDDING_PROVIDER=local python api.py

Contract for the future orchestrator:
    GET  /v1/tool-schema  -> shared.contracts.ToolDefinition (register as LLM tool)
    POST /v1/query        -> RagQueryRequest -> RagQueryResponse
    POST /v1/documents    -> multipart file upload (.pdf/.md/.txt) -> IngestStats
    GET  /healthz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from shared.config import load_settings
from shared.contracts import RAG_TOOL, RagQueryRequest, RagQueryResponse, ToolDefinition
from shared.embeddings import embeddings_from_settings
from shared.llm import LLMError, llm_from_settings
from shared.logging import get_logger, setup_logging

from ingest.pipeline import Document, IngestStats, extract_text, ingest, ingest_document, persist_upload
from retrieval.retriever import Retriever
from retrieval.store import open_collection
from service import LLMUnavailable, RagService

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_SOURCE_NAME_LEN = 200

log = get_logger("rag.api")

DATA_DIR = os.environ.get("RAG_DATA_DIR", "data/fixtures")
CHROMA_DIR = os.environ.get("RAG_CHROMA_DIR", ".chroma")
# ponytail: shared-secret gate, not real auth. Unset -> upload stays open for
# local/demo use like every other endpoint here. Set it for anything network-
# reachable, since this endpoint writes into the shared knowledge base.
UPLOAD_TOKEN = os.environ.get("RAG_UPLOAD_TOKEN")


def build_service(reingest: bool = False) -> RagService:
    """Wire real providers from env config. Ingests fixtures if the store is empty."""
    settings = load_settings()
    setup_logging(settings.log_level)
    embedder = embeddings_from_settings(settings)
    collection = open_collection(CHROMA_DIR)
    if reingest or collection.count() == 0:
        stats = ingest(DATA_DIR, collection, embedder)
        log.info("ingest complete: %d docs, %d chunks", stats.documents, stats.chunks)
    llm, llm_error = None, None
    try:
        llm = llm_from_settings(settings)
    except LLMError as exc:
        llm_error = str(exc)
        log.warning("LLM unavailable (%s); citation-only queries still work", exc)
    return RagService(Retriever(collection, embedder), llm, llm_error)


def create_app(service: RagService | None = None, *, reingest: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = service or build_service(reingest=reingest)
        yield

    app = FastAPI(
        title="rag-service",
        description="Financial-document Q&A over SEC filings and earnings transcripts. "
        "Analysis/summarization only — not investment advice.",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/tool-schema", response_model=ToolDefinition)
    async def tool_schema() -> ToolDefinition:
        return RAG_TOOL

    @app.post("/v1/query", response_model=RagQueryResponse)
    async def query(request: RagQueryRequest) -> RagQueryResponse:
        try:
            return app.state.rag.query(request)
        except LLMUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/documents", response_model=IngestStats)
    async def upload_document(
        file: UploadFile = File(...),
        x_upload_token: str | None = Header(default=None),
    ) -> IngestStats:
        if UPLOAD_TOKEN and x_upload_token != UPLOAD_TOKEN:
            raise HTTPException(status_code=401, detail="missing or invalid X-Upload-Token")

        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file type {ext!r}; allowed: .pdf, .md, .txt",
            )
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"file exceeds {MAX_UPLOAD_BYTES} byte limit")
        try:
            text = extract_text(file.filename, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not text.strip():
            raise HTTPException(status_code=400, detail="no extractable text in file")

        # basename only (drop any path segments) + strip control chars + cap length,
        # since this is stored as metadata and echoed back verbatim in query citations
        raw_name = Path(file.filename.replace("\\", "/")).name
        clean_name = "".join(ch for ch in raw_name if ch.isprintable())[:MAX_SOURCE_NAME_LEN] or "upload"

        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(clean_name).stem).strip("-").lower() or "file"
        content_hash = hashlib.sha256(data).hexdigest()[:8]
        doc = Document(
            doc_id=f"upload-{slug}-{content_hash}",
            text=text,
            meta={"source": clean_name},
        )
        retriever = app.state.rag.retriever
        stats = ingest_document(doc, retriever.collection, retriever.embedder)
        persist_upload(doc, DATA_DIR)
        return stats

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reingest", action="store_true", help="re-chunk and re-embed fixtures")
    parser.add_argument("--demo", metavar="QUESTION", help="answer one question and exit (no server)")
    args = parser.parse_args()

    if args.demo:
        rag = build_service(reingest=args.reingest)
        try:
            response = rag.query(RagQueryRequest(question=args.demo))
        except LLMUnavailable as exc:
            log.warning("%s -- returning citations only", exc)
            response = rag.query(RagQueryRequest(question=args.demo, generate_answer=False))
        print(json.dumps(response.model_dump(), indent=2))
        return

    import uvicorn

    uvicorn.run(create_app(reingest=args.reingest), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
