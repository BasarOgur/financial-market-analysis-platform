"""classifier-service standalone entry point.

Run the API server (trains the embedding classifier on fixtures at startup):
    python api.py [--port 8001]

One-shot demo without a server:
    python api.py --demo "Northwind beat estimates and raised guidance."

Batch mode (JSONL with a "text" field, or one plain snippet per line):
    python api.py --batch headlines.jsonl [--path llm]

Keyless mode (no API keys; embed path works, llm path returns 503):
    FMA_EMBEDDING_PROVIDER=local python api.py

Contract for the future orchestrator:
    GET  /v1/tool-schema  -> shared.contracts.ToolDefinition (register as LLM tool)
    POST /v1/query        -> ClassifyRequest -> ClassifyResponse
    GET  /healthz
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from shared.config import load_settings
from shared.contracts import CLASSIFIER_TOOL, ClassifyRequest, ClassifyResponse, ToolDefinition
from shared.embeddings import embeddings_from_settings
from shared.llm import LLMError, llm_from_settings
from shared.logging import get_logger, setup_logging

from model import load_news
from model.embed_path import EmbeddingClassifier
from model.llm_path import LLMClassifier
from service import ClassifierService, LLMUnavailable

log = get_logger("classifier.api")


def build_service() -> ClassifierService:
    """Wire real providers from env config; train the embed path on the fixture train split."""
    settings = load_settings()
    setup_logging(settings.log_level)
    embed_clf = EmbeddingClassifier(embeddings_from_settings(settings))
    embed_clf.fit(load_news("train"))
    llm_clf, llm_error = None, None
    try:
        llm_clf = LLMClassifier(llm_from_settings(settings))
    except LLMError as exc:
        llm_error = str(exc)
        log.warning("LLM unavailable (%s); embed path still works", exc)
    return ClassifierService(embed_clf, llm_clf, llm_error)


def create_app(service: ClassifierService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = service or build_service()
        yield

    app = FastAPI(
        title="classifier-service",
        description="Financial news classification: sentiment and topic tags. "
        "Analysis only — not investment advice.",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/tool-schema", response_model=ToolDefinition)
    async def tool_schema() -> ToolDefinition:
        return CLASSIFIER_TOOL

    @app.post("/v1/query", response_model=ClassifyResponse)
    async def query(request: ClassifyRequest) -> ClassifyResponse:
        try:
            return app.state.classifier.classify(request)
        except LLMUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


def read_batch(path: str) -> list[str]:
    """JSONL lines with a "text" field, or plain one-snippet-per-line."""
    texts = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            texts.append(json.loads(line)["text"])
        except (json.JSONDecodeError, TypeError, KeyError):
            texts.append(line)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--demo", metavar="TEXT", help="classify one snippet and exit (no server)")
    parser.add_argument("--batch", metavar="FILE", help="classify a JSONL/plain-text file and exit")
    parser.add_argument("--path", choices=["embed", "llm"], default="embed",
                        help="classifier path for --demo/--batch (default: embed)")
    args = parser.parse_args()

    if args.demo or args.batch:
        texts = [args.demo] if args.demo else read_batch(args.batch)
        classifier = build_service()
        results = []
        try:
            for i in range(0, len(texts), 50):  # request model caps texts at 50
                response = classifier.classify(ClassifyRequest(texts=texts[i : i + 50], path=args.path))
                results.extend(r.model_dump() for r in response.results)
        except LLMUnavailable as exc:
            raise SystemExit(f"llm path unavailable: {exc}") from exc
        print(json.dumps({"results": results, "path": response.path, "model": response.model,
                          "disclaimer": response.disclaimer}, indent=2))
        return

    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
