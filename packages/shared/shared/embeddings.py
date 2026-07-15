"""Provider-agnostic embeddings.

Providers:
    gemini -- text-embedding-004 via batchEmbedContents REST (default; supports
              asymmetric task types RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)
    openai -- /v1/embeddings
    local  -- Chroma's bundled ONNX all-MiniLM-L6-v2; no API key, runs offline.
              Used for keyless dev and the offline eval baseline. Requires
              chromadb to be installed (it is, in any service using the store).

Select with FMA_EMBEDDING_PROVIDER / FMA_EMBEDDING_MODEL.
"""

from __future__ import annotations

import time
from typing import Literal, Protocol

import httpx

from .config import Settings

_TIMEOUT = 60.0
_RETRY_STATUSES = {429, 503}
_BACKOFF_SECONDS = (5, 15, 30)  # transient overload/rate-limit, same pattern as shared/llm.py
Task = Literal["document", "query"]


class EmbeddingError(RuntimeError):
    pass


def _post(url: str, headers: dict, payload: dict) -> dict:
    for backoff in (*_BACKOFF_SECONDS, None):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"request failed: {exc}") from exc
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in _RETRY_STATUSES and backoff is not None:
            time.sleep(backoff)
            continue
        raise EmbeddingError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    raise AssertionError("unreachable")


class EmbeddingClient(Protocol):
    model: str

    def embed(self, texts: list[str], *, task: Task = "document") -> list[list[float]]: ...


class GeminiEmbeddings:
    _BATCH = 100  # API limit per batchEmbedContents call
    _TASK_TYPE = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is not set")
        self._api_key = api_key
        self.model = model

    def embed(self, texts, *, task="document") -> list[list[float]]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents"
        )
        out: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH):
            batch = texts[i : i + self._BATCH]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": self._TASK_TYPE[task],
                    }
                    for t in batch
                ]
            }
            data = _post(url, {"x-goog-api-key": self._api_key}, payload)
            try:
                out.extend(e["values"] for e in data["embeddings"])
            except KeyError as exc:
                raise EmbeddingError(f"unexpected response: {str(data)[:500]}") from exc
        return out


class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is not set")
        self._api_key = api_key
        self.model = model

    def embed(self, texts, *, task="document") -> list[list[float]]:
        # OpenAI embeddings are symmetric; `task` is accepted for interface parity.
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self.model, "input": texts},
                timeout=_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"request failed: {exc}") from exc
        if resp.status_code != 200:
            raise EmbeddingError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


class LocalEmbeddings:
    """Chroma's bundled ONNX all-MiniLM-L6-v2. Keyless, offline, ~80MB one-time download."""

    model = "all-MiniLM-L6-v2 (onnx, local)"

    def __init__(self) -> None:
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        except ImportError as exc:
            raise EmbeddingError(
                "local embeddings require chromadb: pip install chromadb"
            ) from exc
        self._ef = ONNXMiniLM_L6_V2()

    def embed(self, texts, *, task="document") -> list[list[float]]:
        return [list(map(float, v)) for v in self._ef(texts)]


def embeddings_from_settings(settings: Settings) -> EmbeddingClient:
    provider = settings.embedding_provider
    if provider == "gemini":
        return GeminiEmbeddings(settings.gemini_api_key, settings.embedding_model)
    if provider == "openai":
        return OpenAIEmbeddings(settings.openai_api_key, settings.embedding_model)
    if provider == "local":
        return LocalEmbeddings()
    raise EmbeddingError(
        f"unknown FMA_EMBEDDING_PROVIDER: {provider!r} (gemini|openai|local)"
    )
