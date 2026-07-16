"""Dense retriever over the Chroma collection.

Extension seam (deliberate, not built yet — see DECISIONS.md #5): hybrid
search adds a lexical (BM25) candidate pass merged with the dense candidates,
and a cross-encoder rerank runs over the merged set before the top_k cut.
Both slot into `retrieve()` between candidate fetch and the return, without
touching callers: they only see `retrieve(query, top_k) -> list[RetrievedChunk]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.embeddings import EmbeddingClient


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float  # cosine similarity, higher is better
    meta: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, collection, embedder: EmbeddingClient) -> None:
        self._collection = collection
        self._embedder = embedder

    @property
    def collection(self):
        return self._collection

    @property
    def embedder(self) -> EmbeddingClient:
        return self._embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        [query_vec] = self._embedder.embed([query], task="query")
        res = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        # ponytail: dense-only. Hybrid/rerank seam is here (merge + rerank
        # candidates before this return) — add when eval shows dense misses.
        return [
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                score=1.0 - dist,  # cosine distance -> similarity
                meta=meta or {},
            )
            for cid, doc, dist, meta in zip(
                res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0]
            )
        ]
