"""Chroma persistence glue."""

from __future__ import annotations

import chromadb

COLLECTION_NAME = "filings"


def open_collection(persist_dir: str, name: str = COLLECTION_NAME):
    """Persistent cosine-space collection. Embeddings are always supplied by us,
    so no Chroma embedding function is attached."""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
