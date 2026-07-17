"""Ingestion pipeline: fixture docs -> chunks -> embeddings -> Chroma.

Documents are markdown files with a minimal frontmatter block:

    ---
    company: Meridian Semiconductors
    doc_type: 10-K
    period: FY2025
    section: MD&A
    ---
    <body>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.embeddings import EmbeddingClient
from shared.logging import get_logger

from .chunker import Chunk, chunk_text

log = get_logger("rag.ingest")


@dataclass
class Document:
    doc_id: str
    text: str
    meta: dict


@dataclass
class IngestStats:
    documents: int
    chunks: int


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    try:
        _, header, body = raw.split("---", 2)
    except ValueError:
        return {}, raw
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        if value:
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def load_documents(data_dir: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text())
        computed_source = " ".join(
            filter(None, (meta.get("company"), meta.get("doc_type"), meta.get("period"), meta.get("section")))
        )
        meta["source"] = meta.get("source") or computed_source or path.stem
        docs.append(Document(doc_id=path.stem, text=body, meta=meta))
    if not docs:
        raise FileNotFoundError(f"no .md documents found in {data_dir}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_text(doc.text, doc.doc_id, doc.meta))
    return chunks


def _embed_and_upsert(chunks: list[Chunk], collection, embedder: EmbeddingClient) -> None:
    log.info("embedding %d chunks", len(chunks))
    vectors = embedder.embed([c.text for c in chunks], task="document")
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[c.meta for c in chunks],
    )
    log.info("ingested %d chunks into collection %r", len(chunks), collection.name)


def ingest(data_dir: str | Path, collection, embedder: EmbeddingClient) -> IngestStats:
    """Chunk every document in data_dir and upsert into the Chroma collection."""
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    _embed_and_upsert(chunks, collection, embedder)
    return IngestStats(documents=len(docs), chunks=len(chunks))


def ingest_document(doc: Document, collection, embedder: EmbeddingClient) -> IngestStats:
    """Chunk, embed, and upsert a single already-loaded document (e.g. a user upload)."""
    chunks = chunk_text(doc.text, doc.doc_id, doc.meta)
    _embed_and_upsert(chunks, collection, embedder)
    return IngestStats(documents=1, chunks=len(chunks))


def persist_upload(doc: Document, data_dir: str | Path) -> Path:
    """Write an uploaded document as frontmatter-tagged markdown into data_dir,
    so a later `--reingest` (which rebuilds only from disk) doesn't drop it."""
    path = Path(data_dir) / f"{doc.doc_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = doc.meta.get("source", doc.doc_id)
    path.write_text(f"---\nsource: {source}\n---\n{doc.text}")
    return path


def extract_text(filename: str, data: bytes) -> str:
    """Plain text for .md/.txt; extracted text for .pdf.

    Raises ValueError on an unsupported extension.
    """
    ext = Path(filename).suffix.lower()
    if ext in (".md", ".txt"):
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from io import BytesIO

        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(BytesIO(data))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except PdfReadError as exc:
            raise ValueError(f"could not read pdf: {exc}") from exc
    raise ValueError(f"unsupported file type {ext!r}; allowed: .pdf, .md, .txt")
