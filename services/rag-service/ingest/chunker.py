"""Paragraph-aware chunking.

Packs whole paragraphs into chunks up to `max_words`; a paragraph that alone
exceeds the budget is split with a sliding word window with `overlap_words`
of overlap. Word counts approximate tokens (~0.75 tokens/word for English
financial text) without pulling in a tokenizer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


def _split_long_paragraph(words: list[str], max_words: int, overlap_words: int) -> list[str]:
    step = max_words - overlap_words
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step)]


def chunk_text(
    text: str,
    doc_id: str,
    meta: dict | None = None,
    *,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[Chunk]:
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    meta = meta or {}

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        words = para.split()
        if len(words) > max_words:
            if current:
                pieces.append("\n\n".join(current))
                current, current_words = [], 0
            pieces.extend(_split_long_paragraph(words, max_words, overlap_words))
            continue
        if current_words + len(words) > max_words and current:
            pieces.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(para)
        current_words += len(words)
    if current:
        pieces.append("\n\n".join(current))

    return [
        Chunk(chunk_id=f"{doc_id}::{i:03d}", doc_id=doc_id, text=piece, meta=dict(meta))
        for i, piece in enumerate(pieces)
    ]
