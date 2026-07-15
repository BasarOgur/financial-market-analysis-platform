"""Retrieval metrics.

A retrieved chunk is relevant if it contains any gold span (whitespace- and
case-normalized substring match). Gold spans instead of gold chunk ids keeps
the dataset stable when chunking parameters change — see DECISIONS.md #6.
"""

from __future__ import annotations


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def relevance_at_ranks(chunk_texts: list[str], gold_spans: list[str]) -> list[bool]:
    """Per-rank relevance flags for one question's retrieved chunks."""
    spans = [normalize(s) for s in gold_spans]
    return [any(s in normalize(t) for s in spans) for t in chunk_texts]


def hit_at_k(relevance: list[list[bool]], k: int) -> float:
    return sum(any(flags[:k]) for flags in relevance) / len(relevance)


def mrr(relevance: list[list[bool]]) -> float:
    total = 0.0
    for flags in relevance:
        for rank, rel in enumerate(flags, 1):
            if rel:
                total += 1.0 / rank
                break
    return total / len(relevance)
