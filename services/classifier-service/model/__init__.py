"""Classification paths for classifier-service.

Two implementations of one interface (see DECISIONS.md #2):
    llm_path.LLMClassifier      -- few-shot via the shared LLM client
    embed_path.EmbeddingClassifier -- logistic regression on shared embeddings

Both expose `classify(texts) -> list[Prediction]`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DATA_PATH = Path(__file__).parent.parent / "data" / "news.jsonl"


@dataclass
class Prediction:
    sentiment: str
    topics: list[str]


class Classifier(Protocol):
    model: str

    def classify(self, texts: list[str]) -> list[Prediction]: ...


def load_news(split: str | None = None, path: Path = DATA_PATH) -> list[dict]:
    """Load the labeled fixture dataset, optionally filtered to one split."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if split is None or r["split"] == split]
