"""Trained path: logistic regression on shared embeddings.

The "depth" seam of this service (analogous to rag-service's hybrid-search
seam): a classical classifier trained on the fixture train split, running on
any embedding provider from packages/shared — fully keyless with
FMA_EMBEDDING_PROVIDER=local. Training is cheap (~50 vectors), so the model
is fit at startup; no persistence layer (DECISIONS.md #4).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from shared.contracts import SENTIMENTS, TOPICS
from shared.embeddings import EmbeddingClient
from shared.logging import get_logger

from . import Prediction

log = get_logger("classifier.embed")

_TOPIC_THRESHOLD = 0.5


class EmbeddingClassifier:
    """Multinomial logistic regression for sentiment + one-vs-rest for topics."""

    def __init__(self, embedder: EmbeddingClient) -> None:
        self._embedder = embedder
        self._sentiment_clf: LogisticRegression | None = None
        self._topic_clfs: dict[str, LogisticRegression] = {}
        self.model = f"logreg@{embedder.model}"

    def fit(self, examples: list[dict]) -> None:
        """Train from dataset rows ({text, sentiment, topics})."""
        # Both fit and classify embed with task="document": classification is
        # symmetric, unlike retrieval's query/document asymmetry.
        X = np.array(self._embedder.embed([e["text"] for e in examples]))
        sentiments = [e["sentiment"] for e in examples]
        self._sentiment_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._sentiment_clf.fit(X, sentiments)
        for topic in TOPICS:
            y = np.array([topic in e["topics"] for e in examples])
            if y.all() or not y.any():
                continue  # degenerate label in this training set; predicted False below
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X, y)
            self._topic_clfs[topic] = clf
        log.info("trained on %d examples (%s)", len(examples), self._embedder.model)

    def classify(self, texts: list[str]) -> list[Prediction]:
        if self._sentiment_clf is None:
            raise RuntimeError("EmbeddingClassifier.classify() called before fit()")
        X = np.array(self._embedder.embed(texts))
        sentiments = self._sentiment_clf.predict(X)
        topic_probas = {t: clf.predict_proba(X)[:, 1] for t, clf in self._topic_clfs.items()}
        out = []
        for i, sentiment in enumerate(sentiments):
            probas = {t: p[i] for t, p in topic_probas.items()}
            topics = [t for t in TOPICS if probas.get(t, 0.0) >= _TOPIC_THRESHOLD]
            if not topics:  # contract requires >=1 topic: fall back to the likeliest
                topics = [max(probas, key=probas.get)] if probas else ["other"]
            assert sentiment in SENTIMENTS
            out.append(Prediction(sentiment=str(sentiment), topics=topics))
        return out
