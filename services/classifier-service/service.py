"""ClassifierService: dispatches requests to one of the two classification paths.

Dependencies are injected (embed_clf, llm_clf) so api.py wires real providers
from config while tests and the eval harness inject fakes.
"""

from __future__ import annotations

from shared.contracts import ClassifyRequest, ClassifyResponse, NewsClassification
from shared.logging import get_logger

from model import Classifier

log = get_logger("classifier.service")


class LLMUnavailable(RuntimeError):
    """LLM path requested but no LLM is configured (api.py maps this to 503)."""


class ClassifierService:
    def __init__(self, embed_clf: Classifier, llm_clf: Classifier | None = None,
                 llm_error: str | None = None) -> None:
        self._paths: dict[str, Classifier | None] = {"embed": embed_clf, "llm": llm_clf}
        self._llm_error = llm_error or "no LLM configured"

    def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        clf = self._paths[request.path]
        if clf is None:
            raise LLMUnavailable(self._llm_error)
        predictions = clf.classify(request.texts)
        log.info("classified %d texts via %s (%s)", len(request.texts), request.path, clf.model)
        return ClassifyResponse(
            results=[
                NewsClassification(text=t, sentiment=p.sentiment, topics=p.topics)
                for t, p in zip(request.texts, predictions)
            ],
            path=request.path,
            model=clf.model,
        )
