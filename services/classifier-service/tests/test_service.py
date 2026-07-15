import pytest

from shared.contracts import ClassifyRequest

from model import Prediction
from service import ClassifierService, LLMUnavailable


class StubClassifier:
    model = "stub"

    def classify(self, texts):
        return [Prediction(sentiment="neutral", topics=["other"]) for _ in texts]


def test_dispatches_to_embed_path():
    svc = ClassifierService(embed_clf=StubClassifier())
    resp = svc.classify(ClassifyRequest(texts=["a", "b"]))
    assert resp.path == "embed"
    assert resp.model == "stub"
    assert [r.text for r in resp.results] == ["a", "b"]
    assert "not investment advice" in resp.disclaimer.lower()


def test_llm_path_without_llm_raises():
    svc = ClassifierService(embed_clf=StubClassifier(), llm_clf=None, llm_error="no key")
    with pytest.raises(LLMUnavailable, match="no key"):
        svc.classify(ClassifyRequest(texts=["a"], path="llm"))


def test_llm_path_dispatches_when_configured():
    svc = ClassifierService(embed_clf=StubClassifier(), llm_clf=StubClassifier())
    resp = svc.classify(ClassifyRequest(texts=["a"], path="llm"))
    assert resp.path == "llm"
