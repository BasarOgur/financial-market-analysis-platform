import pytest

from shared.contracts import DISCLAIMER, RagQueryRequest

from service import LLMUnavailable, RagService
from tests.test_retriever import _ingest
from retrieval.retriever import Retriever


@pytest.fixture
def rag(eval_collection, fake_embedder, fake_llm):
    _ingest(eval_collection, fake_embedder, {"doc": "revenue grew 18% in fiscal 2025"})
    return RagService(Retriever(eval_collection, fake_embedder), fake_llm)


def test_generates_grounded_answer_with_citations(rag, fake_llm):
    resp = rag.query(RagQueryRequest(question="how much did revenue grow"))
    assert resp.answer == fake_llm.reply
    assert resp.model == "fake-llm"
    assert resp.disclaimer == DISCLAIMER
    assert resp.citations and resp.citations[0].chunk_id == "doc::000"
    # system prompt reaches the LLM and forbids investment advice
    assert "investment advice" in fake_llm.calls[0]["system"]
    assert "revenue grew 18%" in fake_llm.calls[0]["prompt"]


def test_citation_only_mode_skips_llm(rag, fake_llm):
    resp = rag.query(RagQueryRequest(question="revenue", generate_answer=False))
    assert resp.answer is None and resp.model is None
    assert resp.citations
    assert fake_llm.calls == []


def test_generation_without_llm_raises(eval_collection, fake_embedder):
    _ingest(eval_collection, fake_embedder, {"doc": "some text"})
    rag = RagService(Retriever(eval_collection, fake_embedder), llm=None)
    with pytest.raises(LLMUnavailable):
        rag.query(RagQueryRequest(question="anything at all"))
