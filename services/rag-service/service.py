"""RagService: retrieval + grounded answer generation.

Dependencies are injected (retriever, llm) so api.py wires real providers
from config while tests and the eval harness inject fakes/alternates.
"""

from __future__ import annotations

from shared.contracts import Citation, RagQueryRequest, RagQueryResponse
from shared.llm import LLMClient
from shared.logging import get_logger

from prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT, format_context
from retrieval.retriever import Retriever

log = get_logger("rag.service")


class LLMUnavailable(RuntimeError):
    """Generation requested but no LLM is configured (api.py maps this to 503)."""


class RagService:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None,
                 llm_error: str | None = None) -> None:
        self._retriever = retriever
        self._llm = llm
        self._llm_error = llm_error or "no LLM configured"

    def query(self, request: RagQueryRequest) -> RagQueryResponse:
        chunks = self._retriever.retrieve(request.question, top_k=request.top_k)
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                source=c.meta.get("source", c.chunk_id),
                snippet=c.text[:200],
                score=round(c.score, 4),
            )
            for c in chunks
        ]
        if not request.generate_answer:
            return RagQueryResponse(answer=None, citations=citations, model=None)

        if self._llm is None:
            raise LLMUnavailable(self._llm_error)
        prompt = ANSWER_TEMPLATE.format(context=format_context(chunks), question=request.question)
        answer = self._llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.1)
        log.info("answered %r with %d citations", request.question[:60], len(citations))
        return RagQueryResponse(answer=answer, citations=citations, model=self._llm.model)
