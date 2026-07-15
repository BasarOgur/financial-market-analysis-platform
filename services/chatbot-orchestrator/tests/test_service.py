import pytest

from shared.contracts import OrchestratorQueryRequest, RAG_TOOL

from service import LLMUnavailable, OrchestratorService, RoutingError
from tests.conftest import ScriptedLLM
from tools import ToolUnavailable


def test_routes_to_tool_and_produces_final_answer(fake_tools):
    llm = ScriptedLLM(
        replies=[
            '{"tool": "query_financial_documents", "arguments": {"question": "how much did revenue grow"}}',
            "Meridian's data center segment grew 32% year over year [1].",
        ]
    )
    orch = OrchestratorService(fake_tools, llm)
    resp = orch.query(OrchestratorQueryRequest(message="How fast did Meridian's data center segment grow?"))

    assert resp.tool_used == "query_financial_documents"
    assert fake_tools.invoked == ("query_financial_documents", {"question": "how much did revenue grow"})
    assert resp.tool_result == {"ok": True}
    assert resp.answer == "Meridian's data center segment grew 32% year over year [1]."
    assert resp.model == "fake-llm"
    # router got the tool catalogue, not just the raw message
    assert RAG_TOOL.name in llm.calls[0]["prompt"]


def test_direct_answer_when_no_tool_applies(fake_tools):
    llm = ScriptedLLM(replies=['{"tool": null, "arguments": {}, "direct_answer": "Hi there!"}'])
    orch = OrchestratorService(fake_tools, llm)
    resp = orch.query(OrchestratorQueryRequest(message="hello"))

    assert resp.tool_used is None
    assert resp.tool_result is None
    assert resp.answer == "Hi there!"
    assert fake_tools.invoked is None


def test_declines_investment_advice_via_direct_answer(fake_tools):
    llm = ScriptedLLM(
        replies=['{"tool": null, "arguments": {}, "direct_answer": "I only provide analysis, not investment advice."}']
    )
    orch = OrchestratorService(fake_tools, llm)
    resp = orch.query(OrchestratorQueryRequest(message="Should I buy Meridian stock?"))

    assert "not investment advice" in resp.answer
    assert fake_tools.invoked is None


def test_query_without_llm_raises(fake_tools):
    orch = OrchestratorService(fake_tools, llm=None, llm_error="no key")
    with pytest.raises(LLMUnavailable):
        orch.query(OrchestratorQueryRequest(message="anything"))


def test_malformed_router_reply_raises_routing_error(fake_tools):
    llm = ScriptedLLM(replies=["not json at all"])
    orch = OrchestratorService(fake_tools, llm)
    with pytest.raises(RoutingError):
        orch.query(OrchestratorQueryRequest(message="anything"))


def test_unreachable_tool_degrades_gracefully_instead_of_crashing(fake_tools):
    fake_tools._error = ToolUnavailable("rag-service unreachable: connection refused")
    llm = ScriptedLLM(
        replies=['{"tool": "query_financial_documents", "arguments": {"question": "x"}}']
    )
    orch = OrchestratorService(fake_tools, llm)
    resp = orch.query(OrchestratorQueryRequest(message="How fast did Meridian grow?"))

    assert resp.tool_used == "query_financial_documents"
    assert "couldn't reach" in resp.answer
    # no second LLM call was made for a final answer -- only the routing call
    assert len(llm.calls) == 1


def test_route_returns_decision_dict_without_invoking_tools(fake_tools):
    llm = ScriptedLLM(replies=['{"tool": null, "arguments": {}, "direct_answer": "hi"}'])
    orch = OrchestratorService(fake_tools, llm)
    decision = orch.route("hello")
    assert decision == {"tool": None, "arguments": {}, "direct_answer": "hi"}
    assert fake_tools.invoked is None
