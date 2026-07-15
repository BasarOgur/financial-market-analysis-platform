"""Inter-module contract: the tool/API schema every service exposes.

A future chatbot-orchestrator discovers a service by GET /v1/tool-schema
(returns a ToolDefinition it can register as an LLM tool) and calls it via
POST /v1/query with the request model below. New services implement the same
pair of endpoints with their own request/response models — no shared-code
changes required to plug one in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Generated from source documents for analysis and summarization only. "
    "Not investment advice."
)


class Citation(BaseModel):
    chunk_id: str
    source: str = Field(description="Human-readable document reference")
    snippet: str = Field(description="First ~200 chars of the cited chunk")
    score: float = Field(description="Retrieval similarity score, higher is better")


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=3, description="Natural-language question about the ingested filings/transcripts")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    generate_answer: bool = Field(default=True, description="False returns citations only (no LLM call)")


class RagQueryResponse(BaseModel):
    answer: str | None = Field(description="Grounded answer with [n] citation markers; null when generate_answer=false")
    citations: list[Citation]
    model: str | None = Field(description="Generation model used, null when no answer was generated")
    disclaimer: str = DISCLAIMER


class ToolDefinition(BaseModel):
    """LLM-tool-style definition an orchestrator can register directly."""

    name: str
    description: str
    input_schema: dict


# --- classifier-service ---

Sentiment = Literal["bullish", "bearish", "neutral"]
Topic = Literal["earnings", "m&a", "regulation", "macro", "product", "other"]
SENTIMENTS: tuple[str, ...] = ("bullish", "bearish", "neutral")
TOPICS: tuple[str, ...] = ("earnings", "m&a", "regulation", "macro", "product", "other")


class NewsClassification(BaseModel):
    text: str = Field(description="The classified news snippet (echoed back)")
    sentiment: Sentiment
    topics: list[Topic] = Field(min_length=1, description="Multi-label topic tags")


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=50, description="Financial news snippets to classify")
    path: Literal["embed", "llm"] = Field(
        default="embed",
        description="Classifier path: 'embed' (trained on embeddings, keyless) or 'llm' (few-shot, needs an LLM key)",
    )


class ClassifyResponse(BaseModel):
    results: list[NewsClassification]
    path: str = Field(description="Classifier path that produced the results")
    model: str = Field(description="Underlying model (embedding or LLM) used")
    disclaimer: str = DISCLAIMER


CLASSIFIER_TOOL = ToolDefinition(
    name="classify_financial_news",
    description=(
        "Classify financial news snippets: sentiment (bullish/bearish/neutral) "
        "and multi-label topics (earnings, m&a, regulation, macro, product, "
        "other). Analysis only — not investment advice."
    ),
    input_schema=ClassifyRequest.model_json_schema(),
)


RAG_TOOL = ToolDefinition(
    name="query_financial_documents",
    description=(
        "Answer questions about ingested SEC filings and earnings-call "
        "transcripts. Returns a grounded answer with citations to source "
        "chunks. Analysis/summarization only — not investment advice."
    ),
    input_schema=RagQueryRequest.model_json_schema(),
)


# --- chatbot-orchestrator ---


class OrchestratorQueryRequest(BaseModel):
    message: str = Field(min_length=1, description="Natural-language user message")


class OrchestratorQueryResponse(BaseModel):
    answer: str = Field(description="Final natural-language answer")
    tool_used: str | None = Field(default=None, description="Name of the tool invoked, if any")
    tool_result: dict | None = Field(default=None, description="Raw JSON response from the invoked tool's service")
    model: str
    disclaimer: str = DISCLAIMER


ORCHESTRATOR_TOOL = ToolDefinition(
    name="ask_financial_assistant",
    description=(
        "Ask a financial-analysis assistant a natural-language question; it "
        "routes to document Q&A or news classification as needed and returns "
        "a final answer. Analysis/summarization only — not investment advice."
    ),
    input_schema=OrchestratorQueryRequest.model_json_schema(),
)


# --- market-data-service ---

MARKET_DATA_DISCLAIMER = "Raw market data from a third-party quote provider. Not investment advice."


class MarketDataRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, description="Stock ticker symbol, e.g. AAPL")


class MarketDataResponse(BaseModel):
    ticker: str
    name: str = Field(description="Company name")
    price: float = Field(description="Latest/regular-market price")
    currency: str
    market_cap: float | None = Field(default=None, description="Market capitalization")
    pe_ratio: float | None = Field(default=None, description="Trailing P/E ratio; null if not applicable")
    volume: int | None = Field(default=None, description="Latest/regular-market trading volume")
    disclaimer: str = MARKET_DATA_DISCLAIMER


MARKET_DATA_TOOL = ToolDefinition(
    name="get_market_data",
    description=(
        "Look up current market data (price, market cap, P/E ratio, volume) "
        "for a stock ticker. Raw data only — not investment advice."
    ),
    input_schema=MarketDataRequest.model_json_schema(),
)
