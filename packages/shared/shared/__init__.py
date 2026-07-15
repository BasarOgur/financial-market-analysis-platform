"""Shared infrastructure for the financial market-analysis platform.

Modules:
    config     -- env-driven settings (provider selection, keys, models)
    logging    -- one-call logging setup
    llm        -- provider-agnostic text generation (gemini | openai | anthropic)
    embeddings -- provider-agnostic embeddings (gemini | openai | local)
    contracts  -- inter-module tool/API schemas (pydantic models)
"""
