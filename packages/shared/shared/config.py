"""Env-driven settings. Switch LLM/embedding providers via env vars, no code changes.

Reads an optional `.env` file from the current working directory (KEY=VALUE lines)
so each service can be configured independently when run standalone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str | Path = ".env") -> None:
    # ponytail: 10-line .env loader instead of a python-dotenv dependency.
    # Add python-dotenv if we ever need interpolation or multiline values.
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    """Provider selection and credentials, shared by every service."""

    llm_provider: str = "gemini"          # gemini | openai | anthropic
    llm_model: str = "gemini-2.5-flash"
    embedding_provider: str = "gemini"    # gemini | openai | local
    embedding_model: str = "gemini-embedding-001"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    log_level: str = "INFO"
    extra: dict = field(default_factory=dict)


def load_settings() -> Settings:
    """Build Settings from environment (after loading ./.env if present)."""
    _load_dotenv()
    env = os.environ
    return Settings(
        llm_provider=env.get("FMA_LLM_PROVIDER", "gemini").lower(),
        llm_model=env.get("FMA_LLM_MODEL", "gemini-2.5-flash"),
        embedding_provider=env.get("FMA_EMBEDDING_PROVIDER", "gemini").lower(),
        embedding_model=env.get("FMA_EMBEDDING_MODEL", "gemini-embedding-001"),
        gemini_api_key=env.get("GEMINI_API_KEY", ""),
        openai_api_key=env.get("OPENAI_API_KEY", ""),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        log_level=env.get("FMA_LOG_LEVEL", "INFO").upper(),
    )
