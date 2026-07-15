"""Provider-agnostic text generation over raw REST via httpx.

One small class per provider instead of three vendor SDKs: the three request
shapes are ~20 lines each, and services stay decoupled from SDK churn.
Select provider/model with FMA_LLM_PROVIDER / FMA_LLM_MODEL (see config.py).
"""

from __future__ import annotations

import time
from typing import Protocol

import httpx

from .config import Settings

_TIMEOUT = 60.0
_RETRY_STATUSES = {429, 503}
_BACKOFF_SECONDS = (15, 30, 60, 120, 240)  # rides out free-tier Gemini per-minute quotas


class LLMError(RuntimeError):
    """Raised when a provider returns an error or an unparseable response."""


class LLMClient(Protocol):
    model: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str: ...


def _post(url: str, headers: dict, payload: dict) -> dict:
    for attempt, backoff in enumerate((*_BACKOFF_SECONDS, None)):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        except httpx.TransportError as exc:
            # transient network failure (DNS blip, dropped connection, timeout):
            # retry on the same ladder instead of killing a long batch run
            if backoff is not None:
                time.sleep(backoff)
                continue
            raise LLMError(f"request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"request failed: {exc}") from exc
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in _RETRY_STATUSES and backoff is not None:
            time.sleep(backoff)  # rate limit / transient overload: wait and retry
            continue
        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    raise AssertionError("unreachable")


class GeminiLLM:
    """Gemini generateContent REST API (v1beta)."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self._api_key = api_key
        self.model = model

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=1024) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}
        data = _post(url, {"x-goog-api-key": self._api_key}, payload)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected Gemini response: {str(data)[:500]}") from exc


class OpenAILLM:
    """OpenAI chat completions REST API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._api_key = api_key
        self.model = model

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=1024) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = _post(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self._api_key}"},
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        )
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected OpenAI response: {str(data)[:500]}") from exc


class AnthropicLLM:
    """Anthropic Messages REST API."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._api_key = api_key
        self.model = model

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=1024) -> str:
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
            payload,
        )
        try:
            return data["content"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected Anthropic response: {str(data)[:500]}") from exc


def llm_from_settings(settings: Settings) -> LLMClient:
    provider = settings.llm_provider
    if provider == "gemini":
        return GeminiLLM(settings.gemini_api_key, settings.llm_model)
    if provider == "openai":
        return OpenAILLM(settings.openai_api_key, settings.llm_model)
    if provider == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.llm_model)
    raise LLMError(f"unknown FMA_LLM_PROVIDER: {provider!r} (gemini|openai|anthropic)")
