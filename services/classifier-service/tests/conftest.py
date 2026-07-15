"""Deterministic fakes so tests run offline with no API keys."""

from __future__ import annotations

import hashlib
import math

import pytest


class FakeEmbeddings:
    """Bag-of-words hashed into a fixed-dim vector; deterministic, offline.
    Similar texts share buckets, so a linear classifier can separate them.
    (Same fake as rag-service's tests.)"""

    model = "fake-bow-512"
    _DIM = 512

    def embed(self, texts, *, task="document"):
        out = []
        for text in texts:
            vec = [0.0] * self._DIM
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._DIM
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply='{"sentiment": "bullish", "topics": ["earnings"]}'):
        self.reply = reply
        self.calls: list[dict] = []

    def generate(self, prompt, *, system=None, temperature=0.2, max_tokens=1024):
        self.calls.append({"prompt": prompt, "system": system})
        return self.reply


@pytest.fixture
def fake_embedder():
    return FakeEmbeddings()


@pytest.fixture
def fake_llm():
    return FakeLLM()
