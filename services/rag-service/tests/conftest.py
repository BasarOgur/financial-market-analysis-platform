"""Deterministic fakes so tests run offline with no API keys."""

from __future__ import annotations

import hashlib
import math

import pytest


class FakeEmbeddings:
    """Bag-of-words hashed into a fixed-dim vector; deterministic, offline.
    Similar texts share buckets, so cosine ranking behaves sensibly."""

    model = "fake-bow-64"
    _DIM = 512  # low bucket-collision odds keep rankings intuitive

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

    def __init__(self, reply="Revenue grew 18% [1]."):
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


@pytest.fixture
def eval_collection():
    import uuid

    import chromadb

    # EphemeralClient shares one in-process instance; unique name isolates tests.
    client = chromadb.EphemeralClient()
    name = f"test-{uuid.uuid4().hex[:8]}"
    yield client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    client.delete_collection(name)
