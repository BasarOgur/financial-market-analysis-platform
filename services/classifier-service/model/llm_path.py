"""Zero/few-shot path: JSON classification via the shared LLM client.

One generate() call per snippet, temperature 0. Malformed or off-vocabulary
output is coerced (unknown labels dropped, defaults applied) rather than
raised: a batch of 50 shouldn't die on one bad generation. Coercions are
logged so the eval can be read honestly.
"""

from __future__ import annotations

import json
import re

from shared.contracts import SENTIMENTS, TOPICS
from shared.llm import LLMClient
from shared.logging import get_logger

# plain import: services run in place from their root (AGENTS.md rule 4)
from prompts import SYSTEM_PROMPT, classify_prompt

from . import Prediction

log = get_logger("classifier.llm")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_reply(reply: str) -> Prediction:
    """Extract and coerce one prediction from a model reply."""
    match = _JSON_RE.search(reply)
    if not match:
        log.warning("no JSON in reply %r; defaulting to neutral/other", reply[:80])
        return Prediction(sentiment="neutral", topics=["other"])
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        log.warning("unparseable JSON in reply %r; defaulting", reply[:80])
        return Prediction(sentiment="neutral", topics=["other"])
    sentiment = str(data.get("sentiment", "")).lower()
    if sentiment not in SENTIMENTS:
        log.warning("invalid sentiment %r coerced to neutral", sentiment)
        sentiment = "neutral"
    raw = data.get("topics", [])
    topics = [t for t in (str(t).lower() for t in raw) if t in TOPICS] or ["other"]
    return Prediction(sentiment=sentiment, topics=topics)


class LLMClassifier:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.model = llm.model

    def classify(self, texts: list[str]) -> list[Prediction]:
        out = []
        for text in texts:
            reply = self._llm.generate(
                classify_prompt(text),
                system=SYSTEM_PROMPT,
                temperature=0.0,
                # ponytail: 1024 not 128 — thinking models (gemini-2.5-flash) burn
                # budget on reasoning tokens and truncate the JSON otherwise
                max_tokens=1024,
            )
            out.append(parse_reply(reply))
        return out
