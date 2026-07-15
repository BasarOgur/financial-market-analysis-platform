"""LLM-as-judge faithfulness scoring.

Faithful = every factual claim in the answer is supported by the retrieved
context. Judged by whichever LLM the shared config selects (ideally a
different/stronger model than the generator: set FMA_LLM_MODEL accordingly
when judging).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.llm import LLMClient

JUDGE_SYSTEM = """\
You are a strict evaluator of grounded question answering over financial \
documents. Judge ONLY whether the answer is supported by the context \
excerpts, not whether it is well written. An answer is unfaithful if any \
factual claim (figure, direction of change, attribution) is absent from or \
contradicted by the context. Respond with raw JSON only, no code fences:
{"faithful": true|false, "reason": "<one short sentence, max 20 words>"}"""

JUDGE_TEMPLATE = """\
Context excerpts:
{context}

Question: {question}

Answer to evaluate:
{answer}

JSON verdict:"""


@dataclass
class Verdict:
    faithful: bool
    reason: str


def judge_faithfulness(llm: LLMClient, question: str, context: str, answer: str) -> Verdict:
    raw = llm.generate(
        JUDGE_TEMPLATE.format(context=context, question=question, answer=answer),
        system=JUDGE_SYSTEM,
        temperature=0.0,
        max_tokens=500,
    )
    match = re.search(r"\{.*\}", raw.replace("```json", "").replace("```", ""), re.DOTALL)
    if not match:
        return Verdict(faithful=False, reason=f"unparseable judge output: {raw[:120]}")
    try:
        data = json.loads(match.group(0))
        return Verdict(faithful=bool(data["faithful"]), reason=str(data.get("reason", "")))
    except (json.JSONDecodeError, KeyError):
        return Verdict(faithful=False, reason=f"unparseable judge output: {raw[:120]}")
