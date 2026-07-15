"""All prompt text for classifier-service in one place (mirrors rag-service)."""

from shared.contracts import SENTIMENTS, TOPICS

SYSTEM_PROMPT = f"""\
You classify financial news snippets. For each snippet you output exactly one \
JSON object, no prose, no code fences:
{{"sentiment": "<label>", "topics": ["<label>", ...]}}

sentiment — one of {list(SENTIMENTS)}:
- bullish: positive for the company/asset mentioned (beats, upgrades, approvals, accretive deals)
- bearish: negative (misses, fines, recalls, blocked deals, weak demand)
- neutral: factual with no clear direction (scheduled events, unchanged dividends, routine filings)

topics — one or more of {list(TOPICS)}:
- earnings: results, guidance, forecasts, financial reports
- m&a: mergers, acquisitions, divestitures, takeover bids, deal talks
- regulation: regulators, fines, approvals, lawsuits, tariffs, rule changes
- macro: rates, inflation, jobs, GDP, sector-wide economic conditions
- product: launches, recalls, design wins, product shutdowns
- other: personnel, buybacks, dividends, index changes, corporate housekeeping

You analyze news only. You never give investment advice.\
"""

# ponytail: few-shots fixed here, not sampled from data/news.jsonl — deterministic
# prompt, and eval on the test split stays honest (train split leakage is fine,
# these ARE train-style examples).
FEW_SHOTS = """\
Snippet: Orion Freight beat quarterly profit estimates and raised its dividend by 8%.
{"sentiment": "bullish", "topics": ["earnings", "other"]}

Snippet: Regulators opened an investigation into Lakeside Insurance's claims practices.
{"sentiment": "bearish", "topics": ["regulation"]}

Snippet: Quartz Software will release its annual results on February 12.
{"sentiment": "neutral", "topics": ["earnings"]}

Snippet: Rising unemployment claims added to concerns that the economy is losing momentum.
{"sentiment": "bearish", "topics": ["macro"]}

Snippet: Aster Pharma agreed to buy a gene-therapy startup, gaining a late-stage candidate.
{"sentiment": "bullish", "topics": ["m&a", "product"]}\
"""

def classify_prompt(text: str) -> str:
    # not str.format: the few-shot JSON braces would be parsed as fields
    return f"{FEW_SHOTS}\n\nSnippet: {text}\n"
