"""Eval harness for classifier-service. Run from the service root:

    python -m eval.run_eval          # embed path only (keyless with FMA_EMBEDDING_PROVIDER=local)
    python -m eval.run_eval --llm    # + the few-shot LLM path (needs an LLM key)
    python -m eval.run_eval --llm --limit 10   # quota-sized subset (free tiers)

Trains the embedding classifier on the train split, evaluates both paths on
the same held-out test split, and reports per-label precision/recall/F1 for
sentiment and topics. Writes eval/results.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.config import load_settings
from shared.contracts import SENTIMENTS, TOPICS
from shared.embeddings import embeddings_from_settings
from shared.llm import LLMError, llm_from_settings
from shared.logging import get_logger, setup_logging

from model import Classifier, load_news
from model.embed_path import EmbeddingClassifier
from model.llm_path import LLMClassifier

from .metrics import LabelScore, macro_f1, per_label_scores

RESULTS_PATH = Path(__file__).parent / "results.md"

log = get_logger("classifier.eval")


def evaluate(clf: Classifier, test: list[dict]) -> dict:
    """Both tasks' per-label scores + macro F1 + per-item predictions."""
    predictions = clf.classify([e["text"] for e in test])
    sent_scores = per_label_scores(
        [{e["sentiment"]} for e in test], [{p.sentiment} for p in predictions], SENTIMENTS
    )
    topic_scores = per_label_scores(
        [set(e["topics"]) for e in test], [set(p.topics) for p in predictions], TOPICS
    )
    return {
        "model": clf.model,
        "sentiment": sent_scores,
        "topics": topic_scores,
        "sentiment_macro_f1": macro_f1(sent_scores),
        "topics_macro_f1": macro_f1(topic_scores),
        "predictions": predictions,
    }


def _score_table(scores: dict[str, LabelScore]) -> str:
    rows = [
        f"| {label} | {s.precision:.2f} | {s.recall:.2f} | {s.f1:.2f} | {s.support} |"
        for label, s in scores.items()
    ]
    return "\n".join(["| label | precision | recall | F1 | support |", "|---|---|---|---|---|", *rows])


def _path_section(title: str, result: dict, test: list[dict]) -> str:
    misses = [
        f"| {e['id']} | {e['sentiment']}/{','.join(e['topics'])} | "
        f"{p.sentiment}/{','.join(p.topics)} |"
        for e, p in zip(test, result["predictions"])
        if p.sentiment != e["sentiment"] or set(p.topics) != set(e["topics"])
    ]
    miss_block = (
        "\n".join(["\n**Misses (gold vs predicted):**\n", "| id | gold | predicted |", "|---|---|---|", *misses])
        if misses
        else "\nNo misses."
    )
    return f"""
## {title} — `{result["model"]}`

### Sentiment (macro F1 {result["sentiment_macro_f1"]:.2f})

{_score_table(result["sentiment"])}

### Topics (macro F1 {result["topics_macro_f1"]:.2f})

{_score_table(result["topics"])}
{miss_block}
"""


def run(with_llm: bool, limit: int | None = None) -> str:
    settings = load_settings()
    setup_logging(settings.log_level)
    train, test = load_news("train"), load_news("test")
    if limit:
        test = test[:limit]

    embed_clf = EmbeddingClassifier(embeddings_from_settings(settings))
    embed_clf.fit(train)
    results = {"Trained path (logreg on embeddings)": evaluate(embed_clf, test)}

    if with_llm:
        llm_clf = LLMClassifier(llm_from_settings(settings))  # raises LLMError if unconfigured
        results["Few-shot LLM baseline"] = evaluate(llm_clf, test)

    summary_rows = [
        f"| {title} | `{r['model']}` | {r['sentiment_macro_f1']:.2f} | {r['topics_macro_f1']:.2f} |"
        for title, r in results.items()
    ]
    report = "\n".join(
        [
            "# classifier-service eval results",
            "",
            f"Test split: {len(test)} held-out snippets (train: {len(train)}). "
            "Same fixture set for both paths.",
            "",
            "| path | model | sentiment macro F1 | topics macro F1 |",
            "|---|---|---|---|",
            *summary_rows,
            *[_path_section(title, r, test) for title, r in results.items()],
        ]
    )
    RESULTS_PATH.write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="also eval the few-shot LLM path (needs an LLM key)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N test items (quota-limited runs)")
    args = parser.parse_args()
    try:
        print(run(with_llm=args.llm, limit=args.limit))
    except LLMError as exc:
        raise SystemExit(f"llm eval needs a configured LLM: {exc}") from exc
    print(f"written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
