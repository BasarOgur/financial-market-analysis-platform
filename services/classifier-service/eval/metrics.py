"""Per-label precision/recall/F1 for single- and multi-label tasks.

Both tasks are scored the same way by treating every prediction as a label
set (sentiment = a one-element set). Hand-rolled (~25 lines) so the eval has
zero coupling to sklearn's report formatting; sklearn stays a model/-only
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabelScore:
    precision: float
    recall: float
    f1: float
    support: int  # gold occurrences of the label


def per_label_scores(
    y_true: list[set[str]], y_pred: list[set[str]], labels: tuple[str, ...]
) -> dict[str, LabelScore]:
    scores = {}
    for label in labels:
        tp = sum(label in t and label in p for t, p in zip(y_true, y_pred))
        fp = sum(label not in t and label in p for t, p in zip(y_true, y_pred))
        fn = sum(label in t and label not in p for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = LabelScore(precision, recall, f1, tp + fn)
    return scores


def macro_f1(scores: dict[str, LabelScore]) -> float:
    """Mean F1 over labels that occur in the gold data (support > 0)."""
    supported = [s.f1 for s in scores.values() if s.support]
    return sum(supported) / len(supported) if supported else 0.0
