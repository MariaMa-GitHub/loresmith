from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CitationScore:
    precision: float
    recall: float
    f1: float
    extraneous_count: int


def per_item_citation_scores(cited: set[str], gold: set[str]) -> CitationScore | None:
    """Return citation precision/recall/F1 for one item, or None when gold is empty."""
    if not gold:
        return None
    intersection = cited & gold
    precision = len(intersection) / len(cited) if cited else 0.0
    recall = len(intersection) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return CitationScore(
        precision=precision,
        recall=recall,
        f1=f1,
        extraneous_count=len(cited - gold),
    )


def aggregate(scores: list[CitationScore]) -> dict:
    """Aggregate a list of CitationScores. Returns None values when scores is empty."""
    if not scores:
        return {
            "f1_mean": None,
            "precision_mean": None,
            "recall_mean": None,
            "extraneous_rate": None,
        }
    n = len(scores)
    return {
        "f1_mean": round(sum(s.f1 for s in scores) / n, 4),
        "precision_mean": round(sum(s.precision for s in scores) / n, 4),
        "recall_mean": round(sum(s.recall for s in scores) / n, 4),
        "extraneous_rate": round(sum(1 for s in scores if s.extraneous_count > 0) / n, 4),
    }
