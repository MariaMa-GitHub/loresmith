from __future__ import annotations

import re
from collections import Counter

from app.rag.citations import parse_inline_citation_indices, strip_inline_citations

_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokens(text: str) -> list[str]:
    stripped = strip_inline_citations(text.lower())
    return _TOKEN_RE.findall(stripped)


def exact_match(expected: str, actual: str) -> bool:
    return " ".join(_tokens(expected)) == " ".join(_tokens(actual))


def token_f1(expected: str, actual: str) -> float:
    expected_counts = Counter(_tokens(expected))
    actual_counts = Counter(_tokens(actual))

    if not expected_counts and not actual_counts:
        return 1.0
    if not expected_counts or not actual_counts:
        return 0.0

    overlap = sum((expected_counts & actual_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return (2 * precision * recall) / (precision + recall)


def token_recall(expected: str, actual: str) -> float:
    expected_counts = Counter(_tokens(expected))
    actual_counts = Counter(_tokens(actual))
    if not expected_counts:
        return 1.0
    overlap = sum((expected_counts & actual_counts).values())
    return overlap / sum(expected_counts.values())


def has_inline_citation(answer: str) -> bool:
    return bool(parse_inline_citation_indices(answer))
