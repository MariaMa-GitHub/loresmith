from __future__ import annotations

from app.eval.citation_metrics import per_item_citation_scores


def test_subset_match_is_perfect_f1():
    result = per_item_citation_scores(cited={"a"}, gold={"a"})
    assert result.precision == 1.0 and result.recall == 1.0 and result.f1 == 1.0
    assert result.extraneous_count == 0


def test_extra_supporting_citation_is_penalized_proportionally():
    # Three cited, one is gold → precision=1/3, recall=1, F1=0.5
    result = per_item_citation_scores(cited={"a", "b", "c"}, gold={"a"})
    assert round(result.precision, 4) == round(1 / 3, 4)
    assert result.recall == 1.0
    assert round(result.f1, 4) == 0.5
    assert result.extraneous_count == 2


def test_missing_gold_zeroes_recall():
    result = per_item_citation_scores(cited={"b"}, gold={"a"})
    assert result.recall == 0.0 and result.f1 == 0.0


def test_no_gold_returns_none():
    assert per_item_citation_scores(cited={"a"}, gold=set()) is None
