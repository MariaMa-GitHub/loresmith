from app.rag.relevance_gate import should_refuse_for_low_relevance


def test_threshold_none_always_returns_false():
    assert should_refuse_for_low_relevance([], None) is False
    assert should_refuse_for_low_relevance([0.1, 0.9], None) is False
    assert should_refuse_for_low_relevance([-100.0], None) is False


def test_empty_scores_with_threshold_returns_true():
    assert should_refuse_for_low_relevance([], 0.0) is True
    assert should_refuse_for_low_relevance([], 0.5) is True


def test_max_score_above_threshold_returns_false():
    assert should_refuse_for_low_relevance([0.3, 0.7, 0.5], 0.6) is False
    assert should_refuse_for_low_relevance([1.0], 1.0) is False


def test_max_score_below_threshold_returns_true():
    assert should_refuse_for_low_relevance([0.1, 0.2, 0.3], 0.5) is True
    assert should_refuse_for_low_relevance([-1.0, -2.0], 0.0) is True
