def should_refuse_for_low_relevance(scores: list[float], threshold: float | None) -> bool:
    """Return True iff the gate is armed and no score meets the threshold.

    The gate is off (returns False) when threshold is None.
    When armed, returns True if scores is empty or max(scores) < threshold.
    """
    if threshold is None:
        return False
    return not scores or max(scores) < threshold
