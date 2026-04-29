from __future__ import annotations


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _bool_rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def aggregate_by_stratum(results: list[dict]) -> dict[str, dict]:
    """Return per-stratum aggregate metrics, keyed by stratum name."""
    strata: dict[str, list[dict]] = {}
    for r in results:
        strata.setdefault(r["stratum"], []).append(r)
    out = {}
    for stratum, items in strata.items():
        out[stratum] = {
            "dataset_size": len(items),
            "faithfulness_rate": _bool_rate(
                [r["faithful"] for r in items if r["faithful"] is not None]
            ),
            "retrieval_recall_at_5_mean": _mean(
                [r["retrieval_recall_at_5"] for r in items
                 if r["retrieval_recall_at_5"] is not None]
            ),
            "citation_f1_mean": _mean(
                [r["citation_f1"] for r in items if r["citation_f1"] is not None]
            ),
            "answer_correctness_rate": _bool_rate(
                [r["answer_correct"] for r in items if r["answer_correct"] is not None]
            ),
            "refusal_appropriateness_rate": _bool_rate(
                [r["refusal_appropriate"] for r in items if r["refusal_appropriate"] is not None]
            ),
        }
    return out
