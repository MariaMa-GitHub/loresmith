import json
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "app" / "eval" / "datasets" / "hades2.jsonl"
VALID_STRATA = {"factual", "multi_hop", "ambiguous", "adversarial"}
MIN_STRATUM_COUNTS = {
    "factual": 15,
    "multi_hop": 15,
    "ambiguous": 10,
    "adversarial": 10,
}
MIN_GOLD_SOURCE_ANNOTATED = 15
MIN_REFUSAL_ANNOTATED = 5


def load_questions():
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def test_dataset_exists():
    assert DATASET_PATH.exists()


def test_dataset_has_at_least_50_questions():
    assert len(load_questions()) >= 50


def test_all_strata_minimums_met():
    counts = Counter(q["stratum"] for q in load_questions())
    for stratum, minimum in MIN_STRATUM_COUNTS.items():
        assert counts[stratum] >= minimum, (
            f"Expected >= {minimum} for {stratum}, got {counts[stratum]}"
        )


def test_ids_are_unique_and_use_hades2_prefix():
    ids = [q["id"] for q in load_questions()]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("hades2-") for i in ids)


def test_all_questions_have_required_fields():
    for q in load_questions():
        for field in ("id", "question", "expected_answer", "stratum", "spoiler_tier"):
            assert field in q, f"Missing {field} in {q['id']}"
        assert q["stratum"] in VALID_STRATA
        assert 0 <= q["spoiler_tier"] <= 3


def test_coverage_thresholds():
    questions = load_questions()
    gold = [q for q in questions if q.get("gold_source_urls")]
    refusals = [q for q in questions if q.get("expects_refusal") is True]
    assert len(gold) >= MIN_GOLD_SOURCE_ANNOTATED
    assert len(refusals) >= MIN_REFUSAL_ANNOTATED
