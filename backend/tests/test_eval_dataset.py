import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "app" / "eval" / "datasets" / "hades.jsonl"
VALID_STRATA = {"factual", "multi_hop", "ambiguous", "adversarial"}


def load_questions():
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def test_dataset_exists():
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"


def test_dataset_has_at_least_100_questions():
    questions = load_questions()
    assert len(questions) >= 100, f"Expected >= 100 questions, got {len(questions)}"


def test_all_questions_have_required_fields():
    for q in load_questions():
        assert "id" in q, f"Missing 'id' in {q}"
        assert "question" in q, f"Missing 'question' in {q}"
        assert "expected_answer" in q, f"Missing 'expected_answer' in {q}"
        assert "stratum" in q, f"Missing 'stratum' in {q}"
        assert "spoiler_tier" in q, f"Missing 'spoiler_tier' in {q}"


def test_all_strata_are_valid():
    for q in load_questions():
        assert q["stratum"] in VALID_STRATA, (
            f"Invalid stratum '{q['stratum']}' in question {q['id']}"
        )


def test_spoiler_tiers_are_integers_0_to_3():
    for q in load_questions():
        assert isinstance(q["spoiler_tier"], int), f"spoiler_tier must be int in {q['id']}"
        assert 0 <= q["spoiler_tier"] <= 3, (
            f"spoiler_tier out of range in {q['id']}: {q['spoiler_tier']}"
        )


def test_ids_are_unique():
    ids = [q["id"] for q in load_questions()]
    assert len(ids) == len(set(ids)), "Duplicate IDs found in dataset"


def test_all_four_strata_represented():
    strata = {q["stratum"] for q in load_questions()}
    assert strata == VALID_STRATA, f"Not all strata represented. Found: {strata}"
