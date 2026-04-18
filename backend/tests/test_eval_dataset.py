import json
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "app" / "eval" / "datasets" / "hades.jsonl"
VALID_STRATA = {"factual", "multi_hop", "ambiguous", "adversarial"}
MIN_STRATUM_COUNTS = {
    "factual": 50,
    "multi_hop": 35,
    "ambiguous": 20,
    "adversarial": 20,
}
MIN_GOLD_SOURCE_ANNOTATED = 30
MIN_MULTI_TURN_ANNOTATED = 5
MIN_REFUSAL_ANNOTATED = 8


def load_questions():
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def test_dataset_exists():
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"


def test_dataset_has_at_least_150_questions():
    questions = load_questions()
    assert len(questions) >= 150, f"Expected >= 150 questions, got {len(questions)}"


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


def test_dataset_is_reasonably_stratified_for_week_4_scope():
    counts = Counter(q["stratum"] for q in load_questions())
    for stratum, minimum in MIN_STRATUM_COUNTS.items():
        assert counts[stratum] >= minimum, (
            f"Expected at least {minimum} {stratum} questions by Week 4, "
            f"got {counts[stratum]}"
        )


def test_optional_eval_fields_are_well_typed_when_present():
    for q in load_questions():
        if "history" in q:
            assert isinstance(q["history"], list), f"history must be a list in {q['id']}"
            for message in q["history"]:
                assert set(message) >= {"role", "content"}, f"Invalid history item in {q['id']}"
        if "gold_source_urls" in q:
            assert isinstance(q["gold_source_urls"], list), (
                f"gold_source_urls must be a list in {q['id']}"
            )
            assert all(isinstance(url, str) for url in q["gold_source_urls"]), (
                f"gold_source_urls must contain strings in {q['id']}"
            )
        if "expects_refusal" in q:
            assert q["expects_refusal"] in {True, False, None}


def test_dataset_has_meaningful_annotation_coverage_for_week_4_eval():
    questions = load_questions()
    gold_annotated = [q for q in questions if q.get("gold_source_urls")]
    multi_turn = [q for q in questions if q.get("history")]
    refusals = [q for q in questions if q.get("expects_refusal") is True]

    assert len(gold_annotated) >= MIN_GOLD_SOURCE_ANNOTATED, (
        "Expected a meaningful number of retrieval-annotated examples for recall@5."
    )
    assert len(multi_turn) >= MIN_MULTI_TURN_ANNOTATED, (
        "Expected several built-in multi-turn eval examples for the query rewriter."
    )
    assert len(refusals) >= MIN_REFUSAL_ANNOTATED, (
        "Expected several explicit refusal examples for ambiguity/adversarial scoring."
    )


def test_gold_source_annotations_cover_all_spoiler_tiers():
    annotated_tiers = {
        q["spoiler_tier"]
        for q in load_questions()
        if q.get("gold_source_urls")
    }
    assert annotated_tiers == {0, 1, 2, 3}, (
        f"Expected gold-source annotations across spoiler tiers 0-3, got {annotated_tiers}"
    )


def test_refusal_examples_are_scoped_and_written_as_refusals():
    for q in load_questions():
        if q.get("expects_refusal") is not True:
            continue

        assert q["stratum"] in {"ambiguous", "adversarial"}, (
            f"Refusal examples should live in ambiguous/adversarial strata: {q['id']}"
        )
        lowered = q["expected_answer"].lower()
        assert "enough evidence" in lowered or "insufficient evidence" in lowered, (
            f"Refusal gold answer should explicitly model grounded refusal behavior: {q['id']}"
        )


def test_multi_turn_examples_are_true_follow_ups():
    for q in load_questions():
        history = q.get("history") or []
        if not history:
            continue

        latest_user_turn = next(
            (
                message["content"].lower()
                for message in reversed(history)
                if message["role"] == "user"
            ),
            "",
        )
        assert latest_user_turn, f"Missing prior user turn in history for {q['id']}"
        assert q["question"].lower() != latest_user_turn, (
            f"Multi-turn example should be a real follow-up, not a duplicate question: {q['id']}"
        )


def test_retrieval_annotations_span_factual_and_multi_hop_questions():
    annotated_strata = {
        q["stratum"]
        for q in load_questions()
        if q.get("gold_source_urls")
    }
    assert {"factual", "multi_hop"}.issubset(annotated_strata), (
        "Expected retrieval annotations to cover factual and multi-hop questions, "
        f"got {annotated_strata}"
    )


def test_reviewed_regression_examples_are_corrected():
    by_id = {q["id"]: q for q in load_questions()}
    assert "40% more damage" in by_id["hades-015"]["expected_answer"]
    assert (
        "There is no Aspect of Poseidon on the Eternal Spear"
        in by_id["hades-100"]["expected_answer"]
    )
    assert (
        "There is no Pact of Punishment condition called Olympian Bane"
        in by_id["hades-101"]["expected_answer"]
    )
    assert "Greater Reflex" in by_id["hades-102"]["expected_answer"]
    assert (
        "There is no Pact condition or boon called Chaos Bane"
        in by_id["hades-104"]["expected_answer"]
    )
    assert "Lucky Tooth can add one more revival" in by_id["hades-109"]["expected_answer"]
    assert "Hunting Blades" in by_id["hades-116"]["expected_answer"]
    assert "Cold Embrace" in by_id["hades-119"]["expected_answer"]
    assert "Blitz Disc" in by_id["hades-122"]["expected_answer"]
    assert "Zeus + Dionysus" in by_id["hades-123"]["expected_answer"]
    assert "Cold Fusion" in by_id["hades-129"]["expected_answer"]
