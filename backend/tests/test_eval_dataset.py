import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from app.eval.source_identity import resolve_source_identities

DATASET_PATH = Path(__file__).parent.parent / "app" / "eval" / "datasets" / "hades.jsonl"
VALID_STRATA = {"factual", "multi_hop", "ambiguous", "adversarial"}
MIN_STRATUM_COUNTS = {
    "factual": 60,
    "multi_hop": 60,
    "ambiguous": 40,
    "adversarial": 40,
}
MIN_GOLD_SOURCE_ANNOTATED = 45
MIN_MULTI_TURN_ANNOTATED = 8
MIN_REFUSAL_ANNOTATED = 15


def load_questions():
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def test_dataset_exists():
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"


def test_dataset_has_at_least_200_questions():
    questions = load_questions()
    assert len(questions) >= 200, f"Expected >= 200 questions, got {len(questions)}"


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


def test_dataset_is_reasonably_stratified_for_week_5_scope():
    counts = Counter(q["stratum"] for q in load_questions())
    for stratum, minimum in MIN_STRATUM_COUNTS.items():
        assert counts[stratum] >= minimum, (
            f"Expected at least {minimum} {stratum} questions by Week 5, "
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


def test_gold_source_annotations_resolve_to_ingested_source_identities():
    unresolved: dict[str, list[str]] = {}
    for q in load_questions():
        gold_source_urls = q.get("gold_source_urls") or []
        resolved = resolve_source_identities("hades", gold_source_urls)
        if resolved.unresolved_urls:
            unresolved[q["id"]] = resolved.unresolved_urls

    assert unresolved == {}, (
        "Each gold source URL should map to an ingested canonical source identity. "
        f"Unresolved entries: {unresolved}"
    )


def test_dataset_has_meaningful_annotation_coverage_for_week_5_eval():
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


def test_gold_annotation_rate_factual_and_multi_hop():
    """At least 90% of factual + multi_hop items must have a gold_source_url."""
    items = load_questions()
    eligible = [i for i in items if i["stratum"] in ("factual", "multi_hop")]
    annotated = [i for i in eligible if i.get("gold_source_urls")]
    rate = len(annotated) / len(eligible) if eligible else 1.0
    assert rate >= 0.90, (
        f"Gold annotation rate {rate:.1%} < 90% "
        f"({len(annotated)}/{len(eligible)} items)"
    )


def test_gold_url_tail_segment_appears_in_expected_answer():
    """For each item with gold_source_urls, at least one URL's last path segment
    must share a content word with the question, expected_answer, or history
    (case-insensitive, with simple prefix matching for plurals/possessives).
    Catches obvious wrong-page typos while permitting multi-URL gold sets where
    some URLs cite broader umbrella pages. Items can opt out by providing an
    explicit empty list."""
    stopwords = {"of", "the", "and", "a", "an", "in", "on", "to", "for", "with"}

    for item in load_questions():
        urls = item.get("gold_source_urls") or []
        if not urls:
            continue
        haystack_parts = [item.get("question", ""), item.get("expected_answer", "")]
        for message in item.get("history") or []:
            haystack_parts.append(message.get("content", ""))
        haystack_words = set(re.findall(r"[a-z]+", " ".join(haystack_parts).lower()))

        def matches(seg_word: str) -> bool:
            return any(
                hw == seg_word or hw.startswith(seg_word) or seg_word.startswith(hw)
                for hw in haystack_words
                if len(hw) >= 3
            )

        any_match = False
        for url in urls:
            segment = unquote(url.rstrip("/").split("/")[-1]).replace("_", " ").lower()
            if not segment:
                continue
            seg_words = [
                w for w in re.findall(r"[a-z]+", segment)
                if w not in stopwords and len(w) >= 3
            ]
            if any(matches(w) for w in seg_words):
                any_match = True
                break
        assert any_match, (
            f"No gold URL tail shares a word with question/answer/history for {item['id']}"
        )
