import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

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
    (case-insensitive, with simple prefix matching for plurals/possessives)."""
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
