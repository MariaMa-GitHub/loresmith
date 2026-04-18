from app.rag.citations import (
    normalize_answer_citations,
    parse_inline_citation_indices,
    strip_inline_citations,
)


def test_parse_inline_citation_indices_supports_grouped_citations():
    text = "Nyx rules [2, 4] and schemes [4][1]."

    assert parse_inline_citation_indices(text) == [2, 4, 1]


def test_strip_inline_citations_removes_grouped_and_single_citations():
    text = "Nyx [2, 4] is powerful [1]."
    stripped = strip_inline_citations(text)

    assert "[" not in stripped
    assert "]" not in stripped
    assert "Nyx" in stripped
    assert "is powerful" in stripped


def test_normalize_answer_citations_renumbers_by_first_reference_order():
    bundle = normalize_answer_citations(
        "Nyx schemes [3][1] and rules [3, 2].",
        passages=[
            {"passage_id": 10, "source_url": "https://x.com/1", "content": "Source One\n\nbody"},
            {"passage_id": 20, "source_url": "https://x.com/2", "content": "Source Two\n\nbody"},
            {"passage_id": 30, "source_url": "https://x.com/3", "content": "Source Three\n\nbody"},
        ],
    )

    assert bundle.answer == "Nyx schemes and rules. [1][2][3]"
    assert bundle.citations == [
        {"index": 1, "passage_id": 30, "source_url": "https://x.com/3", "title": "Source Three"},
        {"index": 2, "passage_id": 10, "source_url": "https://x.com/1", "title": "Source One"},
        {"index": 3, "passage_id": 20, "source_url": "https://x.com/2", "title": "Source Two"},
    ]


def test_normalize_answer_citations_can_fix_stored_history_rows():
    bundle = normalize_answer_citations(
        "Nyx [4] opposes Hades [2].",
        citations=[
            {"index": 2, "passage_id": 20, "source_url": "https://x.com/2", "title": "Hades"},
            {"index": 4, "passage_id": 40, "source_url": "https://x.com/4", "title": "Nyx"},
            {"index": 5, "passage_id": 50, "source_url": "https://x.com/5", "title": "Unused"},
        ],
    )

    assert bundle.answer == "Nyx opposes Hades. [1][2]"
    assert bundle.citations == [
        {"index": 1, "passage_id": 40, "source_url": "https://x.com/4", "title": "Nyx"},
        {"index": 2, "passage_id": 20, "source_url": "https://x.com/2", "title": "Hades"},
    ]


def test_normalize_answer_citations_leaves_plain_bracket_text_alone():
    bundle = normalize_answer_citations("What happened in chamber [2]?")

    assert bundle.answer == "What happened in chamber [2]?"
    assert bundle.citations == []


def test_normalize_answer_citations_dedupes_same_source_url():
    bundle = normalize_answer_citations(
        "Zagreus is important [1][2][3].",
        passages=[
            {"passage_id": 10, "source_url": "https://x.com/zagreus", "content": "Zagreus\n\nbody"},
            {"passage_id": 20, "source_url": "https://x.com/zagreus", "content": "Zagreus\n\nbody"},
            {"passage_id": 30, "source_url": "https://x.com/zagreus", "content": "Zagreus\n\nbody"},
        ],
    )

    assert bundle.answer == "Zagreus is important. [1]"
    assert bundle.citations == [
        {"index": 1, "passage_id": 10, "source_url": "https://x.com/zagreus", "title": "Zagreus"},
    ]


def test_normalize_answer_citations_dedupes_duplicate_history_sources():
    bundle = normalize_answer_citations(
        "Zagreus [3][1] fights Hades [2].",
        citations=[
            {
                "index": 1,
                "passage_id": 10,
                "source_url": "https://x.com/zagreus",
                "title": "Zagreus",
            },
            {
                "index": 2,
                "passage_id": 20,
                "source_url": "https://x.com/hades",
                "title": "Hades",
            },
            {
                "index": 3,
                "passage_id": 30,
                "source_url": "https://x.com/zagreus",
                "title": "Zagreus",
            },
        ],
    )

    assert bundle.answer == "Zagreus fights Hades. [1][2]"
    assert bundle.citations == [
        {"index": 1, "passage_id": 30, "source_url": "https://x.com/zagreus", "title": "Zagreus"},
        {"index": 2, "passage_id": 20, "source_url": "https://x.com/hades", "title": "Hades"},
    ]


def test_normalize_answer_citations_collapses_repeated_paragraph_citations():
    bundle = normalize_answer_citations(
        "Zagreus is the prince [1]. He escapes the Underworld [1]. Hermes helps him [2].",
        passages=[
            {"passage_id": 10, "source_url": "https://x.com/zagreus", "content": "Zagreus\n\nbody"},
            {"passage_id": 20, "source_url": "https://x.com/hermes", "content": "Hermes\n\nbody"},
        ],
    )

    assert bundle.answer == (
        "Zagreus is the prince. He escapes the Underworld. Hermes helps him. [1][2]"
    )
    assert bundle.citations == [
        {"index": 1, "passage_id": 10, "source_url": "https://x.com/zagreus", "title": "Zagreus"},
        {"index": 2, "passage_id": 20, "source_url": "https://x.com/hermes", "title": "Hermes"},
    ]
