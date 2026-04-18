import json
from pathlib import Path

import pytest

from app.eval.judge import AnswerJudgment
from app.eval.report import default_report_path, write_report
from app.eval.runner import EvalExample, load_dataset, run_eval
from app.rag.pipeline import RAGResponse


def test_load_dataset_reads_jsonl(tmp_path):
    dataset_path = tmp_path / "sample.jsonl"
    dataset_path.write_text(
        '\n'.join(
            [
                json.dumps(
                    {
                        "id": "q-1",
                        "question": "Who is Nyx?",
                        "expected_answer": "Nyx is the Goddess of Night.",
                        "stratum": "factual",
                        "spoiler_tier": 0,
                        "history": [{"role": "user", "content": "Tell me about her."}],
                        "gold_source_urls": ["https://hades.fandom.com/wiki/Nyx"],
                    }
                ),
                "",
            ]
        )
    )

    examples = load_dataset(dataset_path)

    assert examples == [
        EvalExample(
            id="q-1",
            question="Who is Nyx?",
            expected_answer="Nyx is the Goddess of Night.",
            stratum="factual",
            spoiler_tier=0,
            history=[{"role": "user", "content": "Tell me about her."}],
            gold_source_urls=["https://hades.fandom.com/wiki/Nyx"],
        )
    ]


@pytest.mark.asyncio
async def test_run_eval_returns_metrics_and_results(tmp_path):
    examples = [
        EvalExample(
            id="q-1",
            question="Who is Nyx?",
            expected_answer="Nyx is the Goddess of Night.",
            stratum="factual",
            spoiler_tier=0,
            history=[{"role": "user", "content": "Tell me about her."}],
            gold_source_urls=["https://hades.fandom.com/wiki/Nyx"],
        )
    ]

    async def _answer(_example: EvalExample) -> RAGResponse:
        return RAGResponse(
            answer="Nyx is the Goddess of Night. [1]",
            passages=[
                {
                    "passage_id": 1,
                    "content": "Nyx\n\nNyx is the Goddess of Night.",
                    "source_url": "https://hades.fandom.com/wiki/Nyx",
                }
            ],
        )

    async def _judge(_example: EvalExample, _response: RAGResponse) -> AnswerJudgment:
        return AnswerJudgment(
            faithful=True,
            answer_correct=True,
            refusal_appropriate=None,
        )

    report = await run_eval(
        game_slug="hades",
        dataset_path=tmp_path / "sample.jsonl",
        examples=examples,
        answer_fn=_answer,
        run_name="test-run",
        judge_fn=_judge,
    )

    assert report["run_name"] == "test-run"
    assert report["metrics"]["dataset_size"] == 1
    assert report["metrics"]["citation_rate"] == 1.0
    assert report["metrics"]["retrieval_recall_at_5_mean"] == 1.0
    assert report["metrics"]["retrieval_recall_at_5_exact_url_mean"] == 1.0
    assert report["metrics"]["annotated_citation_examples"] == 1
    assert report["metrics"]["citation_validity_rate"] == 1.0
    assert report["metrics"]["citation_validity_exact_url_rate"] == 1.0
    assert report["metrics"]["faithfulness_rate"] == 1.0
    assert report["results"][0]["exact_match"] is True
    assert report["results"][0]["passages"][0]["passage_id"] == 1


@pytest.mark.asyncio
async def test_run_eval_normalizes_citation_numbers_before_scoring(tmp_path):
    examples = [
        EvalExample(
            id="q-2",
            question="Who helps Zagreus?",
            expected_answer="Nyx and Achilles help Zagreus.",
            stratum="factual",
            spoiler_tier=0,
            gold_source_urls=[
                "https://hades.fandom.com/wiki/Nyx",
                "https://hades.fandom.com/wiki/Achilles",
            ],
        )
    ]

    async def _answer(_example: EvalExample) -> RAGResponse:
        return RAGResponse(
            answer="Nyx and Achilles help Zagreus. [3][1]",
            passages=[
                {
                    "passage_id": 11,
                    "content": "Achilles\n\nAchilles helps Zagreus.",
                    "source_url": "https://hades.fandom.com/wiki/Achilles",
                },
                {
                    "passage_id": 22,
                    "content": "Hypnos\n\nHypnos greets Zagreus.",
                    "source_url": "https://hades.fandom.com/wiki/Hypnos",
                },
                {
                    "passage_id": 33,
                    "content": "Nyx\n\nNyx helps Zagreus.",
                    "source_url": "https://hades.fandom.com/wiki/Nyx",
                },
            ],
        )

    report = await run_eval(
        game_slug="hades",
        dataset_path=tmp_path / "sample.jsonl",
        examples=examples,
        answer_fn=_answer,
        run_name="renumbered-citations",
    )

    assert report["results"][0]["answer"] == "Nyx and Achilles help Zagreus. [1][2]"
    assert report["results"][0]["cited_indices"] == [1, 2]
    assert report["results"][0]["citation_validity"] is True
    assert report["results"][0]["citations"] == [
        {
            "index": 1,
            "passage_id": 33,
            "source_url": "https://hades.fandom.com/wiki/Nyx",
            "title": "Nyx",
        },
        {
            "index": 2,
            "passage_id": 11,
            "source_url": "https://hades.fandom.com/wiki/Achilles",
            "title": "Achilles",
        },
    ]


@pytest.mark.asyncio
async def test_run_eval_dedupes_same_source_url_citations(tmp_path):
    examples = [
        EvalExample(
            id="q-3",
            question="Who is Zagreus?",
            expected_answer="Zagreus is the Prince of the Underworld.",
            stratum="factual",
            spoiler_tier=0,
            gold_source_urls=["https://hades.fandom.com/wiki/Zagreus"],
        )
    ]

    async def _answer(_example: EvalExample) -> RAGResponse:
        return RAGResponse(
            answer="Zagreus is the Prince of the Underworld. [1][2][3]",
            passages=[
                {
                    "passage_id": 11,
                    "content": "Zagreus\n\nZagreus is the Prince of the Underworld.",
                    "source_url": "https://hades.fandom.com/wiki/Zagreus",
                },
                {
                    "passage_id": 22,
                    "content": "Zagreus\n\nZagreus is a Cthonic God.",
                    "source_url": "https://hades.fandom.com/wiki/Zagreus",
                },
                {
                    "passage_id": 33,
                    "content": "Zagreus\n\nZagreus is the son of Hades.",
                    "source_url": "https://hades.fandom.com/wiki/Zagreus",
                },
            ],
        )

    report = await run_eval(
        game_slug="hades",
        dataset_path=tmp_path / "sample.jsonl",
        examples=examples,
        answer_fn=_answer,
        run_name="deduped-source-citations",
    )

    assert report["results"][0]["answer"] == "Zagreus is the Prince of the Underworld. [1]"
    assert report["results"][0]["cited_indices"] == [1]
    assert report["results"][0]["citation_validity"] is True
    assert report["results"][0]["citations"] == [
        {
            "index": 1,
            "passage_id": 11,
            "source_url": "https://hades.fandom.com/wiki/Zagreus",
            "title": "Zagreus",
        }
    ]


@pytest.mark.asyncio
async def test_run_eval_uses_canonical_source_identity_for_alias_urls(tmp_path):
    examples = [
        EvalExample(
            id="q-alias",
            question="What is the ending of Hades?",
            expected_answer="Persephone returns and the family reconciles.",
            stratum="ambiguous",
            spoiler_tier=3,
            gold_source_urls=[
                "https://hades.fandom.com/wiki/Persephone",
                "https://hades.fandom.com/wiki/Hades_(character)",
            ],
        )
    ]

    async def _answer(_example: EvalExample) -> RAGResponse:
        return RAGResponse(
            answer="Persephone returns and the family reconciles. [1][2]",
            passages=[
                {
                    "passage_id": 11,
                    "content": "Persephone\n\nPersephone returns to the House.",
                    "source_url": "https://hades.fandom.com/wiki/Persephone",
                },
                {
                    "passage_id": 22,
                    "content": "Hades\n\nHades reconciles with Persephone.",
                    "source_url": "https://hades.fandom.com/wiki/Hades",
                },
            ],
        )

    report = await run_eval(
        game_slug="hades",
        dataset_path=tmp_path / "sample.jsonl",
        examples=examples,
        answer_fn=_answer,
        run_name="canonical-source-identity",
    )

    assert report["metrics"]["retrieval_recall_at_5_mean"] == 1.0
    assert report["metrics"]["retrieval_recall_at_5_exact_url_mean"] == 0.5
    assert report["metrics"]["citation_validity_rate"] == 1.0
    assert report["metrics"]["citation_validity_exact_url_rate"] == 0.0
    assert set(report["results"][0]["gold_source_ids"]) == {"persephone", "hades"}
    assert set(report["results"][0]["retrieved_source_ids"]) == {"persephone", "hades"}
    assert report["results"][0]["unresolved_gold_source_urls"] == []
    assert report["results"][0]["citation_validity"] is True
    assert report["results"][0]["citation_validity_exact_url"] is False


@pytest.mark.asyncio
async def test_run_eval_marks_canonical_citation_invalid_when_cited_url_is_unresolved(tmp_path):
    examples = [
        EvalExample(
            id="q-unresolved-citation",
            question="Who is Nyx?",
            expected_answer="Nyx is the Goddess of Night.",
            stratum="factual",
            spoiler_tier=0,
            gold_source_urls=["https://hades.fandom.com/wiki/Nyx"],
        )
    ]

    async def _answer(_example: EvalExample) -> RAGResponse:
        return RAGResponse(
            answer="Nyx is the Goddess of Night. [1]",
            passages=[
                {
                    "passage_id": 1,
                    "content": "Nyx\n\nNyx is the Goddess of Night.",
                    "source_url": "https://example.com/not-nyx",
                }
            ],
        )

    report = await run_eval(
        game_slug="hades",
        dataset_path=tmp_path / "sample.jsonl",
        examples=examples,
        answer_fn=_answer,
        run_name="unresolved-canonical-citation",
    )

    assert report["results"][0]["citation_validity"] is False
    assert report["results"][0]["citation_validity_exact_url"] is False


@pytest.mark.asyncio
async def test_run_eval_leaves_citation_validity_unset_without_gold_sources(tmp_path):
    examples = [
        EvalExample(
            id="q-5",
            question="Who is Nyx?",
            expected_answer="Nyx is the Goddess of Night.",
            stratum="factual",
            spoiler_tier=0,
        )
    ]

    async def _answer(_example: EvalExample) -> RAGResponse:
        return RAGResponse(
            answer="Nyx is the Goddess of Night. [1]",
            passages=[
                {
                    "passage_id": 1,
                    "content": "Wrong Source\n\nNot about Nyx.",
                    "source_url": "https://example.com/wrong",
                }
            ],
        )

    report = await run_eval(
        game_slug="hades",
        dataset_path=tmp_path / "sample.jsonl",
        examples=examples,
        answer_fn=_answer,
        run_name="citation-unknown-without-gold",
    )

    assert report["metrics"]["annotated_retrieval_examples"] == 0
    assert report["metrics"]["annotated_citation_examples"] == 0
    assert report["metrics"]["citation_validity_rate"] is None
    assert report["results"][0]["citation_validity"] is None


@pytest.mark.asyncio
async def test_run_eval_tracks_refusal_metrics(tmp_path):
    examples = [
        EvalExample(
            id="q-4",
            question="Does Hades have a sequel?",
            expected_answer=(
                "The ingested Hades lore passages do not provide enough evidence to answer "
                "sequel-release questions."
            ),
            stratum="ambiguous",
            spoiler_tier=0,
            expects_refusal=True,
        )
    ]

    async def _answer(_example: EvalExample) -> RAGResponse:
        return RAGResponse(
            answer=(
                "The ingested Hades lore passages do not provide enough evidence to answer "
                "sequel-release questions."
            ),
            passages=[],
        )

    async def _judge(_example: EvalExample, _response: RAGResponse) -> AnswerJudgment:
        return AnswerJudgment(
            faithful=True,
            answer_correct=True,
            refusal_appropriate=True,
        )

    report = await run_eval(
        game_slug="hades",
        dataset_path=tmp_path / "sample.jsonl",
        examples=examples,
        answer_fn=_answer,
        run_name="refusal-metrics",
        judge_fn=_judge,
    )

    assert report["metrics"]["refusal_examples"] == 1
    assert report["metrics"]["refusal_appropriateness_rate"] == 1.0
    assert report["results"][0]["refusal_appropriate"] is True


def test_write_report_persists_json(tmp_path):
    report = {"run_name": "demo", "metrics": {"dataset_size": 1}, "results": []}
    output_path = tmp_path / "reports" / "demo.json"

    write_report(report, output_path)

    assert json.loads(output_path.read_text())["run_name"] == "demo"


def test_default_report_path_uses_game_slug():
    path = default_report_path("hades", reports_dir=Path("custom_reports"))
    assert path.parent == Path("custom_reports")
    assert path.name.startswith("hades-")
    assert path.suffix == ".json"
