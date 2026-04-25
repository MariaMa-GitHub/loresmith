from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.db.models import EvalRun
from app.db.session import get_session_factory
from app.eval.judge import AnswerJudgment, judge_answer
from app.eval.metrics import (
    exact_match,
    has_inline_citation,
    token_f1,
    token_recall,
)
from app.eval.report import build_report, default_report_path, write_report
from app.eval.source_identity import resolve_source_identities, resolve_source_identity
from app.games import GAME_DISPLAY
from app.llm.base import TaskType
from app.rag.citations import normalize_answer_citations, parse_inline_citation_indices
from app.rag.pipeline import RAGPipeline, RAGResponse
from app.rag.rewriter import QueryRewriter
from app.services import build_bm25, build_services, resolve_corpus_revision_key

_DATASETS_DIR = Path(__file__).parent / "datasets"
_DEFAULT_DATASETS = {
    "hades": _DATASETS_DIR / "hades.jsonl",
    "hades2": _DATASETS_DIR / "hades2.jsonl",
}


@dataclass(frozen=True)
class EvalExample:
    id: str
    question: str
    expected_answer: str
    stratum: str
    spoiler_tier: int
    history: list[dict[str, str]] = field(default_factory=list)
    gold_source_urls: list[str] = field(default_factory=list)
    expects_refusal: bool | None = None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _bool_rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def load_dataset(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        examples.append(EvalExample(**payload))
    return examples


async def run_eval(
    *,
    game_slug: str,
    dataset_path: Path,
    examples: Sequence[EvalExample],
    answer_fn: Callable[[EvalExample], Awaitable[RAGResponse]],
    run_name: str,
    judge_fn: Callable[[EvalExample, RAGResponse], Awaitable[AnswerJudgment | None]] | None = None,
) -> dict:
    results: list[dict] = []

    for example in examples:
        started = perf_counter()
        response = await answer_fn(example)
        normalized = normalize_answer_citations(
            response.answer,
            passages=response.passages,
            citations=response.citations,
        )
        normalized_response = RAGResponse(
            answer=normalized.answer,
            passages=response.passages,
            citations=normalized.citations,
        )
        latency_ms = round((perf_counter() - started) * 1000, 2)
        context_text = "\n".join(
            passage["content"] for passage in normalized_response.passages
        )
        cited_indices = parse_inline_citation_indices(normalized_response.answer)
        retrieved_source_urls = [
            passage["source_url"] for passage in normalized_response.passages
        ]
        gold_source_urls = sorted(set(example.gold_source_urls))
        gold_source_identities = resolve_source_identities(game_slug, gold_source_urls)
        retrieved_source_ids = [
            source_id
            for url in retrieved_source_urls
            if (source_id := resolve_source_identity(game_slug, url)) is not None
        ]

        retrieval_recall_at_5_exact_url: float | None = None
        if gold_source_urls:
            retrieved_set = set(retrieved_source_urls[:5])
            retrieval_recall_at_5_exact_url = round(
                len(retrieved_set & set(gold_source_urls)) / len(gold_source_urls),
                4,
            )

        retrieval_recall_at_5: float | None = None
        if gold_source_urls and not gold_source_identities.unresolved_urls:
            retrieved_id_set = set(retrieved_source_ids[:5])
            retrieval_recall_at_5 = round(
                len(retrieved_id_set & set(gold_source_identities.resolved_ids))
                / len(gold_source_identities.resolved_ids),
                4,
            )

        citation_validity_exact_url: bool | None = None
        if gold_source_urls:
            citation_validity_exact_url = bool(cited_indices)
            if citation_validity_exact_url:
                citation_validity_exact_url = len(cited_indices) == len(
                    normalized_response.citations
                )
            if citation_validity_exact_url:
                cited_urls = {
                    citation["source_url"] for citation in normalized_response.citations
                }
                citation_validity_exact_url = cited_urls.issubset(set(gold_source_urls))

        citation_validity: bool | None = None
        if gold_source_urls and not gold_source_identities.unresolved_urls:
            citation_validity = bool(cited_indices)
            if citation_validity:
                citation_validity = len(cited_indices) == len(normalized_response.citations)
            if citation_validity:
                cited_source_ids = []
                for citation in normalized_response.citations:
                    source_id = resolve_source_identity(
                        game_slug,
                        citation["source_url"],
                    )
                    if source_id is None:
                        citation_validity = False
                        break
                    cited_source_ids.append(source_id)
            if citation_validity:
                cited_source_ids_set = set(cited_source_ids)
                citation_validity = cited_source_ids_set.issubset(
                    set(gold_source_identities.resolved_ids)
                )

        judgment = (
            await judge_fn(example, normalized_response)
            if judge_fn
            else None
        )

        results.append(
            {
                **asdict(example),
                "answer": normalized_response.answer,
                "latency_ms": latency_ms,
                "num_passages": len(normalized_response.passages),
                "has_inline_citation": has_inline_citation(normalized_response.answer),
                "cited_indices": cited_indices,
                "retrieved_source_urls": retrieved_source_urls,
                "retrieved_source_ids": retrieved_source_ids,
                "gold_source_ids": gold_source_identities.resolved_ids,
                "unresolved_gold_source_urls": gold_source_identities.unresolved_urls,
                "retrieval_recall_at_5": retrieval_recall_at_5,
                "retrieval_recall_at_5_exact_url": retrieval_recall_at_5_exact_url,
                "citation_validity": citation_validity,
                "citation_validity_exact_url": citation_validity_exact_url,
                "exact_match": exact_match(
                    example.expected_answer,
                    normalized_response.answer,
                ),
                "token_f1": round(
                    token_f1(example.expected_answer, normalized_response.answer),
                    4,
                ),
                "context_answer_token_recall_proxy": round(
                    token_recall(example.expected_answer, context_text),
                    4,
                ),
                "faithful": judgment.faithful if judgment else None,
                "answer_correct": judgment.answer_correct if judgment else None,
                "refusal_appropriate": judgment.refusal_appropriate if judgment else None,
                "passages": normalized_response.passages,
                "citations": normalized_response.citations,
            }
        )

    size = len(results) or 1
    retrieval_values = [
        value for item in results
        if (value := item["retrieval_recall_at_5"]) is not None
    ]
    exact_url_retrieval_values = [
        value for item in results
        if (value := item["retrieval_recall_at_5_exact_url"]) is not None
    ]
    citation_values = [
        item["citation_validity"]
        for item in results
        if item["citation_validity"] is not None
    ]
    exact_url_citation_values = [
        item["citation_validity_exact_url"]
        for item in results
        if item["citation_validity_exact_url"] is not None
    ]
    faithfulness_values = [
        item["faithful"] for item in results if item["faithful"] is not None
    ]
    correctness_values = [
        item["answer_correct"] for item in results if item["answer_correct"] is not None
    ]
    refusal_values = [
        item["refusal_appropriate"]
        for item in results
        if item["refusal_appropriate"] is not None
    ]
    metrics = {
        "dataset_size": len(results),
        "annotated_retrieval_examples": len(retrieval_values),
        "annotated_citation_examples": len(citation_values),
        "exact_match_rate": round(
            sum(1 for item in results if item["exact_match"]) / size,
            4,
        ),
        "token_f1_mean": round(sum(item["token_f1"] for item in results) / size, 4),
        "citation_rate": round(
            sum(1 for item in results if item["has_inline_citation"]) / size,
            4,
        ),
        "context_answer_token_recall_proxy_mean": round(
            sum(item["context_answer_token_recall_proxy"] for item in results) / size,
            4,
        ),
        "avg_latency_ms": round(sum(item["latency_ms"] for item in results) / size, 2),
        "avg_context_passages": round(sum(item["num_passages"] for item in results) / size, 2),
        "retrieval_recall_at_5_mean": _mean(retrieval_values),
        "retrieval_recall_at_5_exact_url_mean": _mean(exact_url_retrieval_values),
        "citation_validity_rate": _bool_rate(citation_values),
        "citation_validity_exact_url_rate": _bool_rate(exact_url_citation_values),
        "judged_examples": len(faithfulness_values),
        "faithfulness_rate": _bool_rate(faithfulness_values),
        "answer_correctness_rate": _bool_rate(correctness_values),
        "refusal_examples": len(refusal_values),
        "refusal_appropriateness_rate": _bool_rate(refusal_values),
    }
    return build_report(
        run_name=run_name,
        game_slug=game_slug,
        dataset_path=dataset_path,
        metrics=metrics,
        results=results,
    )


async def run_pipeline_eval(
    *,
    game_slug: str,
    dataset_path: Path,
    output_path: Path,
    limit: int | None = None,
    run_name: str | None = None,
) -> dict:
    services = build_services()
    session_factory = get_session_factory()
    examples = load_dataset(dataset_path)
    if limit is not None:
        examples = examples[:limit]

    resolved_run_name = run_name or (
        f"{game_slug}-eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )

    try:
        async with session_factory() as session:
            bm25, source_map = await build_bm25(session, game_slug)
            pipeline = RAGPipeline(
                embedder=services.embedder,
                bm25_index=bm25,
                dense_retriever=services.dense,
                llm=services.router.for_task(TaskType.ANSWER),
                game_slug=game_slug,
                game_display_name=GAME_DISPLAY[game_slug],
                tracer=services.tracer,
                bm25_source_map=source_map,
                rewriter=QueryRewriter(
                    llm=services.router.for_task(TaskType.REWRITE),
                    tracer=services.tracer,
                ),
                reranker=services.reranker,
                semantic_cache=services.semantic_cache,
                corpus_revision_fn=resolve_corpus_revision_key,
                retrieve_top_k=services.settings.retrieval_top_k_per_method,
                rerank_candidates=services.settings.rerank_candidates,
                final_top_k=services.settings.retrieval_top_k_final,
            )
            judge_llm = services.router.for_task(TaskType.VERIFY)

            async def _answer(example: EvalExample) -> RAGResponse:
                return await pipeline.answer(
                    session=session,
                    question=example.question,
                    max_spoiler_tier=3,
                    history=example.history,
                )

            async def _judge(example: EvalExample, response: RAGResponse) -> AnswerJudgment:
                return await judge_answer(
                    llm=judge_llm,
                    question=example.question,
                    expected_answer=example.expected_answer,
                    actual_answer=response.answer,
                    passages=response.passages,
                    expects_refusal=example.expects_refusal,
                )

            report = await run_eval(
                game_slug=game_slug,
                dataset_path=dataset_path,
                examples=examples,
                answer_fn=_answer,
                run_name=resolved_run_name,
                judge_fn=_judge,
            )

            session.add(
                EvalRun(
                    run_name=resolved_run_name,
                    game_slug=game_slug,
                    metrics=report["metrics"],
                    report_path=str(output_path),
                )
            )
            await session.commit()
    finally:
        services.tracer.flush()

    write_report(report, output_path)
    return report


def _default_dataset_for_game(game_slug: str) -> Path:
    try:
        return _DEFAULT_DATASETS[game_slug]
    except KeyError as exc:
        raise SystemExit(f"No default dataset is configured for game '{game_slug}'") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Loresmith eval harness")
    parser.add_argument("--game", required=True, choices=sorted(GAME_DISPLAY))
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to a JSONL dataset. Defaults to the built-in dataset for the game.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Where to write the JSON report. Defaults to eval_reports/<game>-<timestamp>.json.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of examples.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional eval run name.")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    dataset_path = args.dataset or _default_dataset_for_game(args.game)
    output_path = args.out or default_report_path(args.game)
    report = await run_pipeline_eval(
        game_slug=args.game,
        dataset_path=dataset_path,
        output_path=output_path,
        limit=args.limit,
        run_name=args.run_name,
    )
    print(f"Eval run:         {report['run_name']}")
    print(f"Examples scored:  {report['metrics']['dataset_size']}")
    print(f"Token F1 mean:    {report['metrics']['token_f1_mean']}")
    print(f"Recall@5 mean:    {report['metrics']['retrieval_recall_at_5_mean']}")
    print(f"Citation rate:    {report['metrics']['citation_rate']}")
    print(f"Citation valid:   {report['metrics']['citation_validity_rate']}")
    print(f"Faithfulness:     {report['metrics']['faithfulness_rate']}")
    print(f"Report written:   {output_path}")


if __name__ == "__main__":
    asyncio.run(_main())
