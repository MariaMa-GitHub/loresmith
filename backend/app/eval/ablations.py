from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete

from app.db.models import SemanticCache as SemanticCacheRow
from app.db.session import get_session_factory
from app.eval.judge import judge_answer
from app.eval.runner import load_dataset, run_eval
from app.games import ADAPTERS, GAME_DISPLAY
from app.llm.base import TaskType
from app.llm.tools import ToolDispatcher, build_default_tools
from app.rag.pipeline import RAGPipeline, RAGResponse
from app.rag.rewriter import QueryRewriter
from app.rag.verifier import Verifier
from app.retrieval.reranker import NullReranker
from app.services import build_bm25, build_services, resolve_corpus_revision_key

_VALID_RETRIEVAL = {"hybrid", "bm25_only", "dense_only"}
_DATASETS_DIR = Path(__file__).parent / "datasets"
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class AblationConfig:
    config_id: str
    rewriter: bool = False
    rerank: bool = False
    verifier: bool = False
    cache: bool = False
    tools: bool = False
    retrieval: str = "hybrid"

    def __post_init__(self):
        if self.retrieval not in _VALID_RETRIEVAL:
            raise ValueError(
                f"Unknown retrieval mode {self.retrieval!r}; "
                f"expected one of {sorted(_VALID_RETRIEVAL)}"
            )


@dataclass
class AblationResult:
    config_id: str
    metrics: dict
    output_path: Path
    skipped_reason: str | None = None


def default_matrix() -> list[AblationConfig]:
    return [
        AblationConfig("baseline"),
        AblationConfig("+rewriter", rewriter=True),
        AblationConfig("+rerank", rerank=True),
        AblationConfig("+verifier", verifier=True),
        AblationConfig("+cache", cache=True),
        AblationConfig("+tools", tools=True),
        AblationConfig("full", rewriter=True, rerank=True, verifier=True, cache=True, tools=True),
        AblationConfig("hybrid-no-dense", retrieval="bm25_only"),
        AblationConfig("hybrid-no-bm25", retrieval="dense_only"),
    ]


def render_markdown_report(*, game_slug: str, rows: list[dict]) -> str:
    lines = [
        f"# Eval report — {game_slug} ({datetime.now(UTC).isoformat()})",
        "",
        "| config | faithfulness | recall@5 | citation_valid | correctness | avg_latency_ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    def fmt(x):
        return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))

    for row in rows:
        m = row.get("metrics") or {}
        label = row["config_id"]
        if row.get("skipped_reason"):
            label = f"{label} (skipped: {row['skipped_reason']})"
        lines.append(
            f"| {label} "
            f"| {fmt(m.get('faithfulness_rate'))} "
            f"| {fmt(m.get('retrieval_recall_at_5_mean'))} "
            f"| {fmt(m.get('citation_validity_rate'))} "
            f"| {fmt(m.get('answer_correctness_rate'))} "
            f"| {fmt(m.get('avg_latency_ms'))} |"
        )
    return "\n".join(lines) + "\n"


def provider_supports_tools(provider) -> bool:
    return hasattr(provider, "complete_with_tools")


async def _build_pipeline(
    *,
    services,
    session,
    game_slug: str,
    config: AblationConfig,
) -> RAGPipeline:
    bm25, source_map = await build_bm25(session, game_slug)

    class _EmptyBM25:
        def search(self, *a, **kw):
            return []

    class _EmptyDense:
        async def search(self, **kw):
            return []

    bm25_index = bm25 if config.retrieval in {"hybrid", "bm25_only"} else _EmptyBM25()
    dense = services.dense if config.retrieval in {"hybrid", "dense_only"} else _EmptyDense()

    reranker = services.reranker if config.rerank else NullReranker()
    rewriter = (
        QueryRewriter(llm=services.router.for_task(TaskType.REWRITE), tracer=services.tracer)
        if config.rewriter
        else None
    )
    verifier = (
        Verifier(llm=services.router.for_task(TaskType.VERIFY), tracer=services.tracer)
        if config.verifier
        else None
    )

    entity_schema = ADAPTERS[game_slug].entity_schema
    allowed_entity_types = {t.name for t in entity_schema}
    tool_dispatcher = (
        ToolDispatcher(game_slug=game_slug, allowed_entity_types=allowed_entity_types)
        if config.tools and allowed_entity_types
        else None
    )
    tool_defs = (
        build_default_tools(game_slug=game_slug, entity_schema=entity_schema)
        if tool_dispatcher
        else None
    )

    cache = services.semantic_cache if config.cache else None

    return RAGPipeline(
        embedder=services.embedder,
        bm25_index=bm25_index,
        dense_retriever=dense,
        llm=services.router.for_task(TaskType.ANSWER),
        game_slug=game_slug,
        game_display_name=GAME_DISPLAY[game_slug],
        tracer=services.tracer,
        bm25_source_map=source_map,
        rewriter=rewriter,
        reranker=reranker,
        verifier=verifier,
        semantic_cache=cache,
        corpus_revision_fn=resolve_corpus_revision_key,
        tool_dispatcher=tool_dispatcher,
        tool_definitions=tool_defs,
        tool_loop_max_iters=services.settings.tool_loop_max_iters,
        retrieve_top_k=services.settings.retrieval_top_k_per_method,
        rerank_candidates=services.settings.rerank_candidates,
        final_top_k=services.settings.retrieval_top_k_final,
    )


async def _reset_cache_scope(session, *, game_slug: str, corpus_revision: str) -> None:
    await session.execute(
        delete(SemanticCacheRow)
        .where(SemanticCacheRow.game_slug == game_slug)
        .where(SemanticCacheRow.corpus_revision == corpus_revision)
    )
    await session.commit()


async def run_matrix(
    *,
    game_slug: str,
    dataset_path: Path,
    output_dir: Path,
    configs: list[AblationConfig] | None = None,
    limit: int | None = None,
) -> list[AblationResult]:
    configs = configs or default_matrix()
    services = build_services()
    session_factory = get_session_factory()
    examples = load_dataset(dataset_path)
    answer_provider = services.router.for_task(TaskType.ANSWER)
    if limit is not None:
        examples = examples[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AblationResult] = []

    try:
        async with session_factory() as session:
            for config in configs:
                if config.tools and not provider_supports_tools(answer_provider):
                    skipped_reason = "answer provider lacks complete_with_tools support"
                    out_path = output_dir / f"{game_slug}-ablation-{config.config_id}.json"
                    out_path.write_text(
                        json.dumps(
                            {
                                "config_id": config.config_id,
                                "game_slug": game_slug,
                                "skipped_reason": skipped_reason,
                            },
                            indent=2,
                        )
                    )
                    results.append(
                        AblationResult(
                            config_id=config.config_id,
                            metrics={},
                            output_path=out_path,
                            skipped_reason=skipped_reason,
                        )
                    )
                    continue

                if config.cache:
                    revision = await resolve_corpus_revision_key(session, game_slug)
                    await _reset_cache_scope(
                        session, game_slug=game_slug, corpus_revision=revision
                    )

                pipeline = await _build_pipeline(
                    services=services, session=session, game_slug=game_slug, config=config
                )
                judge_llm = services.router.for_task(TaskType.VERIFY)

                async def _answer(example, _pipeline=pipeline, _session=session):
                    return await _pipeline.answer(
                        session=_session,
                        question=example.question,
                        max_spoiler_tier=example.spoiler_tier,
                        history=example.history,
                    )

                async def _judge(example, response: RAGResponse, _llm=judge_llm):
                    return await judge_answer(
                        llm=_llm,
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
                    run_name=f"{game_slug}-ablation-{config.config_id}",
                    judge_fn=_judge,
                )
                out_path = output_dir / f"{game_slug}-ablation-{config.config_id}.json"
                out_path.write_text(json.dumps(report, indent=2))
                results.append(
                    AblationResult(
                        config_id=config.config_id,
                        metrics=report["metrics"],
                        output_path=out_path,
                    )
                )
    finally:
        services.tracer.flush()
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description="Loresmith ablation harness")
    parser.add_argument("--game", required=True, choices=sorted(GAME_DISPLAY))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "eval_reports" / "ablations",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_REPO_ROOT / "docs" / "EVAL_REPORT.md",
    )
    args = parser.parse_args()

    dataset = args.dataset or (_DATASETS_DIR / f"{args.game}.jsonl")
    results = asyncio.run(
        run_matrix(
            game_slug=args.game,
            dataset_path=dataset,
            output_dir=args.out_dir,
            limit=args.limit,
        )
    )
    rows = [
        {
            "config_id": r.config_id,
            "metrics": r.metrics,
            "skipped_reason": r.skipped_reason,
        }
        for r in results
    ]
    md = render_markdown_report(game_slug=args.game, rows=rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(md)
    print(md)


if __name__ == "__main__":
    _main()
