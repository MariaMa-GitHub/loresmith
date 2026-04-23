"""Shared application services and corpus helpers.

This module exists so that non-web callers (eval harness, ingestion CLI
helpers, tests) can reuse the same Services dataclass and corpus-revision
helpers as the FastAPI app without importing ``app.main`` (which would pull
in the web layer).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select

from app.config import Settings, get_settings
from app.db.models import Passage
from app.ingestion.pipeline import Embedder, make_embedder
from app.llm.router import LLMRouter, build_llm_router
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseRetriever
from app.retrieval.reranker import CrossEncoderReranker, NullReranker
from app.tracing.langfuse import LangfuseTracer


@dataclass
class Services:
    settings: Settings
    tracer: LangfuseTracer
    embedder: Embedder
    dense: DenseRetriever
    router: LLMRouter
    reranker: object


@dataclass(frozen=True)
class CorpusRevision:
    passage_count: int
    latest_updated_at: datetime | None
    max_passage_id: int | None


def build_services() -> Services:
    settings = get_settings()
    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model)
        if settings.reranker_enabled
        else NullReranker()
    )
    return Services(
        settings=settings,
        tracer=LangfuseTracer(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        ),
        embedder=make_embedder(settings),
        dense=DenseRetriever(),
        router=build_llm_router(settings),
        reranker=reranker,
    )


async def build_bm25(session, game_slug: str) -> tuple[BM25Index, dict[int, str]]:
    """Rebuild the in-memory BM25 index and source-URL map for a game.

    Returns a ``(BM25Index, {passage_id: source_url})`` tuple. The source map
    is threaded through rrf_fuse so BM25-only passages retain a valid
    citation URL.
    """
    result = await session.execute(
        select(
            Passage.id,
            Passage.content,
            Passage.source_url,
            Passage.spoiler_tier,
        ).where(
            Passage.game_slug == game_slug
        )
    )
    rows = result.all()
    index = BM25Index()
    source_map: dict[int, str] = {}
    if rows:
        index.build(
            [r.id for r in rows],
            [r.content for r in rows],
            spoiler_tiers=[r.spoiler_tier for r in rows],
        )
        source_map = {r.id: r.source_url for r in rows}
    return index, source_map


async def get_corpus_revision(session, game_slug: str) -> CorpusRevision:
    result = await session.execute(
        select(
            func.count(Passage.id),
            func.max(Passage.updated_at),
            func.max(Passage.id),
        ).where(Passage.game_slug == game_slug)
    )
    passage_count, latest_updated_at, max_passage_id = result.one()
    return CorpusRevision(
        passage_count=int(passage_count or 0),
        latest_updated_at=latest_updated_at,
        max_passage_id=max_passage_id,
    )
