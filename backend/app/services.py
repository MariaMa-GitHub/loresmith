"""Shared application services and corpus helpers.

This module exists so that non-web callers (eval harness, ingestion CLI
helpers, tests) can reuse the same Services dataclass and corpus-revision
helpers as the FastAPI app without importing ``app.main`` (which would pull
in the web layer).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from app.config import Settings, get_settings
from app.db.models import Passage
from app.ingestion.pipeline import Embedder, make_embedder
from app.llm.base import TaskType
from app.llm.router import LLMRouter, build_llm_router
from app.llm.tools import ToolDispatcher
from app.rag.verifier import Verifier
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseRetriever
from app.retrieval.reranker import CrossEncoderReranker, NullReranker, Reranker
from app.tracing.langfuse import LangfuseTracer

if TYPE_CHECKING:
    from app.rag.semantic_cache import SemanticCache


@dataclass
class Services:
    settings: Settings
    tracer: LangfuseTracer
    embedder: Embedder
    dense: DenseRetriever
    router: LLMRouter
    reranker: Reranker
    semantic_cache: SemanticCache | None = None
    verifier: Verifier | None = None
    tool_dispatcher_factory: Callable[[str, set[str]], ToolDispatcher] | None = None


@dataclass(frozen=True)
class CorpusRevision:
    passage_count: int
    latest_updated_at: datetime | None
    max_passage_id: int | None


def build_services() -> Services:
    from app.rag.semantic_cache import (  # noqa: F401 — lazy to avoid circular
        SemanticCache,
        corpus_revision_key,
    )

    settings = get_settings()
    tracer = LangfuseTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    router = build_llm_router(settings)
    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model)
        if settings.reranker_enabled
        else NullReranker()
    )
    cache = (
        SemanticCache(
            similarity_threshold=settings.semantic_cache_threshold,
            lookup_limit=settings.semantic_cache_lookup_limit,
        )
        if settings.semantic_cache_enabled
        else None
    )
    verifier = (
        Verifier(llm=router.for_task(TaskType.VERIFY), tracer=tracer)
        if settings.verifier_enabled
        else None
    )
    tool_dispatcher_factory = (
        (lambda slug, allowed_types: ToolDispatcher(
            game_slug=slug,
            allowed_entity_types=allowed_types,
        ))
        if settings.tools_enabled else None
    )
    return Services(
        settings=settings,
        tracer=tracer,
        embedder=make_embedder(settings),
        dense=DenseRetriever(),
        router=router,
        reranker=reranker,
        semantic_cache=cache,
        verifier=verifier,
        tool_dispatcher_factory=tool_dispatcher_factory,
    )


async def resolve_corpus_revision_key(session, game_slug: str) -> str:
    from app.rag.semantic_cache import corpus_revision_key
    revision = await get_corpus_revision(session, game_slug)
    return corpus_revision_key(revision)


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
