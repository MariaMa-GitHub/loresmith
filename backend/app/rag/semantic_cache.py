from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SemanticCache as SemanticCacheRow
from app.services import CorpusRevision


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    passages: list[dict]
    citations: list[dict]
    similarity: float


def corpus_revision_key(revision: CorpusRevision) -> str:
    """Short, stable identifier derived from a CorpusRevision.

    Any re-ingest that inserts, deletes, or updates passages changes either
    passage_count, max_passage_id, or latest_updated_at, which flips the key
    and makes any cache rows under the old key unreachable.
    """
    updated = revision.latest_updated_at.isoformat() if revision.latest_updated_at else "none"
    return f"{revision.passage_count}:{revision.max_passage_id or 0}:{updated}"[:64]


class SemanticCache:
    """Embedding-similarity cache for generated answers.

    Scope: (game_slug, corpus_revision, max_spoiler_tier, embedding_backend,
    embedding_model). Lookup = nearest row within that exact scope by cosine
    distance; hit = similarity >= similarity_threshold.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        lookup_limit: int = 3,
    ) -> None:
        self._threshold = similarity_threshold
        self._lookup_limit = lookup_limit

    async def get(
        self,
        *,
        session: AsyncSession,
        game_slug: str,
        corpus_revision: str,
        max_spoiler_tier: int,
        embedding_backend: str,
        embedding_model: str,
        query_embedding: list[float],
    ) -> CachedAnswer | None:
        distance = SemanticCacheRow.query_embedding.cosine_distance(
            query_embedding
        ).label("distance")
        stmt = (
            select(SemanticCacheRow.id, SemanticCacheRow.response, distance)
            .where(SemanticCacheRow.game_slug == game_slug)
            .where(SemanticCacheRow.corpus_revision == corpus_revision)
            .where(SemanticCacheRow.max_spoiler_tier == max_spoiler_tier)
            .where(SemanticCacheRow.embedding_backend == embedding_backend)
            .where(SemanticCacheRow.embedding_model == embedding_model)
            .order_by(distance)
            .limit(self._lookup_limit)
        )
        result = await session.execute(stmt)
        row = result.first()
        if row is None:
            return None

        similarity = 1.0 - float(row.distance)
        if similarity < self._threshold:
            return None

        payload = row.response or {}
        return CachedAnswer(
            answer=payload.get("answer", ""),
            passages=payload.get("passages", []),
            citations=payload.get("citations", []),
            similarity=similarity,
        )

    async def put(
        self,
        *,
        session: AsyncSession,
        game_slug: str,
        corpus_revision: str,
        max_spoiler_tier: int,
        embedding_backend: str,
        embedding_model: str,
        query_text: str,
        query_embedding: list[float],
        answer: str,
        passages: list[dict],
        citations: list[dict],
    ) -> None:
        session.add(
            SemanticCacheRow(
                game_slug=game_slug,
                corpus_revision=corpus_revision,
                max_spoiler_tier=max_spoiler_tier,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                query_text=query_text,
                query_embedding=query_embedding,
                response={
                    "answer": answer,
                    "passages": passages,
                    "citations": citations,
                },
            )
        )
        await session.commit()
