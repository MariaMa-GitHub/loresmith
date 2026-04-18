from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Passage


@dataclass
class DenseHit:
    passage_id: int
    score: float        # cosine similarity: 1 - distance
    content: str
    source_url: str


class DenseRetriever:
    """Retrieves passages using pgvector cosine-similarity ANN search."""

    async def search(
        self,
        session: AsyncSession,
        game_slug: str,
        query_embedding: list[float],
        top_k: int = 10,
        max_spoiler_tier: int = 0,
        embedding_backend: str | None = None,
        embedding_model: str | None = None,
    ) -> list[DenseHit]:
        distance = Passage.embedding.cosine_distance(query_embedding).label("distance")

        stmt = (
            select(
                Passage.id,
                Passage.content,
                Passage.source_url,
                distance,
            )
            .where(Passage.game_slug == game_slug)
            .where(Passage.spoiler_tier <= max_spoiler_tier)
            .where(Passage.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )
        if embedding_backend is not None:
            stmt = stmt.where(Passage.embedding_backend == embedding_backend)
        if embedding_model is not None:
            stmt = stmt.where(Passage.embedding_model == embedding_model)

        result = await session.execute(stmt)
        rows = result.all()

        return [
            DenseHit(
                passage_id=row.id,
                score=1.0 - float(row.distance),
                content=row.content,
                source_url=row.source_url,
            )
            for row in rows
        ]
