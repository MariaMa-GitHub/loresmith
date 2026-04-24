from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Entity
from app.entities.schema import ExtractedEntity


async def upsert_entities(
    *,
    session: AsyncSession,
    game_slug: str,
    entities: list[ExtractedEntity],
) -> int:
    if not entities:
        return 0

    touched = 0
    for entity in entities:
        stmt = (
            select(Entity)
            .where(Entity.game_slug == game_slug)
            .where(Entity.slug == entity.slug)
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            existing.name = entity.name
            existing.description = entity.description
            existing.entity_type = entity.entity_type
            touched += 1
        else:
            session.add(Entity(
                game_slug=game_slug,
                slug=entity.slug,
                entity_type=entity.entity_type,
                name=entity.name,
                description=entity.description,
                spoiler_tier=0,
                metadata_={},
            ))
            touched += 1
    await session.commit()
    return touched


async def get_entity(
    session: AsyncSession, *, game_slug: str, slug: str,
) -> Entity | None:
    stmt = (
        select(Entity)
        .where(Entity.game_slug == game_slug)
        .where(Entity.slug == slug)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def list_entities_by_type(
    session: AsyncSession, *, game_slug: str, entity_type: str, limit: int = 50,
) -> list[Entity]:
    stmt = (
        select(Entity)
        .where(Entity.game_slug == game_slug)
        .where(Entity.entity_type == entity_type)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
