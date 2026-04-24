from unittest.mock import AsyncMock, MagicMock

import pytest

from app.entities.schema import ExtractedEntity
from app.entities.store import upsert_entities


@pytest.mark.asyncio
async def test_upsert_inserts_new_entity(monkeypatch):
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)
    added = []
    session.add = lambda obj: added.append(obj)
    session.commit = AsyncMock()

    entities = [ExtractedEntity(slug="zagreus", name="Zagreus",
                                entity_type="character", description="Prince.")]
    await upsert_entities(session=session, game_slug="hades", entities=entities)

    assert len(added) == 1
    assert added[0].slug == "zagreus"


@pytest.mark.asyncio
async def test_upsert_updates_existing_entity():
    session = AsyncMock()
    existing = MagicMock()
    existing.name = "Old Name"; existing.description = "Old"
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock()
    session.commit = AsyncMock()

    entities = [ExtractedEntity(slug="zagreus", name="Zagreus Updated",
                                entity_type="character", description="Fresh.")]
    await upsert_entities(session=session, game_slug="hades", entities=entities)

    assert existing.name == "Zagreus Updated"
    assert existing.description == "Fresh."
    session.add.assert_not_called()
