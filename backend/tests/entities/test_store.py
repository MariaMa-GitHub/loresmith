from unittest.mock import AsyncMock, MagicMock

import pytest

from app.entities.schema import ExtractedEntity
from app.entities.store import upsert_entities


def _mock_session(existing_rows: list) -> AsyncMock:
    scalars = MagicMock()
    scalars.all.return_value = existing_rows
    result = MagicMock()
    result.scalars.return_value = scalars
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_upsert_inserts_new_entity():
    session = _mock_session(existing_rows=[])
    added = []
    session.add = lambda obj: added.append(obj)

    entities = [ExtractedEntity(slug="zagreus", name="Zagreus",
                                entity_type="character", description="Prince.")]
    count = await upsert_entities(session=session, game_slug="hades", entities=entities)

    assert count == 1
    assert len(added) == 1
    assert added[0].slug == "zagreus"
    session.execute.assert_awaited_once()  # single batched SELECT


@pytest.mark.asyncio
async def test_upsert_updates_existing_entity():
    existing = MagicMock()
    existing.slug = "zagreus"
    existing.name = "Old Name"
    existing.description = "Old"
    session = _mock_session(existing_rows=[existing])
    session.add = MagicMock()

    entities = [ExtractedEntity(slug="zagreus", name="Zagreus Updated",
                                entity_type="character", description="Fresh.")]
    count = await upsert_entities(session=session, game_slug="hades", entities=entities)

    assert count == 1
    assert existing.name == "Zagreus Updated"
    assert existing.description == "Fresh."
    session.add.assert_not_called()
    session.execute.assert_awaited_once()  # single batched SELECT


@pytest.mark.asyncio
async def test_upsert_batches_mixed_insert_and_update():
    existing = MagicMock()
    existing.slug = "nyx"
    existing.name = "Nyx"
    existing.description = "Old"
    session = _mock_session(existing_rows=[existing])
    added = []
    session.add = lambda obj: added.append(obj)

    entities = [
        ExtractedEntity(
            slug="nyx", name="Nyx Updated", entity_type="character", description="New."
        ),
        ExtractedEntity(
            slug="charon", name="Charon", entity_type="character", description="Ferryman."
        ),
    ]
    count = await upsert_entities(session=session, game_slug="hades", entities=entities)

    assert count == 2
    assert existing.name == "Nyx Updated"
    assert len(added) == 1
    assert added[0].slug == "charon"
    # Both entities resolved with a single SELECT, not two.
    session.execute.assert_awaited_once()
