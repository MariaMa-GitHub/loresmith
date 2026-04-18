from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from app.ingestion.review_spoilers import (
    _parse_override,
    _preview_text,
    apply_overrides,
    list_passages_for_review,
)


def test_preview_text_compacts_whitespace():
    preview = _preview_text("Nyx   is\n\n  the Goddess   of Night.")
    assert preview == "Nyx is the Goddess of Night."


def test_parse_override_accepts_valid_pair():
    assert _parse_override("123=2") == (123, 2)


def test_parse_override_rejects_invalid_tier():
    with pytest.raises(Exception):
        _parse_override("123=9")


@pytest.mark.asyncio
async def test_list_passages_for_review_builds_entries():
    row = MagicMock()
    row.id = 7
    row.game_slug = "hades"
    row.source_url = "https://hades.fandom.com/wiki/Nyx"
    row.spoiler_tier = 2
    row.content = "Nyx reveals the truth about Zagreus."

    mock_result = MagicMock()
    mock_result.all.return_value = [row]

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    entries = await list_passages_for_review(
        session=session,
        game_slug="hades",
        min_tier=1,
        limit=10,
    )

    assert len(entries) == 1
    assert entries[0].passage_id == 7
    assert entries[0].spoiler_tier == 2
    assert "Zagreus" in entries[0].preview


@pytest.mark.asyncio
async def test_apply_overrides_updates_known_passages():
    row1 = MagicMock()
    row1.id = 7
    row2 = MagicMock()
    row2.id = 8

    mock_result = MagicMock()
    mock_result.all.return_value = [row1, row2]

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.commit = AsyncMock()

    updated = await apply_overrides(
        session=session,
        game_slug="hades",
        overrides={7: 1, 8: 3},
    )

    assert updated == 2
    assert session.execute.await_count == 3
    session.commit.assert_awaited_once()

    update_sql = [
        str(call.args[0].compile(dialect=postgresql.dialect()))
        for call in session.execute.await_args_list[1:]
    ]
    assert all("spoiler_tier" in sql for sql in update_sql)
    assert all("updated_at" in sql for sql in update_sql)


@pytest.mark.asyncio
async def test_apply_overrides_rejects_missing_passage_ids():
    row = MagicMock()
    row.id = 7

    mock_result = MagicMock()
    mock_result.all.return_value = [row]

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(ValueError, match="8"):
        await apply_overrides(
            session=session,
            game_slug="hades",
            overrides={7: 1, 8: 3},
        )
