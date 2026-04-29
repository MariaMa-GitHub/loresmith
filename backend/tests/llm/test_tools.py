from unittest.mock import MagicMock

import pytest

from app.llm.tools import (
    ToolCall,
    ToolDispatcher,
    build_default_tools,
    entity_lookup_schema,
    list_entities_by_type_schema,
)


def test_build_default_tools_returns_two_tools():
    from app.adapters.base import EntityTypeSchema

    tools = build_default_tools(
        game_slug="hades",
        entity_schema=[
            EntityTypeSchema(name="character", description=""),
            EntityTypeSchema(name="weapon", description=""),
        ],
    )
    names = {t.name for t in tools}
    assert names == {"entity_lookup", "list_entities_by_type"}


def test_tool_schemas_have_json_schema_parameters():
    assert "slug" in entity_lookup_schema()["parameters"]["properties"]
    listing = list_entities_by_type_schema(
        allowed_entity_types=["character", "weapon"],
    )
    assert "entity_type" in listing["parameters"]["properties"]
    assert listing["parameters"]["properties"]["entity_type"]["enum"] == [
        "character",
        "weapon",
    ]


@pytest.mark.asyncio
async def test_dispatcher_runs_entity_lookup(monkeypatch):
    from app.entities import store

    fake_entity = MagicMock()
    fake_entity.slug = "zagreus"
    fake_entity.name = "Zagreus"
    fake_entity.entity_type = "character"
    fake_entity.description = "Prince."

    async def fake_get(session, *, game_slug, slug):
        assert game_slug == "hades" and slug == "zagreus"
        return fake_entity

    monkeypatch.setattr(store, "get_entity", fake_get)

    dispatcher = ToolDispatcher(
        game_slug="hades",
        allowed_entity_types={"character", "weapon"},
    )
    result = await dispatcher.run(
        session=None,
        call=ToolCall(name="entity_lookup", arguments={"slug": "Zagreus"}),
    )
    assert result["slug"] == "zagreus"
    assert result["name"] == "Zagreus"
    assert result["entity_type"] == "character"


@pytest.mark.asyncio
async def test_dispatcher_rejects_unknown_tool_name():
    dispatcher = ToolDispatcher(
        game_slug="hades",
        allowed_entity_types={"character", "weapon"},
    )
    result = await dispatcher.run(
        session=None,
        call=ToolCall(name="rm_rf", arguments={}),
    )
    assert "error" in result
