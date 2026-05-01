import json
from unittest.mock import AsyncMock

import pytest

from app.entities.extractor import EntityExtractor


@pytest.mark.asyncio
async def test_extractor_parses_llm_json_and_applies_schema_filter():
    llm = AsyncMock()
    llm.complete.return_value = json.dumps([
        {
            "slug": "Zagreus", "name": "Zagreus",
            "entity_type": "character", "description": "Prince.",
        },
        {"slug": "Unknown", "name": "?", "entity_type": "mystery", "description": ""},
    ])

    extractor = EntityExtractor(llm=llm, allowed_types={"character", "weapon"})
    entities = await extractor.extract(
        page_text="Some wiki text …",
        source_url="https://example.com/Zagreus",
        game_slug="hades",
    )

    # Only the known-type entity survives.
    assert [e.slug for e in entities] == ["zagreus"]
    assert entities[0].entity_type == "character"


@pytest.mark.asyncio
async def test_extractor_returns_empty_on_malformed_json():
    llm = AsyncMock()
    llm.complete.return_value = "I'm just chatter"
    extractor = EntityExtractor(llm=llm, allowed_types={"character"})
    result = await extractor.extract(
        page_text="x", source_url="u", game_slug="hades"
    )
    assert result == []
