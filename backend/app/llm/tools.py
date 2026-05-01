from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.adapters.base import EntityTypeSchema
from app.entities import store as entity_store
from app.entities.schema import normalize_slug


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def entity_lookup_schema() -> dict:
    return {
        "name": "entity_lookup",
        "description": "Look up a single entity by slug in the current game.",
        "parameters": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "Entity slug (lowercase, hyphenated)."}
            },
            "required": ["slug"],
        },
    }


def list_entities_by_type_schema(*, allowed_entity_types: Sequence[str]) -> dict:
    allowed = sorted({value.strip().lower() for value in allowed_entity_types if value.strip()})
    return {
        "name": "list_entities_by_type",
        "description": "List up to 50 entities of a given type for the current game.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_type": {"type": "string", "enum": allowed},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["entity_type"],
        },
    }


def build_default_tools(
    *,
    game_slug: str,
    entity_schema: Sequence[EntityTypeSchema],
) -> list[ToolDefinition]:
    allowed_entity_types = [entry.name for entry in entity_schema]
    if not allowed_entity_types:
        return []
    lookup = entity_lookup_schema()
    listing = list_entities_by_type_schema(allowed_entity_types=allowed_entity_types)
    return [
        ToolDefinition(
            name=lookup["name"], description=lookup["description"], parameters=lookup["parameters"]
        ),
        ToolDefinition(
            name=listing["name"],
            description=listing["description"],
            parameters=listing["parameters"],
        ),
    ]


class ToolDispatcher:
    def __init__(self, *, game_slug: str, allowed_entity_types: set[str]) -> None:
        self._game_slug = game_slug
        self._allowed_entity_types = {
            value.strip().lower() for value in allowed_entity_types if value.strip()
        }

    async def run(self, *, session, call: ToolCall) -> dict:
        if call.name == "entity_lookup":
            slug = normalize_slug(str(call.arguments.get("slug", "")))
            if not slug:
                return {"error": "slug required"}
            entity = await entity_store.get_entity(session, game_slug=self._game_slug, slug=slug)
            if entity is None:
                return {"error": f"no entity with slug {slug!r}"}
            return {
                "slug": entity.slug,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description or "",
            }

        if call.name == "list_entities_by_type":
            entity_type = str(call.arguments.get("entity_type", "")).strip().lower()
            limit = int(call.arguments.get("limit", 50))
            if not entity_type:
                return {"error": "entity_type required"}
            if entity_type not in self._allowed_entity_types:
                return {
                    "error": (
                        "entity_type must be one of "
                        f"{sorted(self._allowed_entity_types)}"
                    )
                }
            entities = await entity_store.list_entities_by_type(
                session, game_slug=self._game_slug, entity_type=entity_type, limit=limit,
            )
            return {
                "entity_type": entity_type,
                "entities": [
                    {"slug": e.slug, "name": e.name, "description": e.description or ""}
                    for e in entities
                ],
            }

        return {"error": f"unknown tool {call.name!r}"}
