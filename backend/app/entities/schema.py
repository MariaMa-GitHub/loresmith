from __future__ import annotations

import re
from dataclasses import dataclass

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class SchemaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractedEntity:
    slug: str
    name: str
    entity_type: str
    description: str


def normalize_slug(text: str) -> str:
    lowered = text.strip().lower()
    slugified = _SLUG_RE.sub("-", lowered).strip("-")
    return slugified


def validate_entity(entity: ExtractedEntity, *, allowed_types: set[str]) -> None:
    if entity.entity_type not in allowed_types:
        raise SchemaValidationError(
            f"Entity type {entity.entity_type!r} is not in allowed set {sorted(allowed_types)!r}"
        )
    if not entity.slug or not entity.name:
        raise SchemaValidationError("Entity requires both slug and name")
