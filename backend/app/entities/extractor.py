from __future__ import annotations

import json
import logging

from app.entities.schema import (
    ExtractedEntity,
    SchemaValidationError,
    normalize_slug,
    validate_entity,
)

logger = logging.getLogger(__name__)

_SYSTEM = """Extract named entities from the wiki text.
Return ONLY a JSON array. Each item has exactly these keys:
- slug: a short human-readable identifier
- name: the canonical display name
- entity_type: one of the allowed types listed in the user message
- description: a single-sentence summary

Skip anything that isn't clearly one of the allowed entity types. Do not
include items with empty slug or name. Do not invent entities not present
in the passage."""


class EntityExtractor:
    def __init__(
        self,
        llm,
        allowed_types: set[str],
    ) -> None:
        self._llm = llm
        self._allowed_types = set(allowed_types)

    async def extract(
        self,
        *,
        page_text: str,
        source_url: str,
        game_slug: str,
    ) -> list[ExtractedEntity]:
        if not page_text.strip() or not self._allowed_types:
            return []

        prompt = (
            f"Allowed entity types: {sorted(self._allowed_types)}\n\n"
            f"Source URL: {source_url}\n"
            f"Game: {game_slug}\n\n"
            f"Wiki text:\n{page_text[:6000]}\n"
        )

        try:
            raw = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM,
            )
        except Exception as exc:
            logger.warning("Entity extractor LLM call failed on %s: %s", source_url, exc)
            return []

        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = "\n".join(
                line for line in stripped.splitlines() if not line.startswith("```")
            ).strip()

        try:
            items = json.loads(stripped)
        except json.JSONDecodeError:
            logger.debug("Entity extractor returned non-JSON output: %r", raw[:200])
            return []
        if not isinstance(items, list):
            return []

        out: list[ExtractedEntity] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entity = ExtractedEntity(
                slug=normalize_slug(str(item.get("slug") or item.get("name") or "")),
                name=str(item.get("name") or "").strip(),
                entity_type=str(item.get("entity_type") or "").strip().lower(),
                description=str(item.get("description") or "").strip(),
            )
            try:
                validate_entity(entity, allowed_types=self._allowed_types)
            except SchemaValidationError:
                continue
            out.append(entity)
        return out
