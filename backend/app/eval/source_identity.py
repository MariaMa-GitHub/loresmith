from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import unquote, urlparse

from app.games import ADAPTERS

_SPACE_RE = re.compile(r"\s+")
_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")

# Explicit wiki aliases/redirects that appear in eval gold annotations.
# These preserve a stable source identity without requiring live network
# resolution during tests or offline eval runs.
_GAME_SOURCE_ALIASES = {
    "hades": {
        "darkness (resource)": "darkness",
        "hades (character)": "hades",
        "obols": "charon's obol",
    },
}


@dataclass(frozen=True)
class ResolvedSourceIdentities:
    resolved_ids: list[str]
    unresolved_urls: list[str]


def canonical_source_id(url: str) -> str | None:
    """Return a normalized wiki-page identity for a source URL.

    The current corpus is wiki-backed, so the source identity is the page
    title normalized for comparison rather than the full URL string. This
    avoids false eval regressions from benign aliasing/redirect differences
    such as `Hades_(character)` vs `Hades`.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path:
        return None

    if "/wiki/" in path:
        raw_title = path.rsplit("/wiki/", maxsplit=1)[-1]
    else:
        raw_title = path.rsplit("/", maxsplit=1)[-1]
    if not raw_title:
        return None

    normalized = unquote(raw_title).replace("_", " ")
    normalized = _SPACE_RE.sub(" ", normalized).strip().casefold()
    return normalized or None


def _strip_parenthetical_suffix(title: str) -> str:
    return _PAREN_SUFFIX_RE.sub("", title).strip()


@lru_cache
def ingested_source_ids(game_slug: str) -> set[str]:
    adapter = ADAPTERS.get(game_slug)
    if adapter is None:
        return set()
    return {
        source_id
        for url in adapter.get_article_urls()
        if (source_id := canonical_source_id(url)) is not None
    }


@lru_cache
def _game_aliases(game_slug: str) -> dict[str, str]:
    aliases = _GAME_SOURCE_ALIASES.get(game_slug, {})
    normalized_aliases: dict[str, str] = {}
    for alias, target in aliases.items():
        alias_id = canonical_source_id(f"https://example.com/wiki/{alias.replace(' ', '_')}")
        target_id = canonical_source_id(f"https://example.com/wiki/{target.replace(' ', '_')}")
        if alias_id and target_id:
            normalized_aliases[alias_id] = target_id
    return normalized_aliases


def resolve_source_identity(game_slug: str, url: str) -> str | None:
    """Resolve a gold or retrieved source URL to an ingested source identity."""
    source_id = canonical_source_id(url)
    if source_id is None:
        return None

    known_ids = ingested_source_ids(game_slug)
    if source_id in known_ids:
        return source_id

    alias_target = _game_aliases(game_slug).get(source_id)
    if alias_target in known_ids:
        return alias_target

    base_id = _strip_parenthetical_suffix(source_id)
    if base_id != source_id and base_id in known_ids:
        return base_id

    return None


def resolve_source_identities(
    game_slug: str,
    urls: list[str],
) -> ResolvedSourceIdentities:
    resolved_ids: list[str] = []
    unresolved_urls: list[str] = []
    seen_ids: set[str] = set()

    for url in urls:
        resolved = resolve_source_identity(game_slug, url)
        if resolved is None:
            unresolved_urls.append(url)
            continue
        if resolved not in seen_ids:
            seen_ids.add(resolved)
            resolved_ids.append(resolved)

    return ResolvedSourceIdentities(
        resolved_ids=resolved_ids,
        unresolved_urls=unresolved_urls,
    )
