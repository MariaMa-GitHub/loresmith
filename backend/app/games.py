"""Central registry of game adapters.

Adapter classes are listed once here; the derived views (``ADAPTERS``,
``GAMES``, ``GAME_SLUGS``, ``GAME_DISPLAY``) are what the rest of the app
reads. Adding a new game means implementing the adapter and appending its
class to ``_ADAPTER_CLASSES`` — no other registration step is needed.
"""
from __future__ import annotations

from app.adapters.base import GameAdapter
from app.adapters.hades import HadesAdapter
from app.adapters.hades2 import HadesIIAdapter

_ADAPTER_CLASSES: list[type] = [HadesAdapter, HadesIIAdapter]

ADAPTERS: dict[str, GameAdapter] = {cls.slug: cls() for cls in _ADAPTER_CLASSES}
GAMES: list[dict[str, str]] = [
    {"slug": adapter.slug, "display_name": adapter.display_name}
    for adapter in ADAPTERS.values()
]
GAME_SLUGS: set[str] = set(ADAPTERS.keys())
GAME_DISPLAY: dict[str, str] = {
    slug: adapter.display_name for slug, adapter in ADAPTERS.items()
}
