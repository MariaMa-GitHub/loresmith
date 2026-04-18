from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.ingestion.chunker import Chunker


class RobotsPolicy(StrEnum):
    RESPECT = "respect"  # IGNORE may be added if a permissive-license source is onboarded


@dataclass
class SourceConfig:
    base_url: str
    allowed_path_prefix: str = "/wiki/"
    license: str = "CC-BY-SA-3.0"
    crawl_delay: float = 1.0


@dataclass(frozen=True)
class SpoilerProfile:
    """Per-game rules for the spoiler tagger.

    - tier_keywords: tuples of (tier, keywords) checked highest-tier-first.
    - ambiguous_keywords: terms that trigger LLM classification when no
      heuristic keyword fires.
    - system_prompt: system prompt for the LLM classifier.
    - fallback_ambiguous_tier: tier assigned when ambiguous text cannot be
      LLM-classified (no LLM available, or the call failed).
    """

    tier_keywords: tuple[tuple[int, tuple[str, ...]], ...]
    ambiguous_keywords: tuple[str, ...]
    system_prompt: str
    fallback_ambiguous_tier: int = 2


@dataclass(frozen=True)
class EntityTypeSchema:
    """A single entity type an adapter promises to produce in its corpus."""

    name: str
    description: str


DEFAULT_SPOILER_PROFILE = SpoilerProfile(
    tier_keywords=(),
    ambiguous_keywords=(),
    system_prompt=(
        "Assign a spoiler tier 0-3 to the passage. "
        "Reply with a single digit: 0, 1, 2, or 3. Nothing else."
    ),
    fallback_ambiguous_tier=2,
)


@runtime_checkable
class GameAdapter(Protocol):
    """Contract every game integration must satisfy.

    Note: @runtime_checkable verifies that named members exist at runtime,
    but it does not validate their precise types or value semantics.
    """

    slug: str
    display_name: str
    sources: list[SourceConfig]
    robots_policy: RobotsPolicy
    license: str
    chunker: Chunker
    starter_prompts: list[str]
    spoiler_profile: SpoilerProfile
    entity_schema: list[EntityTypeSchema]

    def get_article_urls(self) -> list[str]:
        """Return the list of article URLs to ingest for this game."""
        ...
