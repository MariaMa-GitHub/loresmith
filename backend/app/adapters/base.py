from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class RobotsPolicy(StrEnum):
    RESPECT = "respect"  # IGNORE may be added if a permissive-license source is onboarded


@dataclass
class SourceConfig:
    base_url: str
    allowed_path_prefix: str = "/wiki/"
    license: str = "CC-BY-SA-3.0"
    crawl_delay: float = 1.0


@runtime_checkable
class GameAdapter(Protocol):
    """Contract every game integration must satisfy.

    Note: @runtime_checkable only verifies method presence (get_article_urls).
    Attribute fields (slug, sources, etc.) are enforced by type-checkers, not
    isinstance checks.
    """

    slug: str
    display_name: str
    sources: list[SourceConfig]
    robots_policy: RobotsPolicy
    license: str
    chunk_size: int       # target words per passage
    chunk_overlap: int    # overlap words between adjacent passages
    starter_prompts: list[str]

    def get_article_urls(self) -> list[str]:
        """Return the list of article URLs to ingest for this game."""
        ...
