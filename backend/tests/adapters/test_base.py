from dataclasses import dataclass

from app.adapters.base import GameAdapter, RobotsPolicy, SourceConfig


def test_source_config_defaults():
    src = SourceConfig(base_url="https://example.com/wiki/")
    assert src.allowed_path_prefix == "/wiki/"
    assert src.license == "CC-BY-SA-3.0"
    assert src.crawl_delay == 1.0


def test_robots_policy_values():
    assert RobotsPolicy.RESPECT == "respect"


@dataclass
class FakeAdapter:
    slug: str = "test-game"
    display_name: str = "Test Game"
    sources: list = None
    robots_policy: RobotsPolicy = RobotsPolicy.RESPECT
    license: str = "CC-BY-SA-3.0"
    chunk_size: int = 400
    chunk_overlap: int = 50
    starter_prompts: list = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.starter_prompts is None:
            self.starter_prompts = []

    def get_article_urls(self) -> list[str]:
        return []


def test_fake_adapter_satisfies_protocol():
    adapter = FakeAdapter()
    assert isinstance(adapter, GameAdapter)


def test_game_adapter_requires_get_article_urls():
    """Protocol includes get_article_urls — verify it's part of the contract."""
    assert "get_article_urls" in dir(GameAdapter)
