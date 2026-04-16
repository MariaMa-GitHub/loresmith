from app.adapters.base import GameAdapter, RobotsPolicy
from app.adapters.hades import HadesAdapter


def test_hades_adapter_satisfies_protocol():
    adapter = HadesAdapter()
    assert isinstance(adapter, GameAdapter)


def test_hades_slug():
    assert HadesAdapter().slug == "hades"


def test_hades_robots_policy():
    assert HadesAdapter().robots_policy == RobotsPolicy.RESPECT


def test_hades_has_sources():
    adapter = HadesAdapter()
    assert len(adapter.sources) >= 1
    assert "fandom.com" in adapter.sources[0].base_url


def test_hades_get_article_urls_returns_list():
    urls = HadesAdapter().get_article_urls()
    assert isinstance(urls, list)
    assert len(urls) >= 10
    assert all(url.startswith("https://") for url in urls)


def test_hades_has_starter_prompts():
    prompts = HadesAdapter().starter_prompts
    assert len(prompts) >= 3


def test_hades_chunk_size_is_reasonable():
    adapter = HadesAdapter()
    assert 200 <= adapter.chunk_size <= 800
    assert adapter.chunk_overlap < adapter.chunk_size
