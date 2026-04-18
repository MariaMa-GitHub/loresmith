from app.adapters.base import GameAdapter, RobotsPolicy
from app.adapters.hades import HadesAdapter
from app.adapters.hades2 import HadesIIAdapter


def test_hades2_adapter_satisfies_protocol():
    assert isinstance(HadesIIAdapter(), GameAdapter)


def test_hades2_slug():
    assert HadesIIAdapter().slug == "hades2"


def test_hades2_display_name():
    assert "II" in HadesIIAdapter().display_name


def test_hades2_robots_policy():
    assert HadesIIAdapter().robots_policy == RobotsPolicy.RESPECT


def test_hades2_uses_different_base_url_than_hades():
    hades_bases = {s.base_url for s in HadesAdapter().sources}
    hades2_bases = {s.base_url for s in HadesIIAdapter().sources}
    assert hades_bases.isdisjoint(hades2_bases), (
        "Hades II adapter must ingest from a different wiki than Hades"
    )


def test_hades2_has_starter_prompts():
    assert len(HadesIIAdapter().starter_prompts) >= 3


def test_hades2_get_article_urls():
    urls = HadesIIAdapter().get_article_urls()
    assert len(urls) >= 10
    assert all("hades2.fandom.com" in url for url in urls)


def test_hades2_uses_reasonable_chunker_settings():
    adapter = HadesIIAdapter()
    assert 200 <= adapter.chunker.chunk_size <= 800
    assert adapter.chunker.overlap < adapter.chunker.chunk_size
