from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.adapters.base import RobotsPolicy, SourceConfig
from app.ingestion.chunker import Chunker
from app.ingestion.pipeline import IngestResult, run_ingestion
from app.ingestion.scraper import ScrapedPage
from app.ingestion.spoiler_tagger import SpoilerTagger


@dataclass
class _FakeAdapter:
    slug: str = "fake"
    display_name: str = "Fake"
    sources: list = None
    robots_policy: RobotsPolicy = RobotsPolicy.RESPECT
    license: str = "CC-BY-SA-3.0"
    chunk_size: int = 400
    chunk_overlap: int = 50
    starter_prompts: list = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = [SourceConfig(base_url="https://fake.example.com")]
        if self.starter_prompts is None:
            self.starter_prompts = []

    def get_article_urls(self):
        return ["https://fake.example.com/wiki/Foo", "https://fake.example.com/wiki/Bar"]


@pytest.mark.asyncio
async def test_run_ingestion_returns_ingest_result():
    fake_page = ScrapedPage(
        url="https://fake.example.com/wiki/Foo",
        text="Foo is a character in the fake game.",
        title="Foo",
        fetched_at=datetime.now(UTC),
    )

    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=fake_page)

    fake_chunker = Chunker(chunk_size=5, overlap=1)

    fake_embedder = MagicMock()
    fake_embedder.embed = AsyncMock(return_value=[[0.0] * 768] * 10)

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()
    fake_session.commit = AsyncMock()

    adapter = _FakeAdapter()
    result = await run_ingestion(
        adapter=adapter,
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
    )

    assert isinstance(result, IngestResult)
    assert result.game_slug == "fake"
    assert result.pages_fetched >= 1
    assert result.chunks_created >= 1


@pytest.mark.asyncio
async def test_run_ingestion_skips_none_pages():
    """Scraper returns None for disallowed pages — pipeline skips them gracefully."""
    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=None)  # all pages disallowed

    fake_chunker = Chunker(chunk_size=400, overlap=50)
    fake_embedder = MagicMock()
    fake_embedder.embed = AsyncMock(return_value=[])

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()
    fake_session.commit = AsyncMock()

    adapter = _FakeAdapter()
    result = await run_ingestion(
        adapter=adapter,
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
    )

    assert result.pages_fetched == 0
    assert result.chunks_created == 0


@pytest.mark.asyncio
async def test_run_ingestion_dry_run_skips_db_writes():
    fake_page = ScrapedPage(
        url="https://fake.example.com/wiki/Foo",
        text="Foo is a character in the fake game.",
        title="Foo",
        fetched_at=datetime.now(UTC),
    )

    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=fake_page)

    fake_chunker = Chunker(chunk_size=5, overlap=1)

    fake_embedder = MagicMock()
    fake_embedder.embed = AsyncMock(return_value=[[0.0] * 768] * 10)

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()
    fake_session.commit = AsyncMock()

    adapter = _FakeAdapter()
    result = await run_ingestion(
        adapter=adapter,
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
        dry_run=True,
    )

    fake_session.execute.assert_not_called()
    fake_session.commit.assert_not_called()
    assert result.passages_upserted == 0
    assert result.chunks_created >= 1


@pytest.mark.asyncio
async def test_pipeline_uses_spoiler_tagger():
    fake_page = ScrapedPage(
        url="https://fake.example.com/wiki/Foo",
        text="Foo is a character. " * 5,
        title="Foo",
        fetched_at=datetime.now(UTC),
    )

    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=fake_page)
    fake_chunker = Chunker(chunk_size=5, overlap=1)
    fake_embedder = MagicMock()
    fake_embedder.embed = AsyncMock(return_value=[[0.0] * 768] * 10)

    fake_tagger = MagicMock(spec=SpoilerTagger)
    fake_tagger.tag_async = AsyncMock(return_value=0)

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()
    fake_session.commit = AsyncMock()

    adapter = _FakeAdapter()
    await run_ingestion(
        adapter=adapter,
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
        spoiler_tagger=fake_tagger,
    )

    assert fake_tagger.tag_async.call_count >= 1
