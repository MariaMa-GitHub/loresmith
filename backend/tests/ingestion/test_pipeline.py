from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from app.adapters.base import DEFAULT_SPOILER_PROFILE, RobotsPolicy, SourceConfig
from app.ingestion.chunker import Chunk, Chunker
from app.ingestion.pipeline import IngestResult, run_ingestion
from app.ingestion.scraper import ScrapedPage
from app.ingestion.spoiler_tagger import SpoilerTagger
from app.entities.schema import ExtractedEntity


@dataclass
class _FakeAdapter:
    slug: str = "fake"
    display_name: str = "Fake"
    sources: list = None
    robots_policy: RobotsPolicy = RobotsPolicy.RESPECT
    license: str = "CC-BY-SA-3.0"
    chunker: Chunker = None
    starter_prompts: list = None
    spoiler_profile: object = None
    entity_schema: list = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = [SourceConfig(base_url="https://fake.example.com")]
        if self.chunker is None:
            self.chunker = Chunker(chunk_size=400, overlap=50)
        if self.starter_prompts is None:
            self.starter_prompts = []
        if self.spoiler_profile is None:
            self.spoiler_profile = DEFAULT_SPOILER_PROFILE
        if self.entity_schema is None:
            self.entity_schema = []

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
    fake_embedder.embed = AsyncMock(side_effect=lambda texts: [[0.0] * 768 for _ in texts])

    select_result = MagicMock()
    select_result.all.return_value = []
    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(return_value=select_result)
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

    select_result = MagicMock()
    select_result.all.return_value = []
    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(return_value=select_result)
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
    fake_embedder.embed = AsyncMock(side_effect=lambda texts: [[0.0] * 768 for _ in texts])

    select_result = MagicMock()
    select_result.all.return_value = []
    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(return_value=select_result)
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

    assert fake_session.execute.call_count == 1
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
    fake_embedder.embed = AsyncMock(side_effect=lambda texts: [[0.0] * 768 for _ in texts])

    fake_tagger = MagicMock(spec=SpoilerTagger)
    fake_tagger.tag_async = AsyncMock(return_value=0)

    select_result = MagicMock()
    select_result.all.return_value = []
    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(return_value=select_result)
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


@pytest.mark.asyncio
async def test_run_ingestion_skips_unchanged_chunks_before_embedding():
    existing_chunk = Chunk(
        content="Foo is a character in the fake game.",
        source_url="https://fake.example.com/wiki/Foo",
        title="Foo",
    )
    fake_page = ScrapedPage(
        url=existing_chunk.source_url,
        text=existing_chunk.content,
        title="Foo",
        fetched_at=datetime.now(UTC),
    )

    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=fake_page)

    fake_chunker = MagicMock()
    fake_chunker.chunk.return_value = [existing_chunk]

    fake_embedder = MagicMock()
    fake_embedder.backend_name = "local"
    fake_embedder.model_name = "fake-local-v1"
    fake_embedder.embed = AsyncMock(return_value=[])

    existing_row = MagicMock()
    existing_row.id = 1
    existing_row.source_url = existing_chunk.source_url
    existing_row.content_hash = existing_chunk.content_hash
    existing_row.spoiler_tier = 0
    existing_row.embedding_backend = "local"
    existing_row.embedding_model = "fake-local-v1"
    existing_row.has_embedding = True

    select_result = MagicMock()
    select_result.all.return_value = [existing_row]

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(return_value=select_result)
    fake_session.commit = AsyncMock()

    class _SingleUrlAdapter(_FakeAdapter):
        def get_article_urls(self):
            return [existing_chunk.source_url]

    result = await run_ingestion(
        adapter=_SingleUrlAdapter(),
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
    )

    fake_embedder.embed.assert_not_called()
    fake_session.commit.assert_not_called()
    assert result.chunks_created == 1
    assert result.passages_upserted == 0
    assert result.passages_skipped == 1


@pytest.mark.asyncio
async def test_run_ingestion_prunes_removed_sources_from_manifest():
    current_chunk = Chunk(
        content="Current content for Foo",
        source_url="https://fake.example.com/wiki/Foo",
        title="Foo",
    )
    fake_page = ScrapedPage(
        url=current_chunk.source_url,
        text=current_chunk.content,
        title="Foo",
        fetched_at=datetime.now(UTC),
    )

    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=fake_page)
    fake_chunker = MagicMock()
    fake_chunker.chunk.return_value = [current_chunk]

    fake_embedder = MagicMock()
    fake_embedder.backend_name = "local"
    fake_embedder.model_name = "fake-local-v1"
    fake_embedder.embed = AsyncMock(return_value=[[0.0] * 768])

    current_row = MagicMock()
    current_row.id = 1
    current_row.source_url = current_chunk.source_url
    current_row.content_hash = current_chunk.content_hash
    current_row.spoiler_tier = 0
    current_row.embedding_backend = "local"
    current_row.embedding_model = "fake-local-v1"
    current_row.has_embedding = True
    removed_row = MagicMock()
    removed_row.id = 2
    removed_row.source_url = "https://fake.example.com/wiki/Removed"
    removed_row.content_hash = "removed-hash"
    removed_row.spoiler_tier = 0
    removed_row.embedding_backend = "local"
    removed_row.embedding_model = "fake-local-v1"
    removed_row.has_embedding = True

    select_result = MagicMock()
    select_result.all.return_value = [current_row, removed_row]
    executed = []

    async def _execute(stmt, *args, **kwargs):
        executed.append(stmt)
        if getattr(stmt, "is_select", False):
            return select_result
        return MagicMock()

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(side_effect=_execute)
    fake_session.commit = AsyncMock()

    class _SingleUrlAdapter(_FakeAdapter):
        def get_article_urls(self):
            return [current_chunk.source_url]

    await run_ingestion(
        adapter=_SingleUrlAdapter(),
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
    )

    delete_stmt = next(stmt for stmt in executed if stmt.__visit_name__ == "delete")
    compiled = delete_stmt.compile(dialect=postgresql.dialect())
    assert compiled.params["id_1"] == [2]


@pytest.mark.asyncio
async def test_run_ingestion_does_not_prune_failed_source_fetch():
    failed_url = "https://fake.example.com/wiki/Foo"

    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=None)
    fake_chunker = Chunker(chunk_size=5, overlap=1)
    fake_embedder = MagicMock()
    fake_embedder.backend_name = "local"
    fake_embedder.model_name = "fake-local-v1"
    fake_embedder.embed = AsyncMock(return_value=[])

    existing_row = MagicMock()
    existing_row.id = 99
    existing_row.source_url = failed_url
    existing_row.content_hash = "hash-99"
    existing_row.spoiler_tier = 2
    existing_row.embedding_backend = "local"
    existing_row.embedding_model = "fake-local-v1"
    existing_row.has_embedding = True

    select_result = MagicMock()
    select_result.all.return_value = [existing_row]
    executed = []

    async def _execute(stmt, *args, **kwargs):
        executed.append(stmt)
        if getattr(stmt, "is_select", False):
            return select_result
        return MagicMock()

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(side_effect=_execute)
    fake_session.commit = AsyncMock()

    class _SingleUrlAdapter(_FakeAdapter):
        def get_article_urls(self):
            return [failed_url]

    result = await run_ingestion(
        adapter=_SingleUrlAdapter(),
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
    )

    assert result.pages_fetched == 0
    assert all(stmt.__visit_name__ != "delete" for stmt in executed)
    fake_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_run_ingestion_retags_unchanged_chunks_when_spoiler_policy_changes():
    existing_chunk = Chunk(
        content="Foo is a hidden ending reveal.",
        source_url="https://fake.example.com/wiki/Foo",
        title="Foo",
    )
    fake_page = ScrapedPage(
        url=existing_chunk.source_url,
        text=existing_chunk.content,
        title="Foo",
        fetched_at=datetime.now(UTC),
    )

    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=fake_page)
    fake_chunker = MagicMock()
    fake_chunker.chunk.return_value = [existing_chunk]

    fake_embedder = MagicMock()
    fake_embedder.backend_name = "local"
    fake_embedder.model_name = "fake-local-v1"
    fake_embedder.embed = AsyncMock(return_value=[])

    fake_tagger = MagicMock(spec=SpoilerTagger)
    fake_tagger.tag_async = AsyncMock(return_value=3)

    existing_row = MagicMock()
    existing_row.id = 7
    existing_row.source_url = existing_chunk.source_url
    existing_row.content_hash = existing_chunk.content_hash
    existing_row.spoiler_tier = 0
    existing_row.embedding_backend = "local"
    existing_row.embedding_model = "fake-local-v1"
    existing_row.has_embedding = True

    select_result = MagicMock()
    select_result.all.return_value = [existing_row]
    executed = []

    async def _execute(stmt, *args, **kwargs):
        executed.append(stmt)
        if getattr(stmt, "is_select", False):
            return select_result
        return MagicMock()

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(side_effect=_execute)
    fake_session.commit = AsyncMock()

    class _SingleUrlAdapter(_FakeAdapter):
        def get_article_urls(self):
            return [existing_chunk.source_url]

    result = await run_ingestion(
        adapter=_SingleUrlAdapter(),
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
        spoiler_tagger=fake_tagger,
    )

    fake_embedder.embed.assert_not_called()
    assert fake_tagger.tag_async.call_count == 1
    assert any(stmt.__visit_name__ == "update" for stmt in executed)
    fake_session.commit.assert_called_once()
    assert result.passages_upserted == 1
    assert result.passages_skipped == 0


@pytest.mark.asyncio
async def test_run_ingestion_reembeds_unchanged_chunks_when_embedding_identity_changes():
    existing_chunk = Chunk(
        content="Foo is a character in the fake game.",
        source_url="https://fake.example.com/wiki/Foo",
        title="Foo",
    )
    fake_page = ScrapedPage(
        url=existing_chunk.source_url,
        text=existing_chunk.content,
        title="Foo",
        fetched_at=datetime.now(UTC),
    )

    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=fake_page)
    fake_chunker = MagicMock()
    fake_chunker.chunk.return_value = [existing_chunk]

    fake_embedder = MagicMock()
    fake_embedder.backend_name = "gemini"
    fake_embedder.model_name = "gemini-embedding-001"
    fake_embedder.embed = AsyncMock(return_value=[[0.0] * 768])

    existing_row = MagicMock()
    existing_row.id = 8
    existing_row.source_url = existing_chunk.source_url
    existing_row.content_hash = existing_chunk.content_hash
    existing_row.spoiler_tier = 0
    existing_row.embedding_backend = "local"
    existing_row.embedding_model = "fake-local-v1"
    existing_row.has_embedding = True

    select_result = MagicMock()
    select_result.all.return_value = [existing_row]
    executed = []

    async def _execute(stmt, *args, **kwargs):
        executed.append(stmt)
        if getattr(stmt, "is_select", False):
            return select_result
        return MagicMock()

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(side_effect=_execute)
    fake_session.commit = AsyncMock()

    class _SingleUrlAdapter(_FakeAdapter):
        def get_article_urls(self):
            return [existing_chunk.source_url]

    result = await run_ingestion(
        adapter=_SingleUrlAdapter(),
        scraper=fake_scraper,
        chunker=fake_chunker,
        embedder=fake_embedder,
        session=fake_session,
    )

    fake_embedder.embed.assert_awaited_once_with([existing_chunk.content])
    assert any(stmt.__visit_name__ == "update" for stmt in executed)
    fake_session.commit.assert_called_once()
    assert result.passages_upserted == 1
    assert result.passages_skipped == 0


@pytest.mark.asyncio
async def test_run_ingestion_upserts_entities_when_extractor_provided(monkeypatch):
    upserts: list = []

    async def fake_upsert(*, session, game_slug, entities):
        upserts.extend(entities)
        return len(entities)

    monkeypatch.setattr(
        "app.ingestion.pipeline.upsert_entities", fake_upsert,
    )

    class _Extractor:
        async def extract(self, *, page_text, source_url, game_slug):
            return [ExtractedEntity(slug="zag", name="Zagreus",
                                    entity_type="character", description="")]

    fake_page = ScrapedPage(
        url="https://fake.example.com/wiki/Foo",
        text="Foo is a character.",
        title="Foo",
        fetched_at=datetime.now(UTC),
    )
    fake_scraper = MagicMock()
    fake_scraper.fetch = AsyncMock(return_value=fake_page)

    fake_embedder = MagicMock()
    fake_embedder.embed = AsyncMock(side_effect=lambda texts: [[0.0] * 768 for _ in texts])
    fake_embedder.backend_name = "local"
    fake_embedder.model_name = "bge-base"

    select_result = MagicMock()
    select_result.all.return_value = []
    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(return_value=select_result)
    fake_session.commit = AsyncMock()

    adapter = _FakeAdapter(entity_schema=[])  # extractor is passed explicitly

    result = await run_ingestion(
        adapter=adapter,
        scraper=fake_scraper,
        chunker=Chunker(chunk_size=5, overlap=1),
        embedder=fake_embedder,
        session=fake_session,
        entity_extractor=_Extractor(),
    )
    assert len(upserts) >= 1
