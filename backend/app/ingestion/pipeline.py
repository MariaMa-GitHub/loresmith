"""Ingestion pipeline CLI.

Usage:
    python -m app.ingestion.pipeline --game hades
    python -m app.ingestion.pipeline --game hades --dry-run
"""
import argparse
import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import GameAdapter
from app.config import Settings
from app.db.models import Passage
from app.entities.extractor import EntityExtractor
from app.entities.schema import ExtractedEntity
from app.entities.store import upsert_entities
from app.games import ADAPTERS, GAME_SLUGS
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.local_embedder import LocalEmbedder
from app.ingestion.scraper import Scraper
from app.ingestion.spoiler_tagger import SpoilerTagger
from app.llm.base import TaskType
from app.llm.router import build_llm_router
from app.tracing.langfuse import LangfuseTracer

logger = logging.getLogger(__name__)

# Duck-typed: anything with `async def embed(texts) -> list[list[float]]` works.
Embedder = GeminiEmbedder | LocalEmbedder


def make_embedder(settings: Settings) -> Embedder:
    """Pick an embedder implementation based on settings.embedding_backend.

    - local (default): bge-base-en-v1.5 on CPU, free, no quotas.
    - gemini: gemini-embedding-001 @ 768d, requires GEMINI_API_KEY, free-tier
      rate limits apply.
    """
    backend = settings.embedding_backend.lower()
    if backend == "local":
        return LocalEmbedder(model_name=settings.local_embedding_model)
    if backend == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "embedding_backend=gemini requires GEMINI_API_KEY to be set."
            )
        return GeminiEmbedder(api_key=settings.gemini_api_key)
    raise ValueError(
        f"Unknown embedding_backend={backend!r}; expected 'local' or 'gemini'."
    )


@dataclass
class IngestResult:
    game_slug: str
    pages_fetched: int
    chunks_created: int
    passages_upserted: int
    passages_skipped: int


@dataclass(frozen=True)
class _ExistingPassage:
    id: int
    source_url: str
    content_hash: str
    spoiler_tier: int
    embedding_backend: str | None
    embedding_model: str | None
    has_embedding: bool


def _embedder_identity(embedder: Embedder) -> tuple[str, str]:
    """Return (backend, model) for DB writes. Falls back to "unknown"/class name."""
    backend = getattr(embedder, "backend_name", None)
    model = getattr(embedder, "model_name", None)
    return (
        backend if isinstance(backend, str) and backend else "unknown",
        model if isinstance(model, str) and model else type(embedder).__name__,
    )


async def run_ingestion(
    adapter: GameAdapter,
    scraper: Scraper,
    chunker: Chunker | None,
    embedder: Embedder,
    session: AsyncSession,
    dry_run: bool = False,
    spoiler_tagger: SpoilerTagger | None = None,
    entity_extractor: EntityExtractor | None = None,
) -> IngestResult:
    chunker = chunker or adapter.chunker
    embedder_backend, embedder_model = _embedder_identity(embedder)
    urls = adapter.get_article_urls()
    current_url_set = set(urls)
    pages_fetched = 0
    failed_urls: set[str] = set()
    all_chunks = []
    extracted_entities: list[ExtractedEntity] = []

    for url in urls:
        page = await scraper.fetch(url)
        if page is None:
            # Reason (robots.txt / API error / non-JSON / shape mismatch) is
            # already logged by the scraper at WARNING level.
            logger.info("Skipped: %s", url)
            failed_urls.add(url)
            continue
        pages_fetched += 1
        chunks = chunker.chunk(page.text, url, title=page.title)
        all_chunks.extend(chunks)
        if entity_extractor is not None:
            extracted_entities.extend(
                await entity_extractor.extract(
                    page_text=page.text, source_url=url, game_slug=adapter.slug,
                )
            )

    existing_by_source: dict[str, dict[str, _ExistingPassage]] = defaultdict(dict)
    existing_result = await session.execute(
        select(
            Passage.id,
            Passage.source_url,
            Passage.content_hash,
            Passage.spoiler_tier,
            Passage.embedding_backend,
            Passage.embedding_model,
            Passage.embedding.is_not(None).label("has_embedding"),
        )
        .where(Passage.game_slug == adapter.slug)
    )
    existing_rows = existing_result.all()
    for row in existing_rows:
        existing_by_source[row.source_url][row.content_hash] = _ExistingPassage(
            id=row.id,
            source_url=row.source_url,
            content_hash=row.content_hash,
            spoiler_tier=row.spoiler_tier,
            embedding_backend=row.embedding_backend,
            embedding_model=row.embedding_model,
            has_embedding=bool(row.has_embedding),
        )

    seen_hashes_by_source: dict[str, set[str]] = defaultdict(set)
    fresh_chunks = []
    retained_chunks = []
    passages_skipped = 0
    for chunk in all_chunks:
        seen_hashes_by_source[chunk.source_url].add(chunk.content_hash)
        existing = existing_by_source.get(chunk.source_url, {}).get(chunk.content_hash)
        if existing is not None:
            retained_chunks.append((chunk, existing))
            continue
        fresh_chunks.append(chunk)

    stale_ids: list[int] = []
    for source_url, hashes_to_ids in existing_by_source.items():
        if source_url not in current_url_set:
            stale_ids.extend(row.id for row in hashes_to_ids.values())
            continue
        if source_url in failed_urls:
            continue
        current_hashes = seen_hashes_by_source.get(source_url, set())
        for content_hash, passage in hashes_to_ids.items():
            if content_hash not in current_hashes:
                stale_ids.append(passage.id)

    if not all_chunks and not stale_ids:
        return IngestResult(
            game_slug=adapter.slug,
            pages_fetched=pages_fetched,
            chunks_created=0,
            passages_upserted=0,
            passages_skipped=0,
        )

    retained_to_refresh = []
    for chunk, existing in retained_chunks:
        needs_embedding_refresh = (
            not existing.has_embedding
            or existing.embedding_backend != embedder_backend
            or existing.embedding_model != embedder_model
        )
        retained_to_refresh.append((chunk, existing, needs_embedding_refresh))

    chunks_needing_embeddings = [
        *fresh_chunks,
        *[
            chunk
            for chunk, _, needs_embedding_refresh in retained_to_refresh
            if needs_embedding_refresh
        ],
    ]
    embedding_texts = [chunk.content for chunk in chunks_needing_embeddings]
    embeddings = await embedder.embed(embedding_texts) if embedding_texts else []
    if len(embeddings) != len(chunks_needing_embeddings):
        raise RuntimeError(
            "Embedder returned a mismatched number of vectors for the chunks being ingested."
        )
    embeddings_by_key = {
        (chunk.source_url, chunk.content_hash): embedding
        for chunk, embedding in zip(chunks_needing_embeddings, embeddings)
    }

    rows = []
    for chunk in fresh_chunks:
        tier = 0
        if spoiler_tagger is not None:
            tier = await spoiler_tagger.tag_async(chunk.content, game_slug=adapter.slug)
        rows.append({
            "game_slug": adapter.slug,
            "source_url": chunk.source_url,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "spoiler_tier": tier,
            "embedding": embeddings_by_key[(chunk.source_url, chunk.content_hash)],
            "embedding_backend": embedder_backend,
            "embedding_model": embedder_model,
        })

    updates = []
    for chunk, existing, needs_embedding_refresh in retained_to_refresh:
        next_tier = existing.spoiler_tier
        if spoiler_tagger is not None:
            next_tier = await spoiler_tagger.tag_async(chunk.content, game_slug=adapter.slug)

        values = {}
        if next_tier != existing.spoiler_tier:
            values["spoiler_tier"] = next_tier
        if needs_embedding_refresh:
            values["embedding"] = embeddings_by_key[(chunk.source_url, chunk.content_hash)]
            values["embedding_backend"] = embedder_backend
            values["embedding_model"] = embedder_model

        if values:
            updates.append((existing.id, values))
        else:
            passages_skipped += 1

    if spoiler_tagger is not None:
        nonzero = sum(1 for r in rows if r["spoiler_tier"] > 0) + sum(
            1
            for _, values in updates
            if values.get("spoiler_tier", 0) > 0
        )
        logger.info(
            "Spoiler-tagged %d chunks (non-zero tier: %d)",
            len(rows) + len(retained_to_refresh),
            nonzero,
        )

    passages_upserted = 0
    if not dry_run:
        if stale_ids:
            await session.execute(delete(Passage).where(Passage.id.in_(stale_ids)))
        for passage_id, values in updates:
            await session.execute(
                update(Passage)
                .where(Passage.id == passage_id)
                .values(**values, updated_at=func.now())
            )
        if rows:
            stmt = pg_insert(Passage).values(rows)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_passages_game_source_content_hash",
            )
            await session.execute(stmt)
        passages_upserted = len(rows) + len(updates)
        if stale_ids or rows or updates:
            await session.commit()

    if not dry_run and entity_extractor is not None and extracted_entities:
        await upsert_entities(
            session=session, game_slug=adapter.slug, entities=extracted_entities,
        )

    return IngestResult(
        game_slug=adapter.slug,
        pages_fetched=pages_fetched,
        chunks_created=len(all_chunks),
        passages_upserted=passages_upserted,
        passages_skipped=passages_skipped,
    )


async def _main(args: argparse.Namespace) -> None:
    from pathlib import Path

    from app.config import get_settings
    from app.db.session import get_session_factory

    settings = get_settings()
    tracer = LangfuseTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    if args.game not in ADAPTERS:
        raise SystemExit(f"Unknown game: {args.game}")

    adapter = ADAPTERS[args.game]
    scraper = Scraper(
        cache_dir=Path(".scraper_cache") / args.game,
        crawl_delay=adapter.sources[0].crawl_delay,
        cache_ttl=(
            None
            if args.cache_ttl_hours <= 0
            else timedelta(hours=args.cache_ttl_hours)
        ),
        force_refresh=args.refresh_cache,
    )
    embedder = make_embedder(settings)
    logger.info("Using %s as embedding backend", type(embedder).__name__)

    router = build_llm_router(settings)
    spoiler_tagger = SpoilerTagger(llm=router.for_task(TaskType.TAG), tracer=tracer)

    entity_extractor = None
    if adapter.entity_schema:
        entity_extractor = EntityExtractor(
            llm=router.for_task(TaskType.TAG),
            allowed_types={t.name for t in adapter.entity_schema},
        )

    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            result = await run_ingestion(
                adapter=adapter,
                scraper=scraper,
                chunker=adapter.chunker,
                embedder=embedder,
                session=session,
                dry_run=args.dry_run,
                spoiler_tagger=spoiler_tagger,
                entity_extractor=entity_extractor,
            )
    finally:
        tracer.flush()

    print(f"Game:              {result.game_slug}")
    print(f"Pages fetched:     {result.pages_fetched}")
    print(f"Chunks created:    {result.chunks_created}")
    print(f"Passages upserted: {result.passages_upserted}")
    print(f"Dry run:           {args.dry_run}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Loresmith ingestion pipeline")
    parser.add_argument(
        "--game",
        required=True,
        choices=sorted(GAME_SLUGS),
        help="Game slug to ingest",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip DB writes")
    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=24.0,
        help=(
            "Reuse cached source HTML only if it is newer than this many hours. "
            "Use 0 to disable cache expiry."
        ),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached source HTML and refetch every source page.",
    )
    asyncio.run(_main(parser.parse_args()))
