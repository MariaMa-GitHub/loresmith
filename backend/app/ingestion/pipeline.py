"""Ingestion pipeline CLI.

Usage:
    python -m app.ingestion.pipeline --game hades
    python -m app.ingestion.pipeline --game hades --dry-run
"""
import argparse
import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import GameAdapter
from app.config import Settings
from app.db.models import Passage
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.local_embedder import LocalEmbedder
from app.ingestion.scraper import Scraper
from app.ingestion.spoiler_tagger import SpoilerTagger

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


async def run_ingestion(
    adapter: GameAdapter,
    scraper: Scraper,
    chunker: Chunker,
    embedder: Embedder,
    session: AsyncSession,
    dry_run: bool = False,
    spoiler_tagger: SpoilerTagger | None = None,
) -> IngestResult:
    urls = adapter.get_article_urls()
    pages_fetched = 0
    all_chunks = []

    for url in urls:
        page = await scraper.fetch(url)
        if page is None:
            # Reason (robots.txt / API error / non-JSON / shape mismatch) is
            # already logged by the scraper at WARNING level.
            logger.info("Skipped: %s", url)
            continue
        pages_fetched += 1
        chunks = chunker.chunk(page.text, url, title=page.title)
        all_chunks.extend(chunks)

    if not all_chunks:
        return IngestResult(
            game_slug=adapter.slug,
            pages_fetched=pages_fetched,
            chunks_created=0,
            passages_upserted=0,
            passages_skipped=0,
        )

    texts = [c.content for c in all_chunks]
    embeddings = await embedder.embed(texts)

    passages_upserted = 0
    # passages_skipped: ON CONFLICT DO UPDATE touches every conflicting row so
    # we cannot distinguish new inserts from no-op updates without a second
    # query. Tracking skipped passages is deferred to Week 3+ integration tests.
    passages_skipped = 0

    rows = []
    for chunk, embedding in zip(all_chunks, embeddings):
        tier = 0
        if spoiler_tagger is not None:
            tier = await spoiler_tagger.tag_async(chunk.content, game_slug=adapter.slug)
        rows.append({
            "game_slug": adapter.slug,
            "source_url": chunk.source_url,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "spoiler_tier": tier,
            "embedding": embedding,
        })

    if spoiler_tagger is not None:
        nonzero = sum(1 for r in rows if r["spoiler_tier"] > 0)
        logger.info(
            "Spoiler-tagged %d chunks (non-zero tier: %d)", len(rows), nonzero
        )

    if not dry_run:
        stmt = pg_insert(Passage).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_passages_content_hash",
            set_={
                "source_url": stmt.excluded.source_url,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        await session.execute(stmt)
        await session.commit()
        passages_upserted = len(rows)

    return IngestResult(
        game_slug=adapter.slug,
        pages_fetched=pages_fetched,
        chunks_created=len(all_chunks),
        passages_upserted=passages_upserted,
        passages_skipped=passages_skipped,
    )


async def _main(args: argparse.Namespace) -> None:
    from pathlib import Path

    from app.adapters.hades import HadesAdapter
    from app.adapters.hades2 import HadesIIAdapter
    from app.config import get_settings
    from app.db.session import get_session_factory

    settings = get_settings()

    adapter_map = {
        "hades": HadesAdapter,
        "hades2": HadesIIAdapter,
    }

    if args.game not in adapter_map:
        raise SystemExit(f"Unknown game: {args.game}")

    adapter = adapter_map[args.game]()
    scraper = Scraper(
        cache_dir=Path(".scraper_cache") / args.game,
        crawl_delay=adapter.sources[0].crawl_delay,
    )
    chunker = Chunker(chunk_size=adapter.chunk_size, overlap=adapter.chunk_overlap)
    embedder = make_embedder(settings)
    logger.info("Using %s as embedding backend", type(embedder).__name__)

    from app.llm.gemini import GeminiProvider

    fast_llm = (
        GeminiProvider(api_key=settings.gemini_api_key, model_name="gemini-2.5-flash-lite")
        if settings.gemini_api_key
        else None
    )
    spoiler_tagger = SpoilerTagger(llm=fast_llm)

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await run_ingestion(
            adapter=adapter,
            scraper=scraper,
            chunker=chunker,
            embedder=embedder,
            session=session,
            dry_run=args.dry_run,
            spoiler_tagger=spoiler_tagger,
        )

    print(f"Game:              {result.game_slug}")
    print(f"Pages fetched:     {result.pages_fetched}")
    print(f"Chunks created:    {result.chunks_created}")
    print(f"Passages upserted: {result.passages_upserted}")
    print(f"Dry run:           {args.dry_run}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Loresmith ingestion pipeline")
    parser.add_argument(
        "--game", required=True, choices=["hades", "hades2"], help="Game slug to ingest"
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip DB writes")
    asyncio.run(_main(parser.parse_args()))
