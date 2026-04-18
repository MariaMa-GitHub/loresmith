import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def test_alembic_check_passes():
    """Verifies migration scripts are syntactically valid (no DB connection needed)."""
    result = subprocess.run(
        [sys.executable, "-c", "import alembic; print('ok')"],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_baseline_migration_exists():
    versions = list((BACKEND_ROOT / "alembic" / "versions").glob("*.py"))
    assert any("001" in p.name or "baseline" in p.name for p in versions), (
        "Expected baseline migration file not found"
    )


def test_recent_passage_migrations_exist():
    versions = {p.name for p in (BACKEND_ROOT / "alembic" / "versions").glob("*.py")}
    assert any(name.startswith("003_") for name in versions)
    assert any(name.startswith("004_") for name in versions)
    assert any(name.startswith("005_") for name in versions)
    assert any(name.startswith("006_") for name in versions)
    assert any(name.startswith("007_") for name in versions)


async def _reset_database(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


async def _inspect_migrated_schema(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            vector_ext = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            assert vector_ext.scalar_one() == "vector"

            version = await conn.execute(text("SELECT version_num FROM alembic_version"))
            assert version.scalar_one() == "007"

            constraint = await conn.execute(
                text(
                    """
                    SELECT array_agg(a.attname ORDER BY x.ordinality)
                    FROM pg_constraint c
                    JOIN unnest(c.conkey) WITH ORDINALITY AS x(attnum, ordinality) ON TRUE
                    JOIN pg_attribute a
                      ON a.attrelid = c.conrelid
                     AND a.attnum = x.attnum
                    WHERE c.conrelid = 'passages'::regclass
                      AND c.conname = 'uq_passages_game_source_content_hash'
                    GROUP BY c.conname
                    """
                )
            )
            assert constraint.scalar_one() == ["game_slug", "source_url", "content_hash"]

            hnsw_index = await conn.execute(
                text(
                    """
                    SELECT indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename = 'passages'
                      AND indexname = 'ix_passages_embedding_hnsw'
                    """
                )
            )
            indexdef = hnsw_index.scalar_one()
            lowered = indexdef.lower()
            assert "using hnsw" in lowered
            assert "embedding" in lowered

            owner_token_column = await conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'chat_sessions'
                      AND column_name = 'owner_token'
                    """
                )
            )
            assert owner_token_column.scalar_one() == "owner_token"

            embedding_identity_columns = await conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'passages'
                      AND column_name IN ('embedding_backend', 'embedding_model')
                    ORDER BY column_name
                    """
                )
            )
            assert embedding_identity_columns.scalars().all() == [
                "embedding_backend",
                "embedding_model",
            ]

            semantic_cache_scope_columns = await conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'semantic_cache'
                      AND column_name IN (
                          'corpus_revision',
                          'max_spoiler_tier',
                          'embedding_backend',
                          'embedding_model'
                      )
                    ORDER BY column_name
                    """
                )
            )
            assert semantic_cache_scope_columns.scalars().all() == [
                "corpus_revision",
                "embedding_backend",
                "embedding_model",
                "max_spoiler_tier",
            ]
    finally:
        await engine.dispose()


def test_alembic_upgrade_head_applies_expected_passage_schema():
    database_url = os.environ.get("MIGRATION_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("MIGRATION_TEST_DATABASE_URL is not configured")

    asyncio.run(_reset_database(database_url))

    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr

    asyncio.run(_inspect_migrated_schema(database_url))
