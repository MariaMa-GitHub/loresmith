from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.semantic_cache import CachedAnswer, SemanticCache


def _row(distance: float, answer: str = "cached"):
    row = MagicMock()
    row.id = 1
    row.distance = distance
    row.response = {
        "answer": answer,
        "passages": [{"passage_id": 1, "content": "p", "source_url": "u"}],
        "citations": [{"index": 1, "source_url": "u", "title": "t"}],
    }
    return row


@pytest.mark.asyncio
async def test_cache_miss_below_threshold_returns_none():
    session = AsyncMock()
    result = MagicMock()
    result.first.return_value = _row(distance=0.5)  # similarity 0.5
    session.execute = AsyncMock(return_value=result)

    cache = SemanticCache(similarity_threshold=0.92)
    hit = await cache.get(
        session=session,
        game_slug="hades",
        corpus_revision="10:100",
        max_spoiler_tier=0,
        embedding_backend="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        query_embedding=[0.1] * 768,
    )
    assert hit is None


@pytest.mark.asyncio
async def test_cache_hit_above_threshold_returns_cached_answer():
    session = AsyncMock()
    result = MagicMock()
    # distance 0.05 → similarity 0.95, above threshold 0.92
    result.first.return_value = _row(distance=0.05, answer="hello")
    session.execute = AsyncMock(return_value=result)

    cache = SemanticCache(similarity_threshold=0.92)
    hit = await cache.get(
        session=session,
        game_slug="hades",
        corpus_revision="10:100",
        max_spoiler_tier=0,
        embedding_backend="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        query_embedding=[0.1] * 768,
    )
    assert isinstance(hit, CachedAnswer)
    assert hit.answer == "hello"
    assert hit.passages[0]["source_url"] == "u"


@pytest.mark.asyncio
async def test_cache_empty_result_returns_none():
    session = AsyncMock()
    result = MagicMock()
    result.first.return_value = None
    session.execute = AsyncMock(return_value=result)

    cache = SemanticCache(similarity_threshold=0.92)
    hit = await cache.get(
        session=session,
        game_slug="hades",
        corpus_revision="10:100",
        max_spoiler_tier=0,
        embedding_backend="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        query_embedding=[0.1] * 768,
    )
    assert hit is None


@pytest.mark.asyncio
async def test_cache_put_inserts_row_with_full_scope():
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    cache = SemanticCache(similarity_threshold=0.92)
    await cache.put(
        session=session,
        game_slug="hades",
        corpus_revision="10:100",
        max_spoiler_tier=0,
        embedding_backend="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        query_text="who is zag",
        query_embedding=[0.1] * 768,
        answer="Zagreus is …",
        passages=[{"passage_id": 1, "content": "p", "source_url": "u"}],
        citations=[{"index": 1, "source_url": "u", "title": "t"}],
    )
    session.add.assert_called_once()
    saved = session.add.call_args.args[0]
    assert saved.game_slug == "hades"
    assert saved.corpus_revision == "10:100"
    assert saved.max_spoiler_tier == 0
    assert saved.embedding_backend == "local"
    assert saved.embedding_model == "BAAI/bge-base-en-v1.5"
    assert saved.query_text == "who is zag"
    assert saved.response["answer"] == "Zagreus is …"
    session.commit.assert_awaited_once()
