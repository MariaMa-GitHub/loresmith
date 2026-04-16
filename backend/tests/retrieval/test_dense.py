from unittest.mock import AsyncMock, MagicMock

import pytest

from app.retrieval.dense import DenseHit, DenseRetriever


def test_dense_hit_fields():
    hit = DenseHit(passage_id=1, score=0.92, content="some text", source_url="https://x.com")
    assert hit.passage_id == 1
    assert hit.score == 0.92


@pytest.mark.asyncio
async def test_dense_retriever_returns_hits():
    mock_session = AsyncMock()
    # Simulate DB returning two rows
    row1 = MagicMock()
    row1.id = 1
    row1.content = "Zagreus wields the Stygian Blade."
    row1.source_url = "https://hades.fandom.com/wiki/Zagreus"
    row1.distance = 0.12  # cosine distance (lower = more similar)

    row2 = MagicMock()
    row2.id = 2
    row2.content = "The Stygian Blade is an Infernal Arm."
    row2.source_url = "https://hades.fandom.com/wiki/Stygian_Blade"
    row2.distance = 0.25

    mock_result = MagicMock()
    mock_result.all.return_value = [row1, row2]
    mock_session.execute = AsyncMock(return_value=mock_result)

    retriever = DenseRetriever()
    query_embedding = [0.1] * 768
    hits = await retriever.search(
        session=mock_session,
        game_slug="hades",
        query_embedding=query_embedding,
        top_k=5,
        max_spoiler_tier=0,
    )

    assert len(hits) == 2
    assert hits[0].passage_id == 1
    assert hits[0].score > hits[1].score  # higher score = more similar


@pytest.mark.asyncio
async def test_dense_retriever_filters_by_game_slug_and_spoiler():
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    retriever = DenseRetriever()
    hits = await retriever.search(
        session=mock_session,
        game_slug="hades2",
        query_embedding=[0.0] * 768,
        top_k=5,
        max_spoiler_tier=1,
    )
    assert hits == []

    # Verify the compiled SQL carries the expected filter clauses.
    compiled = mock_session.execute.call_args.args[0].compile(
        compile_kwargs={"literal_binds": True}
    )
    sql = str(compiled).lower()
    assert "game_slug" in sql
    assert "spoiler_tier" in sql
    assert "embedding" in sql  # either ordering on distance or IS NOT NULL
