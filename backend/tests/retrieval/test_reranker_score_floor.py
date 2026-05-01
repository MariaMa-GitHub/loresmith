from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.retrieval.hybrid import HybridHit
from app.retrieval.reranker import CrossEncoderReranker


def _hit(i: int) -> HybridHit:
    return HybridHit(
        passage_id=i, rrf_score=0.5, content=f"content {i}", source_url=f"http://x/{i}"
    )


@pytest.mark.asyncio
async def test_score_floor_drops_negative_candidates():
    reranker = CrossEncoderReranker(score_floor=0.0)
    scores = [0.8, -0.2, 0.5, -1.0]
    hits = [_hit(i) for i in range(4)]
    with patch.object(reranker, "_score_pairs", new=AsyncMock(return_value=scores)):
        result = await reranker.rerank(query="q", hits=hits, top_k=10)
    assert len(result) == 2
    assert all(r.rerank_score >= 0.0 for r in result)


@pytest.mark.asyncio
async def test_no_floor_returns_all_candidates():
    reranker = CrossEncoderReranker(score_floor=None)
    scores = [0.8, -0.2, 0.5]
    hits = [_hit(i) for i in range(3)]
    with patch.object(reranker, "_score_pairs", new=AsyncMock(return_value=scores)):
        result = await reranker.rerank(query="q", hits=hits, top_k=10)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_floor_above_all_scores_returns_empty():
    reranker = CrossEncoderReranker(score_floor=5.0)
    scores = [0.8, 0.5]
    hits = [_hit(i) for i in range(2)]
    with patch.object(reranker, "_score_pairs", new=AsyncMock(return_value=scores)):
        result = await reranker.rerank(query="q", hits=hits, top_k=10)
    assert result == []
