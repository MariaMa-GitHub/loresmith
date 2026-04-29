import pytest

from app.retrieval.hybrid import HybridHit
from app.retrieval.reranker import CrossEncoderReranker, NullReranker, RerankedHit


def _hit(pid: int, content: str, url: str = "https://example.com/x") -> HybridHit:
    return HybridHit(passage_id=pid, rrf_score=0.1, content=content, source_url=url)


@pytest.mark.asyncio
async def test_null_reranker_returns_first_k_in_input_order():
    reranker = NullReranker()
    hits = [_hit(1, "a"), _hit(2, "b"), _hit(3, "c")]
    result = await reranker.rerank(query="q", hits=hits, top_k=2)
    assert [r.passage_id for r in result] == [1, 2]
    assert all(isinstance(r, RerankedHit) for r in result)


@pytest.mark.asyncio
async def test_cross_encoder_reranker_reorders_by_score(monkeypatch):
    reranker = CrossEncoderReranker(model_name="dummy/model")
    hits = [_hit(1, "alpha"), _hit(2, "beta"), _hit(3, "gamma")]

    async def fake_async(pairs):
        scores = {"alpha": 0.2, "beta": 0.1, "gamma": 0.9}
        return [scores[p[1]] for p in pairs]

    monkeypatch.setattr(reranker, "_score_pairs", fake_async)
    result = await reranker.rerank(query="q", hits=hits, top_k=2)
    assert [r.passage_id for r in result] == [3, 1]
    assert result[0].rerank_score == pytest.approx(0.9)
    assert result[0].source_url == hits[2].source_url


@pytest.mark.asyncio
async def test_cross_encoder_handles_empty_input(monkeypatch):
    reranker = CrossEncoderReranker(model_name="dummy/model")

    async def unreachable(pairs):
        raise AssertionError("model must not be invoked for empty input")
    monkeypatch.setattr(reranker, "_score_pairs", unreachable)
    result = await reranker.rerank(query="q", hits=[], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_cross_encoder_respects_top_k_larger_than_input(monkeypatch):
    reranker = CrossEncoderReranker(model_name="dummy/model")
    hits = [_hit(1, "only")]

    async def one_score(pairs):
        return [0.5]
    monkeypatch.setattr(reranker, "_score_pairs", one_score)
    result = await reranker.rerank(query="q", hits=hits, top_k=10)
    assert len(result) == 1
    assert result[0].passage_id == 1
