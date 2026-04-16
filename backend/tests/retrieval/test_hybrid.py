from app.retrieval.bm25 import BM25Hit
from app.retrieval.dense import DenseHit
from app.retrieval.hybrid import HybridHit, rrf_fuse


def test_rrf_fuse_combines_both_lists():
    bm25_hits = [
        BM25Hit(passage_id=1, score=2.5, content="passage one"),
        BM25Hit(passage_id=2, score=1.8, content="passage two"),
    ]
    dense_hits = [
        DenseHit(passage_id=2, score=0.95, content="passage two", source_url="https://x.com/2"),
        DenseHit(passage_id=3, score=0.88, content="passage three", source_url="https://x.com/3"),
    ]
    result = rrf_fuse(bm25_hits=bm25_hits, dense_hits=dense_hits, top_k=5)

    ids = [h.passage_id for h in result]
    # passage 2 appears in both lists — should rank highly
    assert 2 in ids
    assert 1 in ids
    assert 3 in ids


def test_rrf_fuse_passage_in_both_lists_ranks_higher():
    bm25_hits = [
        BM25Hit(passage_id=99, score=5.0, content="best bm25"),
        BM25Hit(passage_id=1, score=4.0, content="shared passage"),
    ]
    dense_hits = [
        DenseHit(passage_id=1, score=0.99, content="shared passage", source_url="https://x.com"),
        DenseHit(passage_id=88, score=0.80, content="best dense", source_url="https://x.com/88"),
    ]
    result = rrf_fuse(bm25_hits=bm25_hits, dense_hits=dense_hits, top_k=5)
    # passage_id=1 appears in both; should be ranked first
    assert result[0].passage_id == 1


def test_rrf_fuse_top_k_limits_results():
    bm25_hits = [
        BM25Hit(passage_id=i, score=float(i), content=f"p{i}") for i in range(10)
    ]
    dense_hits = [
        DenseHit(passage_id=i, score=0.5, content=f"p{i}", source_url="x")
        for i in range(10)
    ]
    result = rrf_fuse(bm25_hits=bm25_hits, dense_hits=dense_hits, top_k=3)
    assert len(result) == 3


def test_rrf_fuse_empty_inputs_returns_empty():
    assert rrf_fuse(bm25_hits=[], dense_hits=[], top_k=10) == []


def test_rrf_fuse_returns_hybrid_hit_objects():
    bm25_hits = [BM25Hit(passage_id=1, score=1.0, content="text")]
    dense_hits = [DenseHit(passage_id=1, score=0.9, content="text", source_url="https://x.com")]
    result = rrf_fuse(bm25_hits=bm25_hits, dense_hits=dense_hits, top_k=5)
    assert all(isinstance(h, HybridHit) for h in result)
    assert result[0].source_url == "https://x.com"


def test_rrf_fuse_bm25_only_uses_source_map():
    """BM25Hit has no source_url. The optional map keeps citations populated
    when a passage is only found by BM25, not by dense."""
    bm25_hits = [BM25Hit(passage_id=7, score=2.0, content="BM25-only content")]
    result = rrf_fuse(
        bm25_hits=bm25_hits,
        dense_hits=[],
        top_k=5,
        bm25_source_map={7: "https://hades.fandom.com/wiki/Passage7"},
    )
    assert result[0].passage_id == 7
    assert result[0].source_url == "https://hades.fandom.com/wiki/Passage7"


def test_rrf_fuse_dense_source_url_wins_over_map():
    """If a passage is in both lists, the dense hit's source_url takes precedence."""
    bm25_hits = [BM25Hit(passage_id=7, score=2.0, content="text")]
    dense_hits = [DenseHit(passage_id=7, score=0.9, content="text", source_url="https://dense.example/7")]
    result = rrf_fuse(
        bm25_hits=bm25_hits,
        dense_hits=dense_hits,
        top_k=5,
        bm25_source_map={7: "https://stale.example/7"},
    )
    assert result[0].source_url == "https://dense.example/7"
