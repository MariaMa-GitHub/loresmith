from app.retrieval.bm25 import BM25Hit, BM25Index


def test_bm25_search_returns_relevant_result():
    index = BM25Index()
    ids = [1, 2, 3]
    texts = [
        "Zagreus wields the Stygian Blade in his escape from Tartarus.",
        "Nyx is the Goddess of Night who lives in the House of Hades.",
        "The Mirror of Night allows Zagreus to spend Darkness on upgrades.",
    ]
    index.build(ids, texts)

    hits = index.search("Zagreus blade escape", top_k=3)
    assert len(hits) > 0
    assert hits[0].passage_id == 1  # Most relevant is passage about Zagreus + blade


def test_bm25_search_returns_bm25hit_objects():
    index = BM25Index()
    index.build([10, 20], ["alpha beta gamma", "delta epsilon zeta"])
    hits = index.search("alpha", top_k=2)
    assert all(isinstance(h, BM25Hit) for h in hits)
    assert all(hasattr(h, "passage_id") for h in hits)
    assert all(hasattr(h, "score") for h in hits)
    assert all(hasattr(h, "content") for h in hits)


def test_bm25_search_empty_index_returns_empty():
    index = BM25Index()
    assert index.search("anything") == []


def test_bm25_search_zero_score_results_excluded():
    index = BM25Index()
    index.build([1, 2], ["hades underworld", "olympus zeus"])
    hits = index.search("completely unrelated superlongword")
    # All scores may be 0; none should be returned
    assert all(h.score > 0 for h in hits)


def test_bm25_top_k_limits_results():
    index = BM25Index()
    ids = list(range(20))
    texts = [f"passage about hades zagreus topic{i}" for i in range(20)]
    index.build(ids, texts)

    hits = index.search("hades zagreus", top_k=5)
    assert len(hits) <= 5


def test_bm25_rebuild_replaces_old_index():
    index = BM25Index()
    index.build([1], ["old content only"])
    index.build([2], ["new content only"])
    hits = index.search("new content", top_k=1)
    assert hits[0].passage_id == 2
