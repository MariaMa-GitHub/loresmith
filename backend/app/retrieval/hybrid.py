from dataclasses import dataclass

from app.retrieval.bm25 import BM25Hit
from app.retrieval.dense import DenseHit

_RRF_K = 60  # standard RRF constant


@dataclass
class HybridHit:
    passage_id: int
    rrf_score: float
    content: str
    source_url: str


def rrf_fuse(
    bm25_hits: list[BM25Hit],
    dense_hits: list[DenseHit],
    top_k: int = 10,
    bm25_source_map: dict[int, str] | None = None,
) -> list[HybridHit]:
    """Merge BM25 and dense ranked lists with Reciprocal Rank Fusion.

    ``bm25_source_map`` is an optional ``{passage_id: source_url}`` lookup used
    to enrich BM25-only passages with a citation URL. ``BM25Hit`` does not
    carry a source URL (Week 2 design), so without the map a passage found
    only by BM25 would render as ``[N] (Source: )`` in the prompt — breaking
    inline citation quality. When a passage appears in both lists the dense
    hit's ``source_url`` always wins (it's loaded from the DB and guaranteed
    fresh; the map is a snapshot taken at BM25-rebuild time).
    """
    scores: dict[int, float] = {}

    for rank, hit in enumerate(bm25_hits):
        scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

    for rank, hit in enumerate(dense_hits):
        scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

    sources = bm25_source_map or {}
    meta: dict[int, dict] = {}
    for hit in bm25_hits:
        meta[hit.passage_id] = {
            "content": hit.content,
            "source_url": sources.get(hit.passage_id, ""),
        }
    for hit in dense_hits:
        if hit.passage_id not in meta:
            meta[hit.passage_id] = {"content": hit.content, "source_url": hit.source_url}
        else:
            # Dense URL always wins over the BM25 map snapshot.
            meta[hit.passage_id]["source_url"] = hit.source_url

    sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)[:top_k]

    return [
        HybridHit(
            passage_id=pid,
            rrf_score=scores[pid],
            content=meta[pid]["content"],
            source_url=meta[pid]["source_url"],
        )
        for pid in sorted_ids
    ]
