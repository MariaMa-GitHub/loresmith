from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from app.retrieval.hybrid import HybridHit


@dataclass(frozen=True)
class RerankedHit:
    passage_id: int
    rerank_score: float
    content: str
    source_url: str


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        hits: list[HybridHit],
        top_k: int,
    ) -> list[RerankedHit]: ...


class NullReranker:
    """Identity reranker. Preserves input order, trims to top_k, no scoring."""

    async def rerank(
        self,
        query: str,
        hits: list[HybridHit],
        top_k: int,
    ) -> list[RerankedHit]:
        return [
            RerankedHit(
                passage_id=h.passage_id,
                rerank_score=h.rrf_score,
                content=h.content,
                source_url=h.source_url,
            )
            for h in hits[:top_k]
        ]


class CrossEncoderReranker:
    """Cross-encoder reranker (CPU). Lazy-loads the model on first call."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        score_floor: float | None = None,
    ) -> None:
        self._model_name = model_name
        self._score_floor = score_floor
        self._model = None

    def _load_model_sync(self):
        from sentence_transformers import CrossEncoder
        # max_length matches bge-reranker-v2-m3's 8K training window; avoids
        # silent truncation on passages that exceed 512 tokens.
        return CrossEncoder(self._model_name, max_length=8192)

    async def _ensure_model(self):
        if self._model is None:
            # Model init is CPU-bound (weight loading); run in a thread to
            # avoid blocking the event loop on first use.
            self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    async def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = await self._ensure_model()
        # CrossEncoder.predict is CPU-bound; push it off the event loop.
        return await asyncio.to_thread(lambda: list(model.predict(pairs)))

    async def rerank(
        self,
        query: str,
        hits: list[HybridHit],
        top_k: int,
    ) -> list[RerankedHit]:
        if not hits:
            return []
        pairs = [(query, h.content) for h in hits]
        scores = await self._score_pairs(pairs)
        scored = list(zip(hits, scores, strict=True))
        if self._score_floor is not None:
            scored = [(hit, score) for hit, score in scored if score >= self._score_floor]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            RerankedHit(
                passage_id=hit.passage_id,
                rerank_score=float(score),
                content=hit.content,
                source_url=hit.source_url,
            )
            for hit, score in scored[:top_k]
        ]
