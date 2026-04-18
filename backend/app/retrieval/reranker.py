from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

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

    _model: CrossEncoder | None

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            # max_length matches the model's training window; bge-reranker-base
            # truncates on its own above 512 tokens.
            self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model

    async def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._ensure_model()
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
