"""Local embedder using sentence-transformers (default: bge-base-en-v1.5).

Produces 768-dim unit-norm vectors on CPU with zero external API dependencies.
The underlying `SentenceTransformer.encode` call is blocking, so we run it via
`asyncio.to_thread` to avoid starving the event loop during ingestion.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
_DEFAULT_BATCH_SIZE = 32  # Balanced for CPU throughput and memory.
_EXPECTED_DIM = 768  # Must match EMBEDDING_DIM in app/db/models.py.


class LocalEmbedder:
    """Embeds passages locally with a sentence-transformers model.

    The model is lazily loaded on the first `embed()` call; the first load
    downloads ~400 MB of weights from Hugging Face and caches them under
    `~/.cache/huggingface/`. Subsequent loads are from cache.

    `normalize_embeddings=True` returns unit-norm vectors so cosine similarity
    can be computed as a plain dot product in pgvector.
    """

    backend_name = "local"
    model_name: str

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._batch_size = batch_size
        self._device = device  # None → let sentence-transformers pick (cpu/cuda/mps).
        self._model: SentenceTransformer | None = None
        self._load_lock = asyncio.Lock()

    def _load_model_sync(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading local embedding model %s (first load may download ~400 MB)",
            self.model_name,
        )
        model = SentenceTransformer(self.model_name, device=self._device)
        # sentence-transformers 5.x renamed the method; keep a fallback for
        # 3.x/4.x so the dependency pin stays wide.
        get_dim = getattr(
            model, "get_embedding_dimension", None
        ) or model.get_sentence_embedding_dimension
        dim = get_dim()
        if dim != _EXPECTED_DIM:
            raise RuntimeError(
                f"Local embedding model {self.model_name!r} emits {dim}-dim "
                f"vectors but the pgvector column expects {_EXPECTED_DIM}. "
                "Either pick a 768-dim model or migrate the Passage.embedding column."
            )
        logger.info("Local embedding model loaded (dim=%d)", dim)
        return model

    async def _ensure_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    def _encode_sync(self, model: SentenceTransformer, texts: list[str]) -> Any:
        return model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one 768-dim unit-norm vector per text."""
        if not texts:
            return []
        model = await self._ensure_model()
        arr = await asyncio.to_thread(self._encode_sync, model, texts)
        return arr.tolist()
