import asyncio
import logging
import math
import random

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50  # Conservative; gemini-embedding-001 accepts more per request.
_MODEL = "gemini-embedding-001"
_OUTPUT_DIM = 768  # Matches pgvector column dim (app/db/models.py::EMBEDDING_DIM).

# Free-tier gemini-embedding-001 has tight RPM limits. Pause briefly between
# batches so we don't burst through the per-minute bucket in one go. Tests set
# this to 0 via the `inter_batch_delay_seconds` constructor arg.
_DEFAULT_INTER_BATCH_DELAY_SECONDS = 13.0

# Exponential-backoff retry for transient 429 / 503.
_MAX_RETRY_ATTEMPTS = 6
_RETRY_BASE_SECONDS = 2.0
_RETRY_MAX_SECONDS = 60.0


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. gemini-embedding-001 only returns unit-norm vectors
    at the native 3072 dim; Matryoshka-truncated outputs (dim < 3072) must be
    normalized client-side before cosine similarity is meaningful.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _is_transient(exc: BaseException) -> bool:
    """True for status codes that are worth retrying (rate limit / overload)."""
    code = getattr(exc, "code", None)
    if code in (429, 503):
        return True
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg


class Embedder:
    """Generates 768-dimensional embeddings using gemini-embedding-001.

    Pass api_key=None to use the GOOGLE_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        inter_batch_delay_seconds: float = _DEFAULT_INTER_BATCH_DELAY_SECONDS,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._inter_batch_delay = inter_batch_delay_seconds

    async def _embed_batch_once(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.aio.models.embed_content(
            model=_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=_OUTPUT_DIM),
        )
        return [_l2_normalize(list(e.values)) for e in result.embeddings]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_exc: BaseException | None = None
        for attempt in range(_MAX_RETRY_ATTEMPTS):
            try:
                return await self._embed_batch_once(texts)
            except Exception as exc:
                if not _is_transient(exc) or attempt == _MAX_RETRY_ATTEMPTS - 1:
                    raise
                last_exc = exc
                backoff = min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * (2 ** attempt))
                jitter = random.uniform(0, 1)
                wait = backoff + jitter
                logger.warning(
                    "Embedding batch rate-limited (attempt %d/%d); sleeping %.1fs",
                    attempt + 1, _MAX_RETRY_ATTEMPTS, wait,
                )
                await asyncio.sleep(wait)
        assert last_exc is not None  # pragma: no cover — unreachable
        raise last_exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one 768-dim unit-norm vector per text."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            all_embeddings.extend(await self._embed_batch(batch))
            if i + _BATCH_SIZE < len(texts) and self._inter_batch_delay > 0:
                await asyncio.sleep(self._inter_batch_delay)

        return all_embeddings
