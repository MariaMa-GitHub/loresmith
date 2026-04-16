from google import genai

_BATCH_SIZE = 50  # Max texts per embed_content request for text-embedding-004
_MODEL = "text-embedding-004"


class Embedder:
    """Generates 768-dimensional embeddings using text-embedding-004.

    Pass api_key=None to use the GOOGLE_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = genai.Client(api_key=api_key)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.aio.models.embed_content(
            model=_MODEL,
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one 768-dim vector per text."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            all_embeddings.extend(await self._embed_batch(batch))

        return all_embeddings
