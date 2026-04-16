from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.embedder import Embedder


@pytest.mark.asyncio
async def test_embedder_returns_list_of_vectors():
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]

    with patch("app.ingestion.embedder.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)
        mock_client_cls.return_value = mock_client

        embedder = Embedder(api_key="test-key")
        results = await embedder.embed(["text one"])

    assert len(results) == 1
    assert len(results[0]) == 768


@pytest.mark.asyncio
async def test_embedder_batch_returns_one_vector_per_text():
    # Native batching: 3 texts → 1 API call, 3 embeddings returned
    mock_embeddings = [MagicMock(values=[0.0] * 768) for _ in range(3)]
    mock_result = MagicMock()
    mock_result.embeddings = mock_embeddings

    with patch("app.ingestion.embedder.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_embed = AsyncMock(return_value=mock_result)
        mock_client.aio.models.embed_content = mock_embed
        mock_client_cls.return_value = mock_client

        embedder = Embedder(api_key="test-key")
        results = await embedder.embed(["text one", "text two", "text three"])

    assert len(results) == 3
    assert mock_embed.call_count == 1  # one batch call, not one-per-text


@pytest.mark.asyncio
async def test_embedder_empty_list_returns_empty():
    with patch("app.ingestion.embedder.genai.Client"):
        embedder = Embedder(api_key="test-key")
        results = await embedder.embed([])
    assert results == []


def test_embedder_embedding_dim():
    from app.db.models import EMBEDDING_DIM
    assert EMBEDDING_DIM == 768


@pytest.mark.asyncio
async def test_embedder_splits_into_batches_at_boundary():
    """51 texts → 2 API calls (batch of 50, then batch of 1)."""
    def make_result(n):
        result = MagicMock()
        result.embeddings = [MagicMock(values=[0.0] * 768) for _ in range(n)]
        return result

    with patch("app.ingestion.embedder.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_embed = AsyncMock(side_effect=[make_result(50), make_result(1)])
        mock_client.aio.models.embed_content = mock_embed
        mock_client_cls.return_value = mock_client

        embedder = Embedder(api_key="test-key", inter_batch_delay_seconds=0)
        results = await embedder.embed([f"text {i}" for i in range(51)])

    assert len(results) == 51
    assert mock_embed.call_count == 2
