import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.base import LLMProvider
from app.llm.ollama import OllamaProvider


def test_ollama_satisfies_protocol():
    provider = OllamaProvider()
    assert isinstance(provider, LLMProvider)


def test_ollama_default_model():
    provider = OllamaProvider()
    assert provider.model_name == "qwen2.5:7b"


def test_ollama_custom_model():
    provider = OllamaProvider(model_name="llama3.3:8b")
    assert provider.model_name == "llama3.3:8b"


@pytest.mark.asyncio
async def test_ollama_complete_returns_text():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Zagreus is the son of Hades."}
    }

    with patch("app.llm.ollama.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        provider = OllamaProvider()
        result = await provider.complete([{"role": "user", "content": "Who is Zagreus?"}])

    assert result == "Zagreus is the son of Hades."


@pytest.mark.asyncio
async def test_ollama_complete_sends_correct_payload():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"message": {"content": "answer"}}

    with patch("app.llm.ollama.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        provider = OllamaProvider(model_name="qwen2.5:3b")
        await provider.complete([{"role": "user", "content": "hello"}], system="Be concise.")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "qwen2.5:3b"
        assert payload["stream"] is False
        # System message injected at front
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "Be concise."


@pytest.mark.asyncio
async def test_ollama_stream_yields_chunks():
    chunk1 = json.dumps({"message": {"content": "Hello"}, "done": False}).encode()
    chunk2 = json.dumps({"message": {"content": " world"}, "done": False}).encode()
    done_chunk = json.dumps({"message": {"content": ""}, "done": True}).encode()

    async def aiter(items):
        for item in items:
            yield item

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = lambda: aiter(
        [chunk1.decode(), chunk2.decode(), done_chunk.decode()]
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    with patch("app.llm.ollama.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        provider = OllamaProvider()
        chunks = []
        async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

    assert chunks == ["Hello", " world"]
