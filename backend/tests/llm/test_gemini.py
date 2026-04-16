from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.base import LLMProvider
from app.llm.gemini import GeminiProvider


def test_gemini_provider_satisfies_protocol():
    with patch("app.llm.gemini.genai.Client"):
        provider = GeminiProvider(api_key="test")
        assert isinstance(provider, LLMProvider)


def test_gemini_provider_default_model():
    with patch("app.llm.gemini.genai.Client"):
        provider = GeminiProvider(api_key="test")
        assert provider.model_name == "gemini-2.5-flash"


def test_gemini_provider_custom_model():
    with patch("app.llm.gemini.genai.Client"):
        provider = GeminiProvider(api_key="test", model_name="gemini-2.5-flash-lite")
        assert provider.model_name == "gemini-2.5-flash-lite"


@pytest.mark.asyncio
async def test_gemini_complete_returns_text():
    mock_response = MagicMock()
    mock_response.text = "Zagreus wields the Stygian Blade."

    with patch("app.llm.gemini.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        provider = GeminiProvider(api_key="test")
        result = await provider.complete([{"role": "user", "content": "What does Zagreus use?"}])

    assert result == "Zagreus wields the Stygian Blade."


@pytest.mark.asyncio
async def test_gemini_complete_with_system_prompt():
    mock_response = MagicMock()
    mock_response.text = "answer"

    with patch("app.llm.gemini.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        provider = GeminiProvider(api_key="test")
        await provider.complete(
            [{"role": "user", "content": "q"}],
            system="You are a lore expert.",
        )

        call_kwargs = mock_client.aio.models.generate_content.call_args
        assert call_kwargs is not None
        # Verify config was passed (system instruction)
        assert call_kwargs.kwargs.get("config") is not None or len(call_kwargs.args) >= 3


@pytest.mark.asyncio
async def test_gemini_stream_yields_chunks():
    # The real google-genai async client returns a coroutine that resolves to
    # an AsyncIterator — mock it the same way so the adapter's `await … ;
    # async for …` pattern is exercised.
    async def fake_async_iter():
        for text in ["chunk1", " chunk2"]:
            chunk = MagicMock()
            chunk.text = text
            yield chunk

    async def fake_stream_coro(*args, **kwargs):
        return fake_async_iter()

    with patch("app.llm.gemini.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = fake_stream_coro
        mock_client_cls.return_value = mock_client

        provider = GeminiProvider(api_key="test")
        chunks = []
        async for chunk in provider.stream([{"role": "user", "content": "tell me"}]):
            chunks.append(chunk)

    assert chunks == ["chunk1", " chunk2"]
