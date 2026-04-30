from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.ollama import OllamaProvider


def _mock_client(response_json: dict) -> tuple:
    """Return (mock_client_cls, mock_client, mock_response) wired for httpx.AsyncClient."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response_json

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client, mock_response


@pytest.mark.asyncio
async def test_complete_with_tools_sends_tools_in_payload():
    tools = [{"type": "function", "function": {"name": "entity_lookup", "parameters": {}}}]
    mock_client, _ = _mock_client({"message": {"content": "answer"}})

    with patch("app.llm.ollama.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = mock_client

        provider = OllamaProvider(model_name="qwen2.5:7b")
        await provider.complete_with_tools(
            [{"role": "user", "content": "Who is Zagreus?"}],
            tools=tools,
        )

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs.kwargs["json"]
    assert "tools" in payload
    assert payload["tools"] == tools
    assert payload["stream"] is False


@pytest.mark.asyncio
async def test_complete_with_tools_returns_tool_calls_when_present():
    tool_calls_raw = [{"function": {"name": "entity_lookup", "arguments": {"slug": "zagreus"}}}]
    # Real Ollama responses omit "content" when tool calls are present
    mock_client, _ = _mock_client({"message": {"tool_calls": tool_calls_raw}})

    with patch("app.llm.ollama.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = mock_client

        provider = OllamaProvider()
        text, calls = await provider.complete_with_tools(
            [{"role": "user", "content": "Who is Zagreus?"}],
            tools=[],
        )

    assert text is None
    assert calls == [{"name": "entity_lookup", "arguments": {"slug": "zagreus"}}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        {"content": "Zagreus is the prince of the Underworld."},           # no tool_calls key
        {"content": "Zagreus is the prince of the Underworld.", "tool_calls": None},  # null
        {"content": "Zagreus is the prince of the Underworld.", "tool_calls": []},    # empty list
    ],
)
async def test_complete_with_tools_returns_text_when_no_tool_calls(message):
    mock_client, _ = _mock_client({"message": message})

    with patch("app.llm.ollama.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = mock_client

        provider = OllamaProvider()
        text, calls = await provider.complete_with_tools(
            [{"role": "user", "content": "Who is Zagreus?"}],
            tools=[],
        )

    assert text == "Zagreus is the prince of the Underworld."
    assert calls == []
