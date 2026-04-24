import asyncio
from collections.abc import AsyncIterator

from google import genai
from google.genai import types


def _to_gemini_contents(messages: list[dict]) -> list:
    role_map = {"user": "user", "assistant": "model"}
    return [
        {
            "role": role_map.get(m["role"], m["role"]),
            "parts": [{"text": m["content"]}],
        }
        for m in messages
    ]


class GeminiProvider:
    """LLMProvider adapter for Google Gemini via the google-genai SDK."""

    model_name: str

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        min_call_interval: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self._min_call_interval = min_call_interval
        self._client = genai.Client(api_key=api_key)

    async def _pace(self) -> None:
        if self._min_call_interval > 0:
            await asyncio.sleep(self._min_call_interval)

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> str:
        await self._pace()
        contents = _to_gemini_contents(messages)
        cfg = types.GenerateContentConfig(system_instruction=system) if system else None
        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=cfg,
        )
        return response.text

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
    ) -> tuple[str | None, list[dict]]:
        await self._pace()
        contents = _to_gemini_contents(messages)
        cfg = types.GenerateContentConfig(
            system_instruction=system if system else None,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(**t) for t in tools
            ])],
        )
        response = await self._client.aio.models.generate_content(
            model=self.model_name, contents=contents, config=cfg,
        )
        tool_calls: list[dict] = []
        text_parts: list[str] = []
        for candidate in response.candidates or []:
            for part in (candidate.content.parts or []):
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_calls.append({
                        "name": fc.name,
                        "arguments": dict(fc.args or {}),
                    })
                elif getattr(part, "text", None):
                    text_parts.append(part.text)
        text = "".join(text_parts) if not tool_calls else None
        return text, tool_calls

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        await self._pace()
        contents = _to_gemini_contents(messages)
        cfg = types.GenerateContentConfig(system_instruction=system) if system else None
        # `aio.models.generate_content_stream` is a coroutine that resolves to
        # an AsyncIterator — must be awaited before iterating, unlike the sync
        # client which returns the iterator directly.
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=cfg,
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
