from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from app.llm.base import LLMProvider


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

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> str:
        contents = _to_gemini_contents(messages)
        cfg = types.GenerateContentConfig(system_instruction=system) if system else None
        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=cfg,
        )
        return response.text

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        contents = _to_gemini_contents(messages)
        cfg = types.GenerateContentConfig(system_instruction=system) if system else None
        async for chunk in self._client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=cfg,
        ):
            if chunk.text:
                yield chunk.text


# Satisfy Protocol at type-check time
_: LLMProvider = GeminiProvider.__new__(GeminiProvider)  # type: ignore[assignment]
