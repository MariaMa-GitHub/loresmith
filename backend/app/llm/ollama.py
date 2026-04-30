import json
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel


class OllamaProvider:
    """LLMProvider adapter for local Ollama models via its HTTP API."""

    model_name: str

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")

    def _build_messages(self, messages: list[dict], system: str | None) -> list[dict]:
        if system:
            return [{"role": "system", "content": system}, *messages]
        return messages

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(messages, system),
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]

    async def complete_json(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        system: str | None = None,
    ) -> BaseModel:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(messages, system),
            "stream": False,
            "format": schema.model_json_schema(),
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        return schema.model_validate_json(content)

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
    ) -> tuple[str | None, list[dict]]:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(messages, system),
            "tools": tools,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            msg = response.json()["message"]
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            return (
                None,
                [
                    {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                    for call in tool_calls
                ],
            )
        return (msg.get("content"), [])

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(messages, system),
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        break
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
