import json

import pytest


@pytest.mark.asyncio
async def test_games_endpoint_returns_list(client):
    response = await client.get("/games")
    assert response.status_code == 200
    body = response.json()
    assert "games" in body
    assert isinstance(body["games"], list)
    assert len(body["games"]) >= 1
    assert body["games"][0]["slug"] == "hades"


@pytest.mark.asyncio
async def test_chat_endpoint_requires_question(client):
    response = await client.post("/chat", json={"game": "hades"})
    assert response.status_code == 422  # validation error — question missing


@pytest.mark.asyncio
async def test_chat_endpoint_rejects_unknown_game(client):
    response = await client.post(
        "/chat", json={"game": "unknown-game", "question": "Who is Zagreus?"}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_endpoint_streams_sse_tokens(client, monkeypatch):
    """Inject a stubbed pipeline + fake session factory so no DB/LLM is needed."""

    class _StubPipeline:
        async def stream_answer(self, session, question, max_spoiler_tier=0, history=None):
            for tok in ["Hello ", "world"]:
                yield tok

    class _FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def execute(self, *a, **kw): return None
        async def commit(self): return None
        def add(self, *a, **kw): return None

    def _fake_session_factory():
        return _FakeSession  # returns class; event_stream calls _FakeSession() then __aenter__

    async def _fake_get_pipeline(session, game_slug):
        return _StubPipeline()

    from app import main as main_module
    monkeypatch.setattr(main_module, "_get_pipeline", _fake_get_pipeline)
    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)

    async with client.stream(
        "POST", "/chat", json={"game": "hades", "question": "q"}
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = b""
        async for chunk in response.aiter_bytes():
            body += chunk

    text = body.decode()
    # SSE events use 'data: <json>\n\n'. Both token events and the final
    # done event must be present.
    events = [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"
    streamed = "".join(e["content"] for e in events if e["type"] == "token")
    assert streamed == "Hello world"


@pytest.mark.asyncio
async def test_chat_accepts_history_field(client, monkeypatch):
    """/chat must accept a history list and begin streaming without error."""

    class _StubPipeline:
        async def stream_answer(self, session, question, max_spoiler_tier=0, history=None):
            yield "ok"

    class _FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def execute(self, *a, **kw): return None
        async def commit(self): return None
        def add(self, *a, **kw): return None

    def _fake_session_factory():
        return _FakeSession  # returns class; event_stream calls _FakeSession() then __aenter__

    async def _fake_get_pipeline(session, game_slug):
        return _StubPipeline()

    from app import main as main_module
    monkeypatch.setattr(main_module, "_get_pipeline", _fake_get_pipeline)
    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)

    async with client.stream(
        "POST",
        "/chat",
        json={
            "game": "hades",
            "question": "What about his weapons?",
            "history": [
                {"role": "user", "content": "Tell me about Zagreus."},
                {"role": "assistant", "content": "Zagreus is the son of Hades."},
            ],
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
