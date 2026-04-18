import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from app.config import Settings, get_settings
from app.db.models import ChatMessage, ChatSession, QueryLog
from app.ingestion.pipeline import IngestResult
from app.tracing.langfuse import noop_tracer

_ANON_COOKIE = "loresmith_anon_session"


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
@pytest.mark.parametrize("spoiler_tier", [-1, 4])
async def test_chat_endpoint_rejects_invalid_spoiler_tier(client, spoiler_tier):
    response = await client.post(
        "/chat",
        json={
            "game": "hades",
            "question": "Who is Zagreus?",
            "spoiler_tier": spoiler_tier,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_endpoint_streams_sse_tokens(client, monkeypatch):
    """Inject a stubbed pipeline + fake session factory so no DB/LLM is needed."""

    class _StubPipeline:
        async def prepare_messages(self, session, question, max_spoiler_tier=0, history=None):
            return (
                [{"role": "user", "content": f"Prompt for {question}"}],
                [
                    {"passage_id": 1, "source_url": "https://x.com/1", "content": "Nyx\n\nctx1"},
                    {
                        "passage_id": 2,
                        "source_url": "https://x.com/2",
                        "content": "Persephone\n\nctx2",
                    },
                ],
            )

        async def stream_messages(self, messages):
            for tok in ["Hello ", "world [2]"]:
                yield tok

    class _FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def execute(self, *a, **kw): return None
        async def commit(self): return None
        def add(self, value): self.added.append(value)

    fake_session = _FakeSession()

    def _fake_session_factory():
        return lambda: fake_session

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
        assert _ANON_COOKIE in response.headers.get("set-cookie", "")
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
    assert "answer" in types
    assert "citations" in types
    assert "session_id" in types
    assert types[-1] == "done"
    streamed = "".join(e["content"] for e in events if e["type"] == "token")
    assert streamed == "Hello world [2]"
    answer_event = next(e for e in events if e["type"] == "answer")
    assert answer_event["content"] == "Hello world [1]"
    citations_event = next(e for e in events if e["type"] == "citations")
    assert citations_event["content"] == [
        {"index": 1, "passage_id": 2, "source_url": "https://x.com/2", "title": "Persephone"},
    ]
    assert any(isinstance(item, QueryLog) for item in fake_session.added)
    assistant_message = next(
        item for item in fake_session.added
        if isinstance(item, ChatMessage) and item.role == "assistant"
    )
    assert assistant_message.content == "Hello world [1]"
    assert assistant_message.citations == [
        {"index": 1, "passage_id": 2, "source_url": "https://x.com/2", "title": "Persephone"},
    ]


@pytest.mark.asyncio
async def test_chat_accepts_history_field(client, monkeypatch):
    """/chat must accept a history list and begin streaming without error."""

    class _StubPipeline:
        async def prepare_messages(self, session, question, max_spoiler_tier=0, history=None):
            return (
                [{"role": "user", "content": "prompt"}],
                [{"passage_id": 1, "source_url": "https://x.com/1", "content": "ctx"}],
            )

        async def stream_messages(self, messages):
            yield "ok"

    class _FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def get(self, model, session_id): return None
        async def execute(self, *a, **kw): return None
        async def commit(self): return None
        def add(self, *a, **kw): return None

    def _fake_session_factory():
        return _FakeSession

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
                {
                    "role": "assistant",
                    "content": "Zagreus is the son of Hades.",
                    "citations": [
                        {
                            "index": 1,
                            "source_url": "https://x.com/1",
                            "content": "ctx",
                        }
                    ],
                },
            ],
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_chat_endpoint_dedupes_same_source_url_citations(client, monkeypatch):
    class _StubPipeline:
        async def prepare_messages(self, session, question, max_spoiler_tier=0, history=None):
            return (
                [{"role": "user", "content": "prompt"}],
                [
                    {
                        "passage_id": 1,
                        "source_url": "https://x.com/zagreus",
                        "content": "Zagreus\n\nctx1",
                    },
                    {
                        "passage_id": 2,
                        "source_url": "https://x.com/zagreus",
                        "content": "Zagreus\n\nctx2",
                    },
                ],
            )

        async def stream_messages(self, messages):
            for tok in ["Zagreus ", "[1][2]"]:
                yield tok

    class _FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def execute(self, *a, **kw): return None
        async def commit(self): return None
        def add(self, value): self.added.append(value)

    fake_session = _FakeSession()

    def _fake_session_factory():
        return lambda: fake_session

    async def _fake_get_pipeline(session, game_slug):
        return _StubPipeline()

    from app import main as main_module

    monkeypatch.setattr(main_module, "_get_pipeline", _fake_get_pipeline)
    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)

    async with client.stream(
        "POST", "/chat", json={"game": "hades", "question": "q"}
    ) as response:
        assert response.status_code == 200
        body = b""
        async for chunk in response.aiter_bytes():
            body += chunk

    events = [
        json.loads(line[len("data: "):])
        for line in body.decode().splitlines()
        if line.startswith("data: ")
    ]
    answer_event = next(e for e in events if e["type"] == "answer")
    assert answer_event["content"] == "Zagreus [1]"
    citations_event = next(e for e in events if e["type"] == "citations")
    assert citations_event["content"] == [
        {"index": 1, "passage_id": 1, "source_url": "https://x.com/zagreus", "title": "Zagreus"},
    ]
    assistant_message = next(
        item for item in fake_session.added
        if isinstance(item, ChatMessage) and item.role == "assistant"
    )
    assert assistant_message.content == "Zagreus [1]"
    assert assistant_message.citations == [
        {"index": 1, "passage_id": 1, "source_url": "https://x.com/zagreus", "title": "Zagreus"},
    ]


@pytest.mark.asyncio
async def test_chat_endpoint_round_trips_session_id(client, monkeypatch):
    class _StubPipeline:
        async def prepare_messages(self, session, question, max_spoiler_tier=0, history=None):
            return (
                [{"role": "user", "content": "prompt"}],
                [{"passage_id": 1, "source_url": "https://x.com/1", "content": "ctx"}],
            )

        async def stream_messages(self, messages):
            yield "ok"

    class _FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def get(self, model, session_id):
            return ChatSession(
                id=session_id,
                owner_token="owner-123",
                game_slug="hades",
                is_logging_opted_out=False,
            )
        async def execute(self, *a, **kw):
            return SimpleNamespace(all=lambda: [])
        async def commit(self): return None
        def add(self, *a, **kw): return None

    def _fake_session_factory():
        return _FakeSession

    async def _fake_get_pipeline(session, game_slug):
        return _StubPipeline()

    from app import main as main_module
    monkeypatch.setattr(main_module, "_get_pipeline", _fake_get_pipeline)
    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)
    client.cookies.set(_ANON_COOKIE, "owner-123")

    async with client.stream(
        "POST",
        "/chat",
        json={"game": "hades", "question": "q", "session_id": "session-123"},
    ) as response:
        assert response.status_code == 200
        body = b""
        async for chunk in response.aiter_bytes():
            body += chunk

    text = body.decode()
    events = [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    session_event = next(e for e in events if e["type"] == "session_id")
    assert session_event["content"] == "session-123"


@pytest.mark.asyncio
async def test_chat_endpoint_rejects_cross_game_session_id(client, monkeypatch):
    class _ValidationSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, model, session_id):
            return ChatSession(
                id=session_id,
                owner_token="owner-123",
                game_slug="hades2",
                is_logging_opted_out=False,
            )

    def _fake_session_factory():
        return _ValidationSession

    from app import main as main_module

    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)
    client.cookies.set(_ANON_COOKIE, "owner-123")

    response = await client.post(
        "/chat",
        json={"game": "hades", "question": "q", "session_id": "shared-session"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Session belongs to a different game."


@pytest.mark.asyncio
async def test_chat_endpoint_touches_session_timestamp_on_reuse(client, monkeypatch):
    class _StubPipeline:
        async def prepare_messages(
            self,
            session,
            question,
            max_spoiler_tier=0,
            history=None,
        ):
            return (
                [{"role": "user", "content": "prompt"}],
                [{"passage_id": 1, "source_url": "https://x.com/1", "content": "ctx"}],
            )

        async def stream_messages(self, messages):
            yield "ok"

    class _FakeSession:
        def __init__(self):
            self.executed = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, model, session_id):
            return ChatSession(
                id=session_id,
                owner_token="owner-123",
                game_slug="hades",
                is_logging_opted_out=False,
            )

        async def execute(self, statement, *args, **kwargs):
            self.executed.append(statement)
            return SimpleNamespace(all=lambda: [])

        async def commit(self):
            return None

        def add(self, *args, **kwargs):
            return None

    fake_session = _FakeSession()

    def _fake_session_factory():
        return lambda: fake_session

    async def _fake_get_pipeline(session, game_slug):
        return _StubPipeline()

    from app import main as main_module

    monkeypatch.setattr(main_module, "_get_pipeline", _fake_get_pipeline)
    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)
    client.cookies.set(_ANON_COOKIE, "owner-123")

    async with client.stream(
        "POST",
        "/chat",
        json={"game": "hades", "question": "q", "session_id": "session-123"},
    ) as response:
        assert response.status_code == 200
        async for _ in response.aiter_bytes():
            pass

    upsert_stmt = next(
        stmt
        for stmt in fake_session.executed
        if "ON CONFLICT" in str(stmt.compile(dialect=postgresql.dialect()))
    )
    sql = str(upsert_stmt.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql
    assert "updated_at" in sql


@pytest.mark.asyncio
async def test_sessions_endpoint_sets_anon_cookie_when_missing(client, monkeypatch):
    class _EmptySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, *args, **kwargs):
            result = SimpleNamespace()
            result.scalars = lambda: SimpleNamespace(all=lambda: [])
            return result

    def _fake_session_factory():
        return _EmptySession

    from app import main as main_module

    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)

    response = await client.get("/sessions/hades")

    assert response.status_code == 200
    assert response.json() == {"sessions": []}
    assert _ANON_COOKIE in response.headers.get("set-cookie", "")


@pytest.mark.asyncio
async def test_sessions_endpoint_filters_to_cookie_owner(client, monkeypatch):
    client.cookies.set(_ANON_COOKIE, "owner-123")
    executed_sql = []

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, statement, *args, **kwargs):
            sql = str(statement.compile(dialect=postgresql.dialect()))
            executed_sql.append(sql)
            if "FROM chat_sessions" in sql:
                rows = [
                    ChatSession(
                        id="session-1",
                        owner_token="owner-123",
                        game_slug="hades",
                        is_logging_opted_out=False,
                    )
                ]
                return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))
            messages = [
                ChatMessage(
                    session_id="session-1",
                    role="user",
                    content="Who is Nyx?",
                    citations=[],
                ),
            ]
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: messages))

    def _fake_session_factory():
        return _FakeSession

    from app import main as main_module

    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)

    response = await client.get("/sessions/hades")

    assert response.status_code == 200
    assert response.json()["sessions"][0]["id"] == "session-1"
    assert any("owner_token" in sql for sql in executed_sql)


@pytest.mark.asyncio
async def test_session_messages_endpoint_requires_matching_owner_cookie(client, monkeypatch):
    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, model, session_id):
            return ChatSession(
                id=session_id,
                owner_token="owner-123",
                game_slug="hades",
                is_logging_opted_out=False,
            )

    def _fake_session_factory():
        return _FakeSession

    from app import main as main_module

    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)

    response = await client.get("/sessions/session-1/messages")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_endpoint_uses_persisted_history_for_existing_session(client, monkeypatch):
    captured_history = None

    class _StubPipeline:
        async def prepare_messages(self, session, question, max_spoiler_tier=0, history=None):
            nonlocal captured_history
            captured_history = history
            return (
                [{"role": "user", "content": "prompt"}],
                [{"passage_id": 1, "source_url": "https://x.com/1", "content": "ctx"}],
            )

        async def stream_messages(self, messages):
            yield "ok"

    history_rows = [
        SimpleNamespace(role="user", content="Tell me about Nyx."),
        SimpleNamespace(role="assistant", content="Nyx is the Goddess of Night. [1]"),
        SimpleNamespace(role="user", content="This failed turn should not reach the rewriter."),
    ]

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, model, session_id):
            return ChatSession(
                id=session_id,
                owner_token="owner-123",
                game_slug="hades",
                is_logging_opted_out=False,
            )

        async def execute(self, *args, **kwargs):
            return SimpleNamespace(all=lambda: history_rows)

        async def commit(self):
            return None

        def add(self, *args, **kwargs):
            return None

    def _fake_session_factory():
        return _FakeSession

    async def _fake_get_pipeline(session, game_slug):
        return _StubPipeline()

    from app import main as main_module

    monkeypatch.setattr(main_module, "_get_pipeline", _fake_get_pipeline)
    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)
    client.cookies.set(_ANON_COOKIE, "owner-123")

    async with client.stream(
        "POST",
        "/chat",
        json={
            "game": "hades",
            "question": "What is her title?",
            "session_id": "session-123",
            "history": [{"role": "user", "content": "Malicious client history"}],
        },
    ) as response:
        assert response.status_code == 200
        async for _ in response.aiter_bytes():
            pass

    assert captured_history == [
        {"role": "user", "content": "Tell me about Nyx."},
        {"role": "assistant", "content": "Nyx is the Goddess of Night. [1]"},
    ]


@pytest.mark.asyncio
async def test_chat_endpoint_does_not_persist_partial_assistant_on_stream_error(
    client, monkeypatch,
):
    class _StubPipeline:
        async def prepare_messages(self, session, question, max_spoiler_tier=0, history=None):
            return (
                [{"role": "user", "content": "prompt"}],
                [{"passage_id": 1, "source_url": "https://x.com/1", "content": "ctx"}],
            )

        async def stream_messages(self, messages):
            yield "partial "
            raise RuntimeError("boom")

    class _FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, *args, **kwargs):
            return None

        async def commit(self):
            return None

        def add(self, value):
            self.added.append(value)

    fake_session = _FakeSession()

    def _fake_session_factory():
        return lambda: fake_session

    async def _fake_get_pipeline(session, game_slug):
        return _StubPipeline()

    from app import main as main_module

    monkeypatch.setattr(main_module, "_get_pipeline", _fake_get_pipeline)
    monkeypatch.setattr(main_module, "get_session_factory", _fake_session_factory)

    async with client.stream(
        "POST", "/chat", json={"game": "hades", "question": "q"}
    ) as response:
        assert response.status_code == 200
        async for _ in response.aiter_bytes():
            pass

    assert any(isinstance(item, QueryLog) for item in fake_session.added)
    assert any(
        isinstance(item, ChatMessage) and item.role == "user"
        for item in fake_session.added
    )
    assert not any(
        isinstance(item, ChatMessage) and item.role == "assistant"
        for item in fake_session.added
    )


# ---------------------------------------------------------------------------
# /ingest endpoint tests
# ---------------------------------------------------------------------------

def _real_token_settings():
    return Settings(
        database_url="postgresql+asyncpg://u:p@h/db",
        ingest_token="real-secret",
        _env_file=None,
    )


def _change_me_settings():
    return Settings(
        database_url="postgresql+asyncpg://u:p@h/db",
        ingest_token="change-me",
        _env_file=None,
    )


@pytest.mark.asyncio
async def test_ingest_returns_503_when_token_is_default_change_me(client):
    """The default 'change-me' token must disable the endpoint."""
    from app import main as main_module

    main_module.app.dependency_overrides[get_settings] = _change_me_settings
    try:
        response = await client.post(
            "/ingest",
            json={"game": "hades"},
            headers={"Authorization": "Bearer anything"},
        )
        assert response.status_code == 503
    finally:
        main_module.app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_returns_401_when_authorization_header_is_missing(client):
    from app import main as main_module

    main_module.app.dependency_overrides[get_settings] = _real_token_settings
    try:
        response = await client.post("/ingest", json={"game": "hades"})
        assert response.status_code == 401
    finally:
        main_module.app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_returns_403_on_wrong_token(client):
    from app import main as main_module

    main_module.app.dependency_overrides[get_settings] = _real_token_settings
    try:
        response = await client.post(
            "/ingest",
            json={"game": "hades"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 403
    finally:
        main_module.app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_returns_404_for_unknown_game(client):
    from app import main as main_module

    main_module.app.dependency_overrides[get_settings] = _real_token_settings
    try:
        response = await client.post(
            "/ingest",
            json={"game": "unknown-game"},
            headers={"Authorization": "Bearer real-secret"},
        )
        assert response.status_code == 404
    finally:
        main_module.app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_returns_202_and_result_on_success(client, monkeypatch):
    from app import main as main_module

    fake_result = IngestResult(
        game_slug="hades",
        pages_fetched=5,
        chunks_created=20,
        passages_upserted=18,
        passages_skipped=2,
    )

    class _FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    main_module.app.dependency_overrides[get_settings] = _real_token_settings
    main_module.app.state.services = SimpleNamespace(
        settings=_real_token_settings(),
        router=SimpleNamespace(for_task=lambda t: None),
        tracer=noop_tracer(),
    )
    main_module.app.state.pipeline_cache = {}
    monkeypatch.setattr(main_module, "run_ingestion", AsyncMock(return_value=fake_result))
    monkeypatch.setattr(main_module, "make_embedder", lambda s: None)
    monkeypatch.setattr(main_module, "SpoilerTagger", lambda **kw: None)
    monkeypatch.setattr(main_module, "Scraper", lambda **kw: None)
    monkeypatch.setattr(main_module, "get_session_factory", lambda: _FakeSession)
    try:
        response = await client.post(
            "/ingest",
            json={"game": "hades"},
            headers={"Authorization": "Bearer real-secret"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["game_slug"] == "hades"
        assert body["pages_fetched"] == 5
        assert body["chunks_created"] == 20
        assert body["passages_upserted"] == 18
        assert body["passages_skipped"] == 2
        assert body["dry_run"] is False
    finally:
        main_module.app.dependency_overrides.clear()
