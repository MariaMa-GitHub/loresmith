import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.auth import (
    get_anon_owner_token,
    issue_anon_owner_token,
    set_anon_owner_cookie,
)
from app.config import Settings, get_settings
from app.db.models import ChatMessage, ChatSession, QueryLog
from app.db.session import get_session_factory
from app.games import ADAPTERS, GAME_DISPLAY, GAME_SLUGS, GAMES
from app.ingestion.pipeline import make_embedder, run_ingestion
from app.ingestion.scraper import Scraper
from app.ingestion.spoiler_tagger import SpoilerTagger
from app.llm.base import TaskType
from app.rag.citations import normalize_answer_citations
from app.rag.pipeline import RAGPipeline
from app.rag.rewriter import QueryRewriter
from app.services import (
    CorpusRevision,
    Services,
    build_bm25,
    build_services,
    get_corpus_revision,
)

logger = logging.getLogger(__name__)


@dataclass
class _PipelineCacheEntry:
    revision: CorpusRevision
    pipeline: RAGPipeline


async def _get_pipeline(session, game_slug: str) -> RAGPipeline:
    """Return a cached RAGPipeline for a game, building it lazily on miss.

    The lock prevents concurrent requests from rebuilding the same game's
    index twice when the cache is cold or when a re-ingest changed the corpus.
    """
    revision = await get_corpus_revision(session, game_slug)
    cache: dict[str, _PipelineCacheEntry] = app.state.pipeline_cache
    entry = cache.get(game_slug)
    if entry and entry.revision == revision:
        return entry.pipeline
    async with app.state.pipeline_lock:
        # Re-check inside the lock in case another coroutine refreshed it first.
        entry = cache.get(game_slug)
        if entry and entry.revision == revision:
            return entry.pipeline

        svc: Services = app.state.services
        bm25, source_map = await build_bm25(session, game_slug)
        cache[game_slug] = _PipelineCacheEntry(
            revision=revision,
            pipeline=RAGPipeline(
                embedder=svc.embedder,
                bm25_index=bm25,
                dense_retriever=svc.dense,
                llm=svc.router.for_task(TaskType.ANSWER),
                game_slug=game_slug,
                game_display_name=GAME_DISPLAY[game_slug],
                tracer=svc.tracer,
                bm25_source_map=source_map,
                rewriter=QueryRewriter(
                    llm=svc.router.for_task(TaskType.REWRITE),
                    tracer=svc.tracer,
                ),
                retrieve_top_k=svc.settings.retrieval_top_k_per_method,
                final_top_k=svc.settings.retrieval_top_k_final,
            ),
        )
    return cache[game_slug].pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.services = build_services()
    app.state.pipeline_cache = {}
    app.state.pipeline_lock = asyncio.Lock()
    try:
        yield
    finally:
        app.state.services.tracer.flush()


app = FastAPI(title="Loresmith API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    game: str
    question: str
    spoiler_tier: int = Field(default=0, ge=0, le=3)
    session_id: str | None = None

    class HistoryMessage(BaseModel):
        role: str
        content: str

        model_config = ConfigDict(extra="ignore")

    history: list[HistoryMessage] | None = None


class IngestRequest(BaseModel):
    game: str
    dry_run: bool = False
    cache_ttl_hours: float = 24.0
    refresh_cache: bool = False


def _verify_ingest_token(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Enforce a shared-secret bearer token on ingest endpoints.

    The default token value ``change-me`` is rejected at runtime so a
    misconfigured deployment fails loudly instead of silently accepting
    anonymous ingests.
    """
    if not settings.ingest_token or settings.ingest_token == "change-me":
        raise HTTPException(
            status_code=503,
            detail="Ingest endpoint is disabled: set INGEST_TOKEN to a real secret.",
        )
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.ingest_token:
        raise HTTPException(status_code=403, detail="Invalid ingest token")


async def _ensure_session_matches_game(
    session_id: str,
    game: str,
    owner_token: str | None,
) -> ChatSession:
    """Reject inaccessible or cross-game session reuse before streaming."""
    if owner_token is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session_factory = get_session_factory()
    async with session_factory() as session:
        existing = await session.get(ChatSession, session_id)
    if (
        existing is None
        or existing.owner_token is None
        or existing.owner_token != owner_token
    ):
        raise HTTPException(status_code=404, detail="Session not found")
    if existing.game_slug != game:
        raise HTTPException(
            status_code=409,
            detail="Session belongs to a different game.",
        )
    return existing


async def _load_rewrite_history(session, session_id: str) -> list[dict[str, str]]:
    """Return canonical persisted history for rewrite prompts.

    If the most recent turn is an unanswered user message from a failed stream,
    drop it so the rewriter only sees completed turns.
    """
    result = await session.execute(
        select(ChatMessage.role, ChatMessage.content)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at, ChatMessage.id)
    )
    history = [{"role": row.role, "content": row.content} for row in result.all()]
    if history and history[-1]["role"] == "user":
        return history[:-1]
    return history


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": app.version}


@app.get("/games")
async def list_games():
    return {"games": GAMES}


@app.get("/sessions/{game}")
async def list_sessions(
    game: str,
    request: Request,
    response: Response,
    limit: int = 20,
    settings: Settings = Depends(get_settings),
):
    """Return the most-recently updated chat sessions for a game.

    Each row carries the session id, updated_at timestamp, and a short
    preview drawn from the first user message so the UI can render a
    history sidebar without a follow-up round-trip.
    """
    if game not in GAME_SLUGS:
        raise HTTPException(status_code=404, detail=f"Game '{game}' not found")
    limit = max(1, min(limit, 100))
    owner_token = get_anon_owner_token(request, settings)
    if owner_token is None:
        owner_token = issue_anon_owner_token()
        set_anon_owner_cookie(response, owner_token, settings)
        return {"sessions": []}

    session_factory = get_session_factory()
    async with session_factory() as session:
        sessions_result = await session.execute(
            select(ChatSession)
            .where(ChatSession.game_slug == game)
            .where(ChatSession.owner_token == owner_token)
            .order_by(desc(ChatSession.updated_at))
            .limit(limit)
        )
        rows = list(sessions_result.scalars().all())
        if not rows:
            return {"sessions": []}

        session_ids = [row.id for row in rows]
        message_result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id.in_(session_ids))
            .where(ChatMessage.role == "user")
            .order_by(ChatMessage.session_id, ChatMessage.created_at)
        )
        previews: dict[str, str] = {}
        for message in message_result.scalars().all():
            previews.setdefault(message.session_id, message.content)

    def _trim(text: str, max_chars: int = 80) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 1].rstrip() + "…"

    return {
        "sessions": [
            {
                "id": row.id,
                "game_slug": row.game_slug,
                "preview": _trim(previews.get(row.id, "")),
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]
    }


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Return the full message history for a single chat session.

    Used by the frontend when the user selects a session from the history
    sidebar and wants to resume or review it.
    """
    owner_token = get_anon_owner_token(request, settings)
    if owner_token is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session_factory = get_session_factory()
    async with session_factory() as session:
        session_row = await session.get(ChatSession, session_id)
        if (
            session_row is None
            or session_row.owner_token is None
            or session_row.owner_token != owner_token
        ):
            raise HTTPException(status_code=404, detail="Session not found")
        messages_result = await session.execute(
            select(
                ChatMessage.role,
                ChatMessage.content,
                ChatMessage.citations,
                ChatMessage.created_at,
            )
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        )
        messages = []
        for row in messages_result.all():
            if row.role == "assistant":
                normalized = normalize_answer_citations(
                    row.content,
                    citations=row.citations,
                )
            else:
                normalized = normalize_answer_citations(row.content)
            messages.append(
                {
                    "role": row.role,
                    "content": normalized.answer,
                    "citations": normalized.citations,
                    "created_at": row.created_at.isoformat()
                    if row.created_at
                    else None,
                }
            )
    return {
        "session_id": session_id,
        "game_slug": session_row.game_slug,
        "messages": messages,
    }


@app.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    if req.game not in GAME_SLUGS:
        raise HTTPException(status_code=404, detail=f"Game '{req.game}' not found")
    owner_token = get_anon_owner_token(request, settings)
    issued_owner_token = False
    if req.session_id is not None:
        existing = await _ensure_session_matches_game(req.session_id, req.game, owner_token)
        owner_token = existing.owner_token
    elif owner_token is None:
        owner_token = issue_anon_owner_token()
        issued_owner_token = True

    async def event_stream() -> AsyncIterator[str]:
        session_id = req.session_id or str(uuid.uuid4())
        session_factory = get_session_factory()
        collected: list[str] = []

        async with session_factory() as session:
            history = [
                {"role": message.role, "content": message.content}
                for message in (req.history or [])
            ]
            if req.session_id is not None:
                history = await _load_rewrite_history(session, req.session_id)

            pipeline = await _get_pipeline(session, req.game)
            messages, passages = await pipeline.prepare_messages(
                session=session,
                question=req.question,
                max_spoiler_tier=req.spoiler_tier,
                history=history,
            )

            # Persist the user turn + query log up-front so mid-stream
            # failures don't drop the record of the question having been
            # asked. The assistant row is written after the stream
            # completes so it carries the final content.
            upsert_stmt = pg_insert(ChatSession).values(
                id=session_id,
                owner_token=owner_token,
                game_slug=req.game,
                is_logging_opted_out=False,
            ).on_conflict_do_update(
                index_elements=[ChatSession.id],
                set_={"updated_at": func.now()},
            )
            await session.execute(upsert_stmt)
            session.add(QueryLog(
                game_slug=req.game,
                query_text=req.question,
                session_id=session_id,
            ))
            session.add(ChatMessage(
                session_id=session_id, role="user", content=req.question, citations=[]
            ))
            await session.commit()

            stream_failed = False
            try:
                async for chunk in pipeline.stream_messages(messages):
                    collected.append(chunk)
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            except Exception as exc:
                stream_failed = True
                logger.exception("Chat stream failed for session %s: %s", session_id, exc)
                yield (
                    f"data: {json.dumps({'type': 'error', 'content': 'stream_failed'})}\n\n"
                )

            if not stream_failed:
                assistant_text = "".join(collected)
                normalized = normalize_answer_citations(
                    assistant_text,
                    passages=passages,
                )
                yield f"data: {json.dumps({'type': 'answer', 'content': normalized.answer})}\n\n"
                citations_event = {
                    "type": "citations",
                    "content": normalized.citations,
                }
                yield f"data: {json.dumps(citations_event)}\n\n"

                session.add(ChatMessage(
                    session_id=session_id,
                    role="assistant",
                    content=normalized.answer,
                    citations=normalized.citations,
                ))
                await session.commit()

            yield f"data: {json.dumps({'type': 'session_id', 'content': session_id})}\n\n"
            if stream_failed:
                yield f"data: {json.dumps({'type': 'done', 'status': 'error'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    if issued_owner_token and owner_token is not None:
        set_anon_owner_cookie(response, owner_token, settings)
    return response


@app.post("/ingest", status_code=202, dependencies=[Depends(_verify_ingest_token)])
async def ingest(req: IngestRequest):
    """Kick off an ingestion run for a registered game.

    Protected by ``INGEST_TOKEN`` (``Authorization: Bearer <token>``). The
    default placeholder ``change-me`` is rejected so this endpoint is
    inert until the operator sets a real secret.
    """
    if req.game not in ADAPTERS:
        raise HTTPException(status_code=404, detail=f"Game '{req.game}' not found")
    adapter = ADAPTERS[req.game]
    svc: Services = app.state.services

    scraper = Scraper(
        cache_dir=Path(".scraper_cache") / req.game,
        crawl_delay=adapter.sources[0].crawl_delay,
        cache_ttl=(
            None
            if req.cache_ttl_hours <= 0
            else timedelta(hours=req.cache_ttl_hours)
        ),
        force_refresh=req.refresh_cache,
    )
    embedder = make_embedder(svc.settings)
    tagger = SpoilerTagger(llm=svc.router.for_task(TaskType.TAG), tracer=svc.tracer)

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await run_ingestion(
            adapter=adapter,
            scraper=scraper,
            chunker=adapter.chunker,
            embedder=embedder,
            session=session,
            dry_run=req.dry_run,
            spoiler_tagger=tagger,
        )

    # Invalidate the pipeline cache so the next /chat rebuilds against the
    # fresh corpus revision.
    app.state.pipeline_cache.pop(req.game, None)

    return {
        "game_slug": result.game_slug,
        "pages_fetched": result.pages_fetched,
        "chunks_created": result.chunks_created,
        "passages_upserted": result.passages_upserted,
        "passages_skipped": result.passages_skipped,
        "dry_run": req.dry_run,
        "started_at": datetime.now(UTC).isoformat(),
    }
