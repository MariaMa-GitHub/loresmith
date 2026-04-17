import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import Settings, get_settings
from app.db.models import ChatMessage, ChatSession, Passage
from app.db.session import get_session_factory
from app.ingestion.pipeline import Embedder, make_embedder
from app.llm.gemini import GeminiProvider
from app.rag.pipeline import RAGPipeline
from app.rag.rewriter import QueryRewriter
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseRetriever
from app.tracing.langfuse import LangfuseTracer

_GAMES = [
    {"slug": "hades", "display_name": "Hades"},
    {"slug": "hades2", "display_name": "Hades II"},
]
_GAME_SLUGS = {g["slug"] for g in _GAMES}
_GAME_DISPLAY = {g["slug"]: g["display_name"] for g in _GAMES}


@dataclass
class _Services:
    settings: Settings
    tracer: LangfuseTracer
    embedder: Embedder
    dense: DenseRetriever
    llm: GeminiProvider
    fast_llm: GeminiProvider


def _build_services() -> _Services:
    settings = get_settings()
    return _Services(
        settings=settings,
        tracer=LangfuseTracer(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        ),
        embedder=make_embedder(settings),
        dense=DenseRetriever(),
        llm=GeminiProvider(api_key=settings.gemini_api_key),
        fast_llm=GeminiProvider(
            api_key=settings.gemini_api_key,
            model_name="gemini-2.5-flash-lite",
        ),
    )


async def _build_bm25(
    session, game_slug: str
) -> tuple[BM25Index, dict[int, str]]:
    """Rebuild the in-memory BM25 index and source-URL map for a game.

    Returns a (BM25Index, {passage_id: source_url}) tuple. The source map
    is threaded through rrf_fuse so BM25-only passages retain a valid
    citation URL.
    """
    result = await session.execute(
        select(Passage.id, Passage.content, Passage.source_url).where(
            Passage.game_slug == game_slug
        )
    )
    rows = result.all()
    index = BM25Index()
    source_map: dict[int, str] = {}
    if rows:
        index.build([r.id for r in rows], [r.content for r in rows])
        source_map = {r.id: r.source_url for r in rows}
    return index, source_map


async def _get_pipeline(session, game_slug: str) -> RAGPipeline:
    """Return a cached RAGPipeline for a game, building it lazily on miss.

    The lock prevents concurrent cold-start requests from building the same
    index twice (each `await` inside `_build_bm25` yields control).
    """
    pipelines: dict[str, RAGPipeline] = app.state.pipelines
    if game_slug in pipelines:
        return pipelines[game_slug]
    async with app.state.pipeline_lock:
        # Re-check inside the lock in case another coroutine built it first.
        if game_slug not in pipelines:
            svc: _Services = app.state.services
            bm25, source_map = await _build_bm25(session, game_slug)
            app.state.bm25_indexes[game_slug] = bm25
            pipelines[game_slug] = RAGPipeline(
                embedder=svc.embedder,
                bm25_index=bm25,
                dense_retriever=svc.dense,
                llm=svc.llm,
                game_slug=game_slug,
                game_display_name=_GAME_DISPLAY[game_slug],
                tracer=svc.tracer,
                bm25_source_map=source_map,
                rewriter=QueryRewriter(llm=svc.fast_llm),
            )
    return pipelines[game_slug]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.services = _build_services()
    app.state.bm25_indexes = {}
    app.state.pipelines = {}
    app.state.pipeline_lock = asyncio.Lock()
    try:
        yield
    finally:
        app.state.services.tracer.flush()


app = FastAPI(title="Loresmith API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://loresmith.vercel.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    game: str
    question: str
    spoiler_tier: int = 0
    session_id: str | None = None
    history: list[dict[str, str]] | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": app.version}


@app.get("/games")
async def list_games():
    return {"games": _GAMES}


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.game not in _GAME_SLUGS:
        raise HTTPException(status_code=404, detail=f"Game '{req.game}' not found")

    async def event_stream() -> AsyncIterator[str]:
        session_id = req.session_id or str(uuid.uuid4())
        session_factory = get_session_factory()
        collected: list[str] = []

        async with session_factory() as session:
            pipeline = await _get_pipeline(session, req.game)
            async for chunk in pipeline.stream_answer(
                session=session,
                question=req.question,
                max_spoiler_tier=req.spoiler_tier,
                history=req.history or [],
            ):
                collected.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            stmt = pg_insert(ChatSession).values(
                id=session_id,
                game_slug=req.game,
                is_logging_opted_out=False,
            ).on_conflict_do_nothing()
            await session.execute(stmt)
            # citations=[] is a W4 placeholder; passage metadata wired in Week 5.
            session.add(ChatMessage(
                session_id=session_id, role="user", content=req.question, citations=[]
            ))
            session.add(ChatMessage(
                session_id=session_id, role="assistant", content="".join(collected), citations=[]
            ))
            await session.commit()

        yield f"data: {json.dumps({'type': 'session_id', 'content': session_id})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
