import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.citations import normalize_answer_citations
from app.rag.rewriter import QueryRewriter
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import rrf_fuse
from app.tracing.langfuse import noop_tracer

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)

logger = logging.getLogger(__name__)


def _embedder_identity(embedder) -> tuple[str | None, str | None]:
    """Return (backend, model) for SQL filtering. Returns None to skip a filter."""
    backend = getattr(embedder, "backend_name", None)
    model = getattr(embedder, "model_name", None)
    if not isinstance(backend, str) or not backend:
        return None, None
    if not isinstance(model, str) or not model:
        return backend, None
    return backend, model


@dataclass
class RAGResponse:
    answer: str
    passages: list[dict]  # [{passage_id, content, source_url}]
    citations: list[dict] = field(default_factory=list)


class RAGPipeline:
    """Single-turn RAG: embed query → BM25+dense retrieve → RRF → prompt → stream."""

    def __init__(
        self,
        embedder,
        bm25_index: BM25Index,
        dense_retriever: DenseRetriever,
        llm,
        game_slug: str,
        game_display_name: str,
        tracer=None,
        bm25_source_map: dict[int, str] | None = None,
        rewriter: QueryRewriter | None = None,
        retrieve_top_k: int = 10,
        final_top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._bm25 = bm25_index
        self._dense = dense_retriever
        self._llm = llm
        self._game_slug = game_slug
        self._game_display_name = game_display_name
        self._tracer = tracer or noop_tracer()
        self._bm25_source_map = bm25_source_map or {}
        self._rewriter = rewriter
        self._retrieve_top_k = retrieve_top_k
        self._final_top_k = final_top_k

    async def _retrieve(
        self,
        session: AsyncSession,
        question: str,
        max_spoiler_tier: int,
    ) -> list[dict]:
        with self._tracer.trace(
            "rag.retrieve",
            metadata={"game": self._game_slug, "max_spoiler_tier": max_spoiler_tier},
        ) as span:
            embeddings = await self._embedder.embed([question])
            query_embedding = embeddings[0]

            bm25_hits = self._bm25.search(
                question,
                top_k=self._retrieve_top_k,
                max_spoiler_tier=max_spoiler_tier,
            )
            embedding_backend, embedding_model = _embedder_identity(self._embedder)
            dense_hits = await self._dense.search(
                session=session,
                game_slug=self._game_slug,
                query_embedding=query_embedding,
                top_k=self._retrieve_top_k,
                max_spoiler_tier=max_spoiler_tier,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
            )

            fused = rrf_fuse(
                bm25_hits=bm25_hits,
                dense_hits=dense_hits,
                top_k=self._final_top_k,
                bm25_source_map=self._bm25_source_map,
            )
            passages = [
                {
                    "passage_id": h.passage_id,
                    "content": h.content,
                    "source_url": h.source_url,
                }
                for h in fused
            ]
            span.set_output(
                {
                    "num_bm25": len(bm25_hits),
                    "num_dense": len(dense_hits),
                    "num_fused": len(passages),
                }
            )
            return passages

    def _build_prompt(self, question: str, passages: list[dict]) -> str:
        template = _JINJA_ENV.get_template("answer.j2")
        return template.render(
            game_display_name=self._game_display_name,
            passages=passages,
            question=question,
        )

    async def _effective_question(self, question: str, history: list[dict] | None = None) -> str:
        effective_question = question
        if self._rewriter and history:
            try:
                rewritten = (await self._rewriter.rewrite(question, history)).strip()
                if rewritten:
                    effective_question = rewritten
                else:
                    logger.warning(
                        "Query rewriter returned an empty question for %s; "
                        "falling back to the original input",
                        self._game_slug,
                    )
            except Exception as exc:
                logger.warning(
                    "Query rewrite failed for %s; falling back to the "
                    "original question: %s",
                    self._game_slug,
                    exc,
                )
        return effective_question

    async def prepare_messages(
        self,
        session: AsyncSession,
        question: str,
        max_spoiler_tier: int,
        history: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        effective_question = await self._effective_question(question, history)
        passages = await self._retrieve(session, effective_question, max_spoiler_tier)
        prompt = self._build_prompt(effective_question, passages)
        return [{"role": "user", "content": prompt}], passages

    async def stream_messages(self, messages: list[dict]) -> AsyncIterator[str]:
        with self._tracer.trace(
            "rag.generate",
            metadata={"game": self._game_slug, "model": getattr(self._llm, "model_name", "")},
        ) as span:
            collected: list[str] = []
            async for chunk in self._llm.stream(messages):
                collected.append(chunk)
                yield chunk
            span.set_output("".join(collected))

    async def answer(
        self,
        session: AsyncSession,
        question: str,
        max_spoiler_tier: int = 0,
        history: list[dict] | None = None,
    ) -> RAGResponse:
        messages, passages = await self.prepare_messages(
            session,
            question,
            max_spoiler_tier,
            history,
        )

        with self._tracer.trace(
            "rag.generate",
            metadata={"game": self._game_slug, "model": getattr(self._llm, "model_name", "")},
        ) as span:
            answer = await self._llm.complete(messages)
            span.set_output(answer)
        normalized = normalize_answer_citations(answer, passages=passages)
        return RAGResponse(
            answer=normalized.answer,
            passages=passages,
            citations=normalized.citations,
        )

