from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import rrf_fuse
from app.tracing.langfuse import noop_tracer

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)

_TOP_K_RETRIEVE = 10   # candidates from each retriever
_TOP_K_FINAL = 5       # passages passed to the LLM


@dataclass
class RAGResponse:
    answer: str
    passages: list[dict]  # [{passage_id, content, source_url}]


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
    ) -> None:
        self._embedder = embedder
        self._bm25 = bm25_index
        self._dense = dense_retriever
        self._llm = llm
        self._game_slug = game_slug
        self._game_display_name = game_display_name
        self._tracer = tracer or noop_tracer()
        self._bm25_source_map = bm25_source_map or {}

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

            bm25_hits = self._bm25.search(question, top_k=_TOP_K_RETRIEVE)
            dense_hits = await self._dense.search(
                session=session,
                game_slug=self._game_slug,
                query_embedding=query_embedding,
                top_k=_TOP_K_RETRIEVE,
                max_spoiler_tier=max_spoiler_tier,
            )

            fused = rrf_fuse(
                bm25_hits=bm25_hits,
                dense_hits=dense_hits,
                top_k=_TOP_K_FINAL,
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

    async def stream_answer(
        self,
        session: AsyncSession,
        question: str,
        max_spoiler_tier: int = 0,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Retrieve context, build prompt, and stream the LLM response."""
        passages = await self._retrieve(session, question, max_spoiler_tier)
        prompt = self._build_prompt(question, passages)
        messages = [{"role": "user", "content": prompt}]

        with self._tracer.trace(
            "rag.generate",
            metadata={"game": self._game_slug, "model": getattr(self._llm, "model_name", "")},
        ) as span:
            collected: list[str] = []
            async for chunk in self._llm.stream(messages):
                collected.append(chunk)
                yield chunk
            span.set_output("".join(collected))
