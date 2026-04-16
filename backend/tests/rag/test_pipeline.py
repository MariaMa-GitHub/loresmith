from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Hit


def _make_pipeline(*, bm25_hits, dense_hits, llm_stream):
    """Builder that keeps every test's setup explicit and short."""
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = bm25_hits

    mock_dense = MagicMock()
    mock_dense.search = AsyncMock(return_value=dense_hits)

    mock_llm = MagicMock()
    mock_llm.stream = llm_stream

    return RAGPipeline(
        embedder=mock_embedder,
        bm25_index=mock_bm25,
        dense_retriever=mock_dense,
        llm=mock_llm,
        game_slug="hades",
        game_display_name="Hades",
    )


@pytest.mark.asyncio
async def test_rag_pipeline_streams_tokens():
    async def fake_stream(messages, system=None):
        yield "Zagreus "
        yield "is the son of Hades."

    pipeline = _make_pipeline(
        bm25_hits=[BM25Hit(passage_id=1, score=1.5, content="Zagreus is the son of Hades.")],
        dense_hits=[],
        llm_stream=fake_stream,
    )

    chunks = []
    async for chunk in pipeline.stream_answer(
        session=AsyncMock(),
        question="Who is Zagreus?",
        max_spoiler_tier=0,
    ):
        chunks.append(chunk)

    assert "".join(chunks) == "Zagreus is the son of Hades."


@pytest.mark.asyncio
async def test_rag_pipeline_uses_retrieved_context():
    """Verify the prompt is built from retrieved passages."""
    captured_messages = []

    async def capturing_stream(messages, system=None):
        captured_messages.extend(messages)
        yield "answer"

    pipeline = _make_pipeline(
        bm25_hits=[BM25Hit(passage_id=42, score=2.0, content="Nyx is the Goddess of Night.")],
        dense_hits=[],
        llm_stream=capturing_stream,
    )

    async for _ in pipeline.stream_answer(
        session=AsyncMock(),
        question="Who is Nyx?",
        max_spoiler_tier=0,
    ):
        pass

    assert captured_messages
    prompt_content = captured_messages[0]["content"]
    assert "Nyx is the Goddess of Night" in prompt_content


@pytest.mark.asyncio
async def test_rag_pipeline_traces_retrieval_and_generation():
    """Pipeline must open spans via the injected tracer on every run."""
    async def fake_stream(messages, system=None):
        yield "ok"

    trace_names: list[str] = []

    class _RecordingSpan:
        def set_output(self, output): pass
        def set_metadata(self, metadata): pass

    class _RecordingTracer:
        enabled = True

        def trace(self, name, metadata=None, **kwargs):
            trace_names.append(name)
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield _RecordingSpan()

            return _cm()

    pipeline = RAGPipeline(
        embedder=AsyncMock(embed=AsyncMock(return_value=[[0.0] * 768])),
        bm25_index=MagicMock(search=MagicMock(return_value=[])),
        dense_retriever=MagicMock(search=AsyncMock(return_value=[])),
        llm=MagicMock(stream=fake_stream),
        game_slug="hades",
        game_display_name="Hades",
        tracer=_RecordingTracer(),
    )

    async for _ in pipeline.stream_answer(
        session=AsyncMock(), question="q", max_spoiler_tier=0,
    ):
        pass

    assert "rag.retrieve" in trace_names
    assert "rag.generate" in trace_names
