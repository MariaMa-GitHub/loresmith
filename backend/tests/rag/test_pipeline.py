from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Hit
from app.retrieval.hybrid import HybridHit


def _make_pipeline(*, bm25_hits, dense_hits, llm_stream, rewriter=None):
    """Builder that keeps every test's setup explicit and short."""
    mock_embedder = MagicMock()
    mock_embedder.backend_name = "local"
    mock_embedder.model_name = "BAAI/bge-base-en-v1.5"
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
        rewriter=rewriter,
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

    session = AsyncMock()
    messages, _ = await pipeline.prepare_messages(session=session, question="Who is Zagreus?", max_spoiler_tier=0)
    chunks = []
    async for chunk in pipeline.stream_messages(messages):
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

    session = AsyncMock()
    messages, _ = await pipeline.prepare_messages(session=session, question="Who is Nyx?", max_spoiler_tier=0)
    async for _ in pipeline.stream_messages(messages):
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

    session = AsyncMock()
    messages, _ = await pipeline.prepare_messages(session=session, question="q", max_spoiler_tier=0)
    async for _ in pipeline.stream_messages(messages):
        pass

    assert "rag.retrieve" in trace_names
    assert "rag.generate" in trace_names


@pytest.mark.asyncio
async def test_rag_pipeline_falls_back_to_original_question_when_rewriter_fails():
    async def fake_stream(messages, system=None):
        yield "answer"

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite = AsyncMock(side_effect=RuntimeError("rewriter down"))

    pipeline = _make_pipeline(
        bm25_hits=[],
        dense_hits=[],
        llm_stream=fake_stream,
        rewriter=mock_rewriter,
    )

    session = AsyncMock()
    messages, _ = await pipeline.prepare_messages(
        session=session,
        question="Who is Zagreus?",
        max_spoiler_tier=0,
        history=[{"role": "user", "content": "Tell me about him."}],
    )
    async for _ in pipeline.stream_messages(messages):
        pass

    pipeline._bm25.search.assert_called_once_with(
        "Who is Zagreus?",
        top_k=10,
        max_spoiler_tier=0,
    )


@pytest.mark.asyncio
async def test_rag_pipeline_uses_configured_top_k(monkeypatch):
    async def fake_stream(messages, system=None):
        yield "answer"

    fused_top_ks = []

    def fake_rrf_fuse(*, bm25_hits, dense_hits, top_k, bm25_source_map=None):
        fused_top_ks.append(top_k)
        return [HybridHit(passage_id=1, rrf_score=1.0, content="ctx", source_url="https://x.com")]

    monkeypatch.setattr("app.rag.pipeline.rrf_fuse", fake_rrf_fuse)

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [BM25Hit(passage_id=1, score=1.0, content="ctx")]

    mock_dense = MagicMock()
    mock_dense.search = AsyncMock(return_value=[])

    pipeline = RAGPipeline(
        embedder=AsyncMock(
            embed=AsyncMock(return_value=[[0.0] * 768]),
            backend_name="local",
            model_name="BAAI/bge-base-en-v1.5",
        ),
        bm25_index=mock_bm25,
        dense_retriever=mock_dense,
        llm=MagicMock(stream=fake_stream),
        game_slug="hades",
        game_display_name="Hades",
        retrieve_top_k=3,
        final_top_k=2,
    )

    session = AsyncMock()
    messages, _ = await pipeline.prepare_messages(session=session, question="q", max_spoiler_tier=0)
    async for _ in pipeline.stream_messages(messages):
        pass

    mock_bm25.search.assert_called_once_with("q", top_k=3, max_spoiler_tier=0)
    mock_dense.search.assert_awaited_once()
    assert mock_dense.search.await_args.kwargs["top_k"] == 3
    assert mock_dense.search.await_args.kwargs["embedding_backend"] == "local"
    assert (
        mock_dense.search.await_args.kwargs["embedding_model"]
        == "BAAI/bge-base-en-v1.5"
    )
    assert fused_top_ks == [2]


@pytest.mark.asyncio
async def test_rag_pipeline_answer_returns_answer_and_passages():
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="Nyx is the Goddess of Night. [1]")

    pipeline = _make_pipeline(
        bm25_hits=[BM25Hit(passage_id=42, score=2.0, content="Nyx is the Goddess of Night.")],
        dense_hits=[],
        llm_stream=AsyncMock(),
    )
    pipeline._llm = mock_llm

    response = await pipeline.answer(
        session=AsyncMock(),
        question="Who is Nyx?",
        max_spoiler_tier=0,
    )

    assert response.answer == "Nyx is the Goddess of Night. [1]"
    assert response.passages[0]["passage_id"] == 42
