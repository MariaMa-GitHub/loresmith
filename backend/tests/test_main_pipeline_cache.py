import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import main as main_module
from app import services as services_module


class _FakeRouter:
    def for_task(self, task):
        return f"llm-{task}"


class _FakePipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_query_rewriter(llm, tracer=None):
    return {"llm": llm, "tracer": tracer}


def _install_fake_app_state():
    main_module.app.state.services = services_module.Services(
        settings=SimpleNamespace(
            retrieval_top_k_per_method=10,
            retrieval_top_k_final=5,
            rerank_candidates=20,
            tool_loop_max_iters=3,
        ),
        tracer="tracer",
        embedder="embedder",
        dense="dense",
        router=_FakeRouter(),
        reranker="reranker",
        semantic_cache="cache",
        verifier="verifier",
        tool_dispatcher_factory=lambda slug, allowed_types: "dispatcher",
    )
    main_module.app.state.pipeline_cache = {}
    main_module.app.state.pipeline_lock = asyncio.Lock()


@pytest.mark.asyncio
async def test_get_pipeline_reuses_cache_when_corpus_is_unchanged(monkeypatch):
    _install_fake_app_state()

    revision = services_module.CorpusRevision(
        passage_count=1,
        latest_updated_at=None,
        max_passage_id=1,
    )
    build_bm25 = AsyncMock(return_value=("bm25", {1: "https://x.com"}))

    monkeypatch.setattr(main_module, "get_corpus_revision", AsyncMock(return_value=revision))
    monkeypatch.setattr(main_module, "build_bm25", build_bm25)
    monkeypatch.setattr(main_module, "RAGPipeline", _FakePipeline)
    monkeypatch.setattr(main_module, "QueryRewriter", _fake_query_rewriter)

    pipeline_one = await main_module._get_pipeline(AsyncMock(), "hades")
    pipeline_two = await main_module._get_pipeline(AsyncMock(), "hades")

    assert pipeline_one is pipeline_two
    assert build_bm25.await_count == 1


@pytest.mark.asyncio
async def test_get_pipeline_rebuilds_when_corpus_revision_changes(monkeypatch):
    _install_fake_app_state()

    revision_one = services_module.CorpusRevision(
        passage_count=1,
        latest_updated_at=None,
        max_passage_id=1,
    )
    revision_two = services_module.CorpusRevision(
        passage_count=2,
        latest_updated_at=None,
        max_passage_id=2,
    )
    build_bm25 = AsyncMock(
        side_effect=[
            ("bm25-v1", {1: "https://x.com/1"}),
            ("bm25-v2", {2: "https://x.com/2"}),
        ]
    )

    monkeypatch.setattr(
        main_module,
        "get_corpus_revision",
        AsyncMock(side_effect=[revision_one, revision_two]),
    )
    monkeypatch.setattr(main_module, "build_bm25", build_bm25)
    monkeypatch.setattr(main_module, "RAGPipeline", _FakePipeline)
    monkeypatch.setattr(main_module, "QueryRewriter", _fake_query_rewriter)

    pipeline_one = await main_module._get_pipeline(AsyncMock(), "hades")
    pipeline_two = await main_module._get_pipeline(AsyncMock(), "hades")

    assert pipeline_one is not pipeline_two
    assert build_bm25.await_count == 2
