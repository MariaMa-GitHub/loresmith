from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from sqlalchemy import delete, text

from app.db.models import Base, EvalRun, Passage
from app.db.session import get_engine, get_session_factory
from app.eval.runner import run_pipeline_eval
from app.llm.base import TaskType
from app.tracing.langfuse import noop_tracer


class _FakeAnswerProvider:
    model_name = "fake-answer"

    async def complete(self, messages, system=None):
        prompt = messages[0]["content"]
        if "Nyx" in prompt or "Goddess of Night" in prompt:
            return "Nyx is the Goddess of Night. [1]"
        return "Insufficient evidence. [1]"

    async def stream(self, messages, system=None):
        yield "unused"


class _FakeRewriteProvider:
    model_name = "fake-rewrite"

    async def complete(self, messages, system=None):
        return "Who is Nyx?"

    async def stream(self, messages, system=None):
        yield "unused"


class _FakeVerifyProvider:
    model_name = "fake-verify"

    async def complete(self, messages, system=None):
        return '{"faithful": true, "answer_correct": true, "refusal_appropriate": null}'

    async def stream(self, messages, system=None):
        yield "unused"


class _FakeRouter:
    def __init__(self):
        self.answer = _FakeAnswerProvider()
        self.rewrite = _FakeRewriteProvider()
        self.verify = _FakeVerifyProvider()

    def for_task(self, task: TaskType):
        if task == TaskType.ANSWER:
            return self.answer
        if task == TaskType.REWRITE:
            return self.rewrite
        return self.verify


class _FakeEmbedder:
    async def embed(self, texts):
        return [[0.0] * 768 for _ in texts]


class _FakeDenseRetriever:
    async def search(self, **kwargs):
        return []


@pytest.mark.asyncio
async def test_run_pipeline_eval_smoke(monkeypatch, tmp_path):
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL is not configured")

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(delete(EvalRun))
        await session.execute(delete(Passage))
        session.add(
            Passage(
                game_slug="hades",
                source_url="https://hades.fandom.com/wiki/Nyx",
                content="Nyx is the Goddess of Night and a maternal figure to Zagreus.",
                content_hash="smoke-nyx",
                spoiler_tier=0,
                embedding=[0.0] * 768,
            )
        )
        await session.commit()

    dataset_path = tmp_path / "smoke.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"smoke-001","question":"What is her title?",'
                    '"expected_answer":"Nyx is the Goddess of Night.","stratum":"multi_hop",'
                    '"spoiler_tier":0,'
                    '"history":[{"role":"user","content":"Tell me about Nyx."},'
                    '{"role":"assistant","content":"Nyx is important in the House of Hades."}],'
                    '"gold_source_urls":["https://hades.fandom.com/wiki/Nyx"]}'
                )
            ]
        )
    )
    output_path = tmp_path / "smoke-report.json"

    fake_services = SimpleNamespace(
        settings=SimpleNamespace(
            retrieval_top_k_per_method=5,
            retrieval_top_k_final=5,
        ),
        tracer=noop_tracer(),
        embedder=_FakeEmbedder(),
        dense=_FakeDenseRetriever(),
        router=_FakeRouter(),
    )
    monkeypatch.setattr("app.eval.runner.build_services", lambda: fake_services)

    report = await run_pipeline_eval(
        game_slug="hades",
        dataset_path=dataset_path,
        output_path=output_path,
        run_name="smoke-eval",
    )

    assert report["metrics"]["dataset_size"] == 1
    assert report["metrics"]["retrieval_recall_at_5_mean"] == 1.0
    assert report["metrics"]["retrieval_recall_at_5_exact_url_mean"] == 1.0
    assert report["metrics"]["citation_validity_rate"] == 1.0
    assert report["metrics"]["citation_validity_exact_url_rate"] == 1.0
    assert report["metrics"]["faithfulness_rate"] == 1.0
    assert output_path.exists()
