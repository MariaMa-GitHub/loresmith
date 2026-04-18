from __future__ import annotations

import json
from dataclasses import dataclass

from app.llm.base import LLMProvider

_JUDGE_SYSTEM = """You are grading a retrieval-augmented QA answer.
Return strict JSON with these keys only:
- faithful: true if the answer is fully supported by the retrieved passages
- answer_correct: true if the answer matches the expected answer closely enough
- refusal_appropriate: true/false when the task expects a refusal, otherwise null

Do not include markdown fences or any extra text."""


@dataclass(frozen=True)
class AnswerJudgment:
    faithful: bool | None
    answer_correct: bool | None
    refusal_appropriate: bool | None
    raw: str | None = None


def _parse_json_object(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
        if lowered in {"null", "none", ""}:
            return None
    return None


async def judge_answer(
    *,
    llm: LLMProvider,
    question: str,
    expected_answer: str,
    actual_answer: str,
    passages: list[dict],
    expects_refusal: bool | None = None,
) -> AnswerJudgment:
    passages_block = "\n\n".join(
        f"[{idx}] {passage['source_url']}\n{passage['content']}"
        for idx, passage in enumerate(passages, start=1)
    )
    prompt = (
        f"Question:\n{question}\n\n"
        f"Expected answer:\n{expected_answer}\n\n"
        f"Actual answer:\n{actual_answer}\n\n"
        f"Retrieved passages:\n{passages_block or '(none)'}\n\n"
        f"Expects refusal: {expects_refusal!r}\n"
    )

    raw = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        system=_JUDGE_SYSTEM,
    )
    payload = _parse_json_object(raw)
    if payload is None:
        return AnswerJudgment(
            faithful=None,
            answer_correct=None,
            refusal_appropriate=None,
            raw=raw,
        )

    return AnswerJudgment(
        faithful=_coerce_bool(payload.get("faithful")),
        answer_correct=_coerce_bool(payload.get("answer_correct")),
        refusal_appropriate=_coerce_bool(payload.get("refusal_appropriate")),
        raw=raw,
    )
