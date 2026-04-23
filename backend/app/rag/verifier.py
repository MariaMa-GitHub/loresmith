from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.llm.base import LLMProvider
from app.tracing.langfuse import noop_tracer

_SYSTEM_PROMPT = """You are a strict retrieval-grounding judge.
Return ONLY a JSON object with these keys — no markdown, no commentary:
- is_faithful: true if every factual claim in the answer is supported by the passages.
- has_sufficient_evidence: true if the passages contain enough information to answer the question.
- unsupported_claims: list of short strings naming any claims not supported by the passages. [] if none.
- rewrite_suggestions: list of up to 3 short rewording suggestions that might retrieve better passages. [] if not applicable.
Grounded refusals ("I don't have enough evidence …") are considered faithful if they match the passages.
"""


@dataclass(frozen=True)
class VerifierVerdict:
    is_faithful: bool
    has_sufficient_evidence: bool
    unsupported_claims: list[str] = field(default_factory=list)
    rewrite_suggestions: list[str] = field(default_factory=list)
    raw: str | None = None


def _parse_json(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class Verifier:
    def __init__(self, llm: LLMProvider, tracer=None) -> None:
        self._llm = llm
        self._tracer = tracer or noop_tracer()

    async def verify(
        self,
        *,
        question: str,
        answer: str,
        passages: list[dict],
    ) -> VerifierVerdict:
        if not passages:
            return VerifierVerdict(
                is_faithful=False,
                has_sufficient_evidence=False,
                unsupported_claims=[],
                rewrite_suggestions=[],
            )

        passages_block = "\n\n".join(
            f"[{idx}] {p.get('source_url', '')}\n{p.get('content', '')}"
            for idx, p in enumerate(passages, start=1)
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Retrieved passages:\n{passages_block}\n"
        )

        with self._tracer.trace("rag.verify") as span:
            raw = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM_PROMPT,
            )
            span.set_output(raw)

        payload = _parse_json(raw)
        if payload is None:
            # Fail-open: don't convert every answer into a refusal when the
            # judge itself returns garbage.
            return VerifierVerdict(
                is_faithful=True,
                has_sufficient_evidence=True,
                unsupported_claims=[],
                rewrite_suggestions=[],
                raw=raw,
            )

        return VerifierVerdict(
            is_faithful=bool(payload.get("is_faithful", True)),
            has_sufficient_evidence=bool(payload.get("has_sufficient_evidence", True)),
            unsupported_claims=list(payload.get("unsupported_claims") or []),
            rewrite_suggestions=list(payload.get("rewrite_suggestions") or []),
            raw=raw,
        )
