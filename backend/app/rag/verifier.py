from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)
from app.rag.structured_output import VerifierVerdictSchema
from app.tracing.langfuse import noop_tracer

_SYSTEM_PROMPT = """You are a strict retrieval-grounding judge.
Return ONLY a JSON object with these keys — no markdown, no commentary:
- is_faithful: true if every factual claim in the answer is supported by the passages.
- has_sufficient_evidence: true if the passages contain enough information to answer.
- unsupported_claims: list of short strings naming claims not in the passages. [] if none.
- rewrite_suggestions: up to 3 short rewording suggestions for better retrieval. [] if n/a.
Grounded refusals ("I don't have enough evidence …") are faithful if they match the passages.
"""


@dataclass(frozen=True)
class VerifierVerdict:
    is_faithful: bool
    has_sufficient_evidence: bool
    unsupported_claims: list[str] = field(default_factory=list)
    rewrite_suggestions: list[str] = field(default_factory=list)
    raw: str | None = None
    abstained: bool = False


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
        messages = [{"role": "user", "content": prompt}]

        with self._tracer.trace("rag.verify") as span:
            if hasattr(self._llm, "complete_json"):
                try:
                    result = await self._llm.complete_json(
                        messages=messages,
                        schema=VerifierVerdictSchema,
                        system=_SYSTEM_PROMPT,
                    )
                except Exception as exc:
                    logger.warning("verifier abstained on complete_json failure: %s", exc)
                    span.set_output("abstained")
                    return VerifierVerdict(
                        is_faithful=False,
                        has_sufficient_evidence=False,
                        abstained=True,
                    )
                span.set_output(result.model_dump_json())
                return VerifierVerdict(
                    is_faithful=result.is_faithful,
                    has_sufficient_evidence=result.has_sufficient_evidence,
                    unsupported_claims=list(result.unsupported_claims),
                    rewrite_suggestions=list(result.rewrite_suggestions),
                )

            raw = await self._llm.complete(messages=messages, system=_SYSTEM_PROMPT)
            span.set_output(raw)

        payload = _parse_json(raw)
        if payload is None:
            # Abstain — never silently coerce malformed judge output into
            # is_faithful=True; the caller must treat this as low confidence.
            return VerifierVerdict(
                is_faithful=False,
                has_sufficient_evidence=False,
                raw=raw,
                abstained=True,
            )

        return VerifierVerdict(
            is_faithful=bool(payload.get("is_faithful", False)),
            has_sufficient_evidence=bool(payload.get("has_sufficient_evidence", False)),
            unsupported_claims=list(payload.get("unsupported_claims") or []),
            rewrite_suggestions=list(payload.get("rewrite_suggestions") or []),
            raw=raw,
        )
