"""Pydantic schemas for structured LLM outputs.

These schemas are passed to provider ``complete_json`` methods to constrain
the model's response shape and validate it before downstream consumers see
it. Treat parse/validation failures as ``abstain``, never as a permissive
fallback — the verifier in particular must not silently fail open.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class VerifierVerdictSchema(BaseModel):
    """Schema for the verifier's grounded-judge JSON output."""

    model_config = ConfigDict(extra="forbid")

    is_faithful: bool
    has_sufficient_evidence: bool
    unsupported_claims: list[str] = []
    rewrite_suggestions: list[str] = []


class AnswerWithUsedPassages(BaseModel):
    """Schema for structured answer generation with explicit passage selection."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    used_passages: list[int]
