import json
from unittest.mock import AsyncMock

import pytest

from app.rag.verifier import Verifier, VerifierVerdict


@pytest.mark.asyncio
async def test_verifier_parses_json_verdict():
    llm = AsyncMock()
    llm.complete.return_value = json.dumps({
        "is_faithful": True,
        "has_sufficient_evidence": True,
        "unsupported_claims": [],
        "rewrite_suggestions": [],
    })
    verifier = Verifier(llm=llm)

    verdict = await verifier.verify(
        question="Who is Nyx?",
        answer="Nyx is the Goddess of Night [1].",
        passages=[{"passage_id": 1, "content": "Nyx is the goddess of night.", "source_url": "u"}],
    )

    assert isinstance(verdict, VerifierVerdict)
    assert verdict.is_faithful is True
    assert verdict.has_sufficient_evidence is True


@pytest.mark.asyncio
async def test_verifier_flags_unsupported_claims():
    llm = AsyncMock()
    llm.complete.return_value = json.dumps({
        "is_faithful": False,
        "has_sufficient_evidence": False,
        "unsupported_claims": ["Nyx has three children"],
        "rewrite_suggestions": ["Ask about Nyx's role, not her children"],
    })
    verifier = Verifier(llm=llm)

    verdict = await verifier.verify(
        question="How many children does Nyx have?",
        answer="Nyx has three children.",
        passages=[{"passage_id": 1, "content": "Nyx is the goddess of night.", "source_url": "u"}],
    )

    assert verdict.is_faithful is False
    assert verdict.has_sufficient_evidence is False
    assert verdict.unsupported_claims == ["Nyx has three children"]
    assert verdict.rewrite_suggestions == ["Ask about Nyx's role, not her children"]


@pytest.mark.asyncio
async def test_verifier_handles_malformed_json_by_defaulting_to_sufficient():
    """A broken judge should not take down the answer path — fail open."""
    llm = AsyncMock()
    llm.complete.return_value = "not json at all"
    verifier = Verifier(llm=llm)

    verdict = await verifier.verify(
        question="q", answer="a", passages=[{"passage_id": 1, "content": "p", "source_url": "u"}]
    )
    assert verdict.is_faithful is True
    assert verdict.has_sufficient_evidence is True
    assert verdict.unsupported_claims == []
    assert verdict.rewrite_suggestions == []


@pytest.mark.asyncio
async def test_verifier_skips_when_no_passages():
    """No retrieved passages ⇒ insufficient evidence, no LLM call."""
    llm = AsyncMock()
    verifier = Verifier(llm=llm)

    verdict = await verifier.verify(question="q", answer="a", passages=[])

    assert verdict.has_sufficient_evidence is False
    llm.complete.assert_not_called()
