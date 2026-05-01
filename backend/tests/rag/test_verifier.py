import json
from unittest.mock import AsyncMock

import pytest

from app.rag.structured_output import VerifierVerdictSchema
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
    # Force fallback path by removing complete_json from the AsyncMock spec.
    del llm.complete_json
    verifier = Verifier(llm=llm)

    verdict = await verifier.verify(
        question="Who is Nyx?",
        answer="Nyx is the Goddess of Night [1].",
        passages=[{"passage_id": 1, "content": "Nyx is the goddess of night.", "source_url": "u"}],
    )

    assert isinstance(verdict, VerifierVerdict)
    assert verdict.is_faithful is True
    assert verdict.has_sufficient_evidence is True
    assert verdict.abstained is False


@pytest.mark.asyncio
async def test_verifier_flags_unsupported_claims():
    llm = AsyncMock()
    llm.complete.return_value = json.dumps({
        "is_faithful": False,
        "has_sufficient_evidence": False,
        "unsupported_claims": ["Nyx has three children"],
        "rewrite_suggestions": ["Ask about Nyx's role, not her children"],
    })
    del llm.complete_json
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
    assert verdict.abstained is False


@pytest.mark.asyncio
async def test_verifier_abstains_on_malformed_json_fallback():
    """If a non-structured provider returns garbage, verifier abstains
    (no longer fails-open into is_faithful=True)."""
    llm = AsyncMock()
    llm.complete.return_value = "not json at all"
    del llm.complete_json
    verifier = Verifier(llm=llm)

    verdict = await verifier.verify(
        question="q", answer="a", passages=[{"passage_id": 1, "content": "p", "source_url": "u"}]
    )
    assert verdict.abstained is True
    assert verdict.is_faithful is False
    assert verdict.has_sufficient_evidence is False
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


class _StructuredFakeLLM:
    model_name = "fake-structured"

    def __init__(self, schema_result=None, raise_exc=None):
        self._schema_result = schema_result
        self._raise_exc = raise_exc

    async def complete(self, messages, system=None):
        return ""

    async def stream(self, messages, system=None):
        yield ""

    async def complete_json(self, messages, schema, system=None):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._schema_result


@pytest.mark.asyncio
async def test_verifier_uses_complete_json_when_available():
    schema_result = VerifierVerdictSchema(
        is_faithful=True,
        has_sufficient_evidence=True,
        unsupported_claims=[],
        rewrite_suggestions=[],
    )
    llm = _StructuredFakeLLM(schema_result=schema_result)
    verdict = await Verifier(llm=llm).verify(
        question="q",
        answer="a",
        passages=[{"source_url": "x", "content": "y"}],
    )
    assert verdict.abstained is False
    assert verdict.is_faithful is True
    assert verdict.has_sufficient_evidence is True


@pytest.mark.asyncio
async def test_verifier_abstains_on_complete_json_exception():
    llm = _StructuredFakeLLM(raise_exc=ValueError("malformed"))
    verdict = await Verifier(llm=llm).verify(
        question="q",
        answer="a",
        passages=[{"source_url": "x", "content": "y"}],
    )
    assert verdict.abstained is True
    assert verdict.is_faithful is False
    assert verdict.has_sufficient_evidence is False
