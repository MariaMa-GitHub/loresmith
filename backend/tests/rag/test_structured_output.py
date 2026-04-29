import pytest
from pydantic import ValidationError

from app.rag.structured_output import VerifierVerdictSchema


def test_accepts_minimal_valid():
    v = VerifierVerdictSchema(is_faithful=True, has_sufficient_evidence=False)
    assert v.unsupported_claims == []
    assert v.rewrite_suggestions == []


def test_rejects_extra_keys():
    with pytest.raises(ValidationError):
        VerifierVerdictSchema(
            is_faithful=True,
            has_sufficient_evidence=True,
            unknown_key="x",
        )


def test_defaults_for_list_fields():
    v = VerifierVerdictSchema(is_faithful=False, has_sufficient_evidence=False)
    assert v.rewrite_suggestions == []
    assert v.unsupported_claims == []


def test_round_trip_full_payload():
    v = VerifierVerdictSchema(
        is_faithful=False,
        has_sufficient_evidence=False,
        unsupported_claims=["Zeus has a third brother"],
        rewrite_suggestions=["Try canonical Olympian relationships"],
    )
    assert v.unsupported_claims == ["Zeus has a third brother"]
    assert v.rewrite_suggestions == ["Try canonical Olympian relationships"]
