from app.rag.refusal import RefusalPayload, build_refusal
from app.rag.verifier import VerifierVerdict


def test_build_refusal_uses_verdict_rewrites_when_available():
    verdict = VerifierVerdict(
        is_faithful=False,
        has_sufficient_evidence=False,
        unsupported_claims=["Zeus has a secret third brother"],
        rewrite_suggestions=["Ask about Zeus's siblings in Olympus, not Hades"],
    )
    refusal = build_refusal(
        question="Who is Zeus's secret brother?",
        verdict=verdict,
        passages=[{"passage_id": 1, "source_url": "u", "content": "Zeus is …"}],
    )
    assert isinstance(refusal, RefusalPayload)
    assert "enough evidence" in refusal.message.lower()
    assert refusal.rewrite_suggestions == ["Ask about Zeus's siblings in Olympus, not Hades"]
    assert refusal.unsupported_claims == ["Zeus has a secret third brother"]


def test_build_refusal_falls_back_to_heuristic_rewrites_when_verdict_has_none():
    verdict = VerifierVerdict(
        is_faithful=False,
        has_sufficient_evidence=False,
        unsupported_claims=[],
        rewrite_suggestions=[],
    )
    refusal = build_refusal(
        question="Who is the hidden mentor in Elysium?",
        verdict=verdict,
        passages=[],
    )
    assert len(refusal.rewrite_suggestions) >= 1
    assert all(isinstance(s, str) and s.strip() for s in refusal.rewrite_suggestions)
