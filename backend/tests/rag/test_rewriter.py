from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.rewriter import QueryRewriter


@pytest.mark.asyncio
async def test_rewriter_no_history_returns_question_unchanged():
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="should not be called")

    rewriter = QueryRewriter(llm=mock_llm)
    result = await rewriter.rewrite(
        question="Who is Zagreus?",
        history=[],
    )

    assert result == "Who is Zagreus?"
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_rewriter_with_history_calls_llm():
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(
        return_value="What weapons does Zagreus use in his escape from Tartarus?"
    )

    rewriter = QueryRewriter(llm=mock_llm)
    result = await rewriter.rewrite(
        question="What weapons does he use?",
        history=[
            {"role": "user", "content": "Tell me about Zagreus's escape attempts."},
            {"role": "assistant", "content": "Zagreus tries to escape the Underworld repeatedly."},
        ],
    )

    assert "Zagreus" in result
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_rewriter_strips_whitespace_from_llm_output():
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="  rewritten query  \n")

    rewriter = QueryRewriter(llm=mock_llm)
    result = await rewriter.rewrite(
        question="any question",
        history=[{"role": "user", "content": "prior"}, {"role": "assistant", "content": "reply"}],
    )

    assert result == "rewritten query"


@pytest.mark.asyncio
async def test_rewriter_uses_at_most_last_n_turns():
    captured_prompts = []

    async def capture_complete(messages, system=None):
        captured_prompts.extend(messages)
        return "rewritten"

    mock_llm = MagicMock()
    mock_llm.complete = capture_complete

    rewriter = QueryRewriter(llm=mock_llm, max_history_turns=2)
    history = []
    for i in range(5):  # 5 turns of history
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    await rewriter.rewrite(question="current question", history=history)

    assert len(captured_prompts) == 1
    prompt_content = captured_prompts[0]["content"]
    # Only the last 2 turns (q3/a3, q4/a4) should appear in the prompt
    assert "q4" in prompt_content
    assert "a4" in prompt_content
    assert "q3" in prompt_content
    assert "a3" in prompt_content
    # Earlier turns must NOT appear
    assert "q0" not in prompt_content
    assert "q1" not in prompt_content
    assert "q2" not in prompt_content
