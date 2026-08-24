"""The LLM-backed judge.

Dataset v3 moved every absence assertion here, so what this module refuses to do
matters more than what it does: it must never manufacture a verdict, and a judge
that failed must never read as an answer that failed.

Offline throughout, against the deterministic fakes.
"""

import pytest

from open_context.llm import RateLimitError
from open_context.llm.fakes import FailingProvider, FakeProvider
from open_context_eval.judge import (
    JUDGE_PROMPT_V1,
    JudgeVerdict,
    KeywordJudge,
    LLMJudge,
    parse_verdict,
)


def judge_with(reply: str) -> LLMJudge:
    return LLMJudge(FakeProvider(reply=reply))


# ----------------------------------------------------------------------
# Reading a verdict


def test_yes_passes():
    verdict = parse_verdict("YES\nIt commits to SQLite.")
    assert verdict.evaluated and verdict.passed
    assert verdict.reasoning == "It commits to SQLite."


def test_no_fails():
    verdict = parse_verdict("NO\nIt recommends Redis outright.")
    assert verdict.evaluated and not verdict.passed


def test_case_and_whitespace_do_not_matter():
    for text in ["yes\nreason", "  Yes  \nreason", "YES"]:
        assert parse_verdict(text).passed, text


def test_markdown_and_punctuation_around_the_word_are_tolerated():
    """Shapes a model actually produces around a one-word answer."""
    for text in ["**YES**\nreason", "`NO`\nreason", "YES.\nreason", "NO:\nreason"]:
        assert parse_verdict(text).evaluated, text


def test_an_unreadable_reply_is_undecided_not_a_failure():
    """The property that keeps the judge from manufacturing verdicts.

    A guessed NO is indistinguishable from a real one in a results file, so
    guessing would silently corrupt exactly the measurements this judge was
    added to provide.
    """
    verdict = parse_verdict("It depends on what you mean by avoided.")
    assert not verdict.evaluated
    assert "could not read a verdict" in verdict.reasoning


def test_an_empty_reply_is_undecided():
    for text in ["", "   ", "\n\n"]:
        assert not parse_verdict(text).evaluated, repr(text)


def test_a_verdict_buried_in_prose_is_not_mined_for():
    """Finding one there means deciding which of several words was the answer."""
    assert not parse_verdict("Well, yes and no, but on balance yes.").evaluated


# ----------------------------------------------------------------------
# Calling a model


def test_the_judge_asks_and_answers():
    assert judge_with("YES\nlooks right").judge("some answer", "Is it right?").passed


def test_the_question_and_the_answer_both_reach_the_model():
    provider = FakeProvider(reply="YES\nfine")
    LLMJudge(provider).judge("THE ANSWER TEXT", "THE QUESTION TEXT")
    sent = " ".join(m.content for m in provider.requests[0].messages)
    assert "THE QUESTION TEXT" in sent
    assert "THE ANSWER TEXT" in sent


def test_the_prompt_tells_the_judge_that_naming_a_rejection_is_allowed():
    """The distinction the substring checks could not make, stated to the grader.

    Without this the judge would plausibly reproduce the very artefact that
    dataset v3 exists to remove.
    """
    text = JUDGE_PROMPT_V1.text.lower()
    assert "reject" in text or "exclude" in text
    assert "yes" in text


def test_a_model_error_is_undecided_not_a_failed_answer():
    """A rate limit says nothing about the answer being graded."""
    judge = LLMJudge(FailingProvider(error=RateLimitError("slow down")))
    verdict = judge.judge("an answer", "a question")
    assert not verdict.evaluated
    assert "judge model failed" in verdict.reasoning


def test_budget_exhaustion_is_not_swallowed():
    """It is not an ``LLMError``, and a judge absorbing it would let a run
    continue past the ceiling it was given."""
    from open_context_eval.budget import BudgetedProvider, RequestBudget, RequestBudgetExhausted

    judge = LLMJudge(BudgetedProvider(FakeProvider(reply="YES\nok"), RequestBudget(0)))
    with pytest.raises(RequestBudgetExhausted):
        judge.judge("an answer", "a question")


def test_the_name_records_which_model_graded():
    """Two runs judged by different models are different experiments.

    ``name`` goes into the run fingerprint, so an id that did not move between
    them would file both under one identity.
    """
    name = LLMJudge(FakeProvider(model="grader-9")).name
    assert "grader-9" in name
    assert JUDGE_PROMPT_V1.identifier in name


def test_judging_costs_one_request_per_question():
    provider = FakeProvider(reply="YES\nok")
    judge = LLMJudge(provider)
    judge.judge("a", "q1")
    judge.judge("b", "q2")
    assert len(provider.requests) == 2


def test_the_judge_asks_at_temperature_zero():
    provider = FakeProvider(reply="YES\nok")
    LLMJudge(provider).judge("a", "q")
    assert provider.requests[0].temperature == 0.0


# ----------------------------------------------------------------------
# The stand-in


def test_the_keyword_stand_in_is_undecided_without_a_rule():
    """Previously a failure, which penalised an answer for the harness's silence."""
    verdict = KeywordJudge().judge("anything", "a question nobody wrote a rule for")
    assert not verdict.evaluated


def test_the_keyword_stand_in_still_answers_what_it_was_given():
    judge = KeywordJudge(answers={"q": ("sqlite",)})
    assert judge.judge("we will use SQLite", "q").passed
    assert not judge.judge("we will use Postgres", "q").passed


def test_undecided_carries_no_pass():
    assert JudgeVerdict.undecided("nope").passed is False
    assert JudgeVerdict.undecided("nope").evaluated is False
