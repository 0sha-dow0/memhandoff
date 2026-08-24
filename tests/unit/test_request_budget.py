"""The request budget: counting, refusing, and staying out of the error hierarchy.

Every test here is offline. The budget is exactly the machinery that lets a
metered run be sized before it starts, so it must be testable without spending
the thing it meters.
"""

import pytest

from open_context.llm import ChatMessage, GenerationRequest, LLMError
from open_context.llm.fakes import FakeProvider
from open_context_eval.budget import BudgetedProvider, RequestBudget, RequestBudgetExhausted

REQUEST = GenerationRequest.of(ChatMessage.user("hello"))


def test_a_budget_starts_unspent():
    budget = RequestBudget(5)
    assert budget.spent == 0
    assert budget.remaining == 5
    assert not budget.exhausted


def test_spending_reduces_what_remains():
    budget = RequestBudget(3)
    budget.spend()
    budget.spend()
    assert budget.spent == 2
    assert budget.remaining == 1


def test_a_budget_refuses_the_request_past_its_limit():
    budget = RequestBudget(2)
    budget.spend()
    budget.spend()
    with pytest.raises(RequestBudgetExhausted):
        budget.spend()


def test_a_refused_request_is_not_charged():
    """The count stops at the limit rather than running past it.

    A budget that kept incrementing while refusing would report a spend it never
    made, and a report reading "60 of 50 requests" describes nothing that
    happened.
    """
    budget = RequestBudget(1)
    budget.spend()
    for _ in range(3):
        with pytest.raises(RequestBudgetExhausted):
            budget.spend()
    assert budget.spent == 1
    assert budget.remaining == 0


def test_a_zero_budget_refuses_immediately():
    with pytest.raises(RequestBudgetExhausted):
        RequestBudget(0).spend()


def test_a_negative_budget_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        RequestBudget(-1)


def test_exhaustion_is_not_an_llm_error():
    """The single most important property in this module.

    The runner turns any ``LLMError`` into a ``MODEL_ERROR`` result. If
    exhaustion were one, every cell past the budget would be recorded as a
    strategy that was tried and failed — which is precisely the outcome the
    budget exists to prevent, and it would be indistinguishable from a real
    provider failure in the results file.
    """
    assert not issubclass(RequestBudgetExhausted, LLMError)


def test_the_wrapper_charges_one_request_per_generate():
    provider = FakeProvider(reply="ok")
    budget = RequestBudget(10)
    wrapped = BudgetedProvider(provider, budget)

    wrapped.generate(REQUEST)
    wrapped.generate(REQUEST)

    assert budget.spent == 2
    assert len(provider.requests) == 2


def test_the_wrapper_stops_the_provider_being_called_at_all():
    """Exhaustion must prevent the request, not discard its answer.

    A budget checked after the call would have already spent the quota it was
    protecting.
    """
    provider = FakeProvider(reply="ok")
    wrapped = BudgetedProvider(provider, RequestBudget(1))

    wrapped.generate(REQUEST)
    with pytest.raises(RequestBudgetExhausted):
        wrapped.generate(REQUEST)

    assert len(provider.requests) == 1, "the second request never reached the provider"


def test_model_info_is_not_charged():
    """It answers from configuration and sends nothing.

    Charging it would make the count wrong in the direction that matters: a run
    would stop short of the quota it actually had.
    """
    provider = FakeProvider(reply="ok")
    budget = RequestBudget(1)
    wrapped = BudgetedProvider(provider, budget)

    wrapped.model_info()
    wrapped.model_info()

    assert budget.spent == 0
    wrapped.generate(REQUEST)
    assert budget.spent == 1


def test_the_wrapper_passes_the_response_through_unchanged():
    provider = FakeProvider(reply="the answer")
    wrapped = BudgetedProvider(provider, RequestBudget(2))
    assert wrapped.generate(REQUEST).text == "the answer"


def test_the_error_reports_what_was_spent():
    budget = RequestBudget(2)
    budget.spend()
    budget.spend()
    with pytest.raises(RequestBudgetExhausted) as caught:
        budget.spend()
    assert caught.value.spent == 2
    assert caught.value.limit == 2
    assert "2 of 2" in str(caught.value)


def test_the_error_says_the_cells_are_absent_rather_than_failed():
    """The message is read by whoever finds a short results file.

    It has to say which of the two things happened, because the file alone
    cannot: a cell that was never attempted looks exactly like one that was
    never written.
    """
    with pytest.raises(RequestBudgetExhausted) as caught:
        RequestBudget(0).spend()
    message = str(caught.value)
    assert "not attempted" in message
    assert "rather than recorded as failures" in message


# ----------------------------------------------------------------------
# Pacing


def test_pacing_is_off_by_default():
    """An offline run must not be slowed by machinery meant for a metered service."""
    slept: list[float] = []
    wrapped = BudgetedProvider(
        FakeProvider(reply="ok"), RequestBudget(5), sleep=slept.append, clock=lambda: 0.0
    )
    wrapped.generate(REQUEST)
    wrapped.generate(REQUEST)
    assert slept == []


def test_a_second_request_waits_out_the_interval():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    wrapped = BudgetedProvider(
        FakeProvider(reply="ok"),
        RequestBudget(5),
        min_interval_seconds=3.0,
        clock=lambda: now[0],
        sleep=sleep,
    )
    wrapped.generate(REQUEST)
    wrapped.generate(REQUEST)
    assert slept == [3.0]


def test_the_first_request_never_waits():
    slept: list[float] = []
    wrapped = BudgetedProvider(
        FakeProvider(reply="ok"),
        RequestBudget(5),
        min_interval_seconds=3.0,
        clock=lambda: 0.0,
        sleep=slept.append,
    )
    wrapped.generate(REQUEST)
    assert slept == []


def test_time_already_spent_in_a_slow_call_counts_towards_the_gap():
    """The limit counts requests per minute, so a slow call has supplied part of it.

    Sleeping a full interval after a call that already took longer than one would
    halve the throughput for no reason.
    """
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    provider = FakeProvider(reply="ok")
    wrapped = BudgetedProvider(
        provider,
        RequestBudget(5),
        min_interval_seconds=3.0,
        clock=lambda: now[0],
        sleep=sleep,
    )
    wrapped.generate(REQUEST)
    now[0] += 5.0  # the request itself took five seconds
    wrapped.generate(REQUEST)
    assert slept == [], "the gap was already satisfied"


def test_a_partial_gap_is_topped_up():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    wrapped = BudgetedProvider(
        FakeProvider(reply="ok"),
        RequestBudget(5),
        min_interval_seconds=3.0,
        clock=lambda: now[0],
        sleep=sleep,
    )
    wrapped.generate(REQUEST)
    now[0] += 1.0
    wrapped.generate(REQUEST)
    assert slept == [2.0]


def test_an_exhausted_budget_refuses_without_sleeping_first():
    """It declines the request, so there is nothing to pace."""
    slept: list[float] = []
    wrapped = BudgetedProvider(
        FakeProvider(reply="ok"),
        RequestBudget(1),
        min_interval_seconds=3.0,
        clock=lambda: 0.0,
        sleep=slept.append,
    )
    wrapped.generate(REQUEST)
    with pytest.raises(RequestBudgetExhausted):
        wrapped.generate(REQUEST)
    assert slept == []


def test_a_negative_interval_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        BudgetedProvider(FakeProvider(reply="ok"), RequestBudget(1), min_interval_seconds=-1)
