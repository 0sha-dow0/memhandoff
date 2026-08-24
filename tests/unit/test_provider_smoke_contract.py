"""What the real-provider smoke test promises, checked without a network.

The smoke test only runs when a credential is present, so for most of this
project's life it was skipped — and while it was skipped it stopped working. A
flat `max_output_tokens=300` fell inside the 512-token reasoning floor that every
approved Groq model turned out to have, so `check_output_cap` refused the request
before it was sent and the test failed on every run where a key existed.

These run always, and pin the two properties that made it rot: the budget is
derived from the model, and only a rate limit counts as transient.
"""

import inspect

import pytest

from open_context.llm.errors import (
    AuthenticationError,
    InvalidRequestError,
    LLMError,
    ModelUnavailableError,
    RateLimitError,
)
from open_context.llm.free_models import (
    APPROVED_FREE_MODELS,
    REASONING_FLOOR,
    Reasoning,
    check_output_cap,
)

smoke = pytest.importorskip("tests.integration.test_real_provider_smoke")


# ----------------------------------------------------------------------
# The budget is derived, not written down


def test_the_requested_budget_clears_every_approved_model_gate():
    """Derived from the registry, so the test and the gate agree by
    construction — a model whose floor changes moves this with it."""
    for spec in APPROVED_FREE_MODELS:
        request = smoke.prompt_for(spec)
        assert request.max_output_tokens is not None
        check_output_cap(spec.provider, spec.model_id, request.max_output_tokens)


def test_a_reasoning_model_gets_room_beyond_its_floor():
    reasoning = [s for s in APPROVED_FREE_MODELS if s.reasoning is not Reasoning.NONE]
    assert reasoning, "the registry should still hold at least one reasoning model"
    for spec in reasoning:
        assert smoke.prompt_for(spec).max_output_tokens > REASONING_FLOOR


def test_the_budget_is_not_a_constant():
    """A flat number is exactly what broke: it cannot follow the registry.

    Checked against the numbers the code actually uses rather than its text —
    the docstring explains the old value and so necessarily contains it.
    """
    import ast

    source = inspect.getsource(smoke.prompt_for)
    tree = ast.parse(source.lstrip())
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    }
    assert "reasoning_overhead" in source
    assert not literals, f"the budget still hardcodes {literals}"


def test_a_direct_model_is_not_charged_a_reasoning_floor():
    direct = [s for s in APPROVED_FREE_MODELS if s.reasoning is Reasoning.NONE]
    for spec in direct:
        assert smoke.prompt_for(spec).max_output_tokens == smoke.ANSWER_TOKENS


# ----------------------------------------------------------------------
# Only a rate limit is transient


def test_the_smoke_test_skips_on_a_rate_limit_and_not_otherwise():
    """A shared free tier throttles without warning, and failing the suite over
    somebody else's traffic would make a green run meaningless. Every other
    category is a real defect."""
    source = inspect.getsource(smoke.test_one_real_request)
    assert "except RateLimitError" in source
    assert "pytest.skip" in source

    transient = source.split("except RateLimitError")[1].split("except")[0]
    assert "skip" in transient


def test_authentication_failure_is_never_treated_as_transient():
    """A wrong key is a wrong key. Skipping on it would hide a broken setup
    behind a message about someone else being busy."""
    source = inspect.getsource(smoke.test_one_real_request)
    for fatal in ("AuthenticationError", "ModelUnavailableError"):
        block = source.split(f"except {fatal}")[1].split("except")[0]
        assert "raise" in block
        assert "skip" not in block


def test_the_error_taxonomy_separates_the_cases_the_test_relies_on():
    """The provider layer already sorts failures; this test leans on that rather
    than matching on message strings."""
    for error in (
        RateLimitError,
        AuthenticationError,
        ModelUnavailableError,
        InvalidRequestError,
    ):
        assert issubclass(error, LLMError)
        assert error is not LLMError
    assert not issubclass(AuthenticationError, RateLimitError)
    assert not issubclass(ModelUnavailableError, RateLimitError)


# ----------------------------------------------------------------------
# The free-model gate is unchanged


def test_the_smoke_test_only_ever_selects_an_approved_model():
    source = inspect.getsource(smoke.test_one_real_request)
    assert "find_free_model" in source
    assert "billing_class_of" in source


def test_a_paid_model_is_refused_before_any_request():
    from open_context.llm.free_models import ModelNotApprovedError, require_free_model

    with pytest.raises(ModelNotApprovedError):
        require_free_model("openrouter", "openai/gpt-4o")
