"""The free-only policy.

**The invariant these exist to protect: the project never makes a paid request.**
An unapproved or paid model must cost zero network calls, which means the check
runs before a request is built rather than after a response comes back.
"""

import pytest

from open_context.llm import (
    ChatMessage,
    GenerationRequest,
    InvalidRequestError,
    ProviderConfig,
    create_provider,
)
from open_context.llm.free_models import (
    APPROVED_FREE_MODELS,
    REASONING_FLOOR,
    BillingClass,
    Evidence,
    FreeModelSpec,
    ModelNotApprovedError,
    Reasoning,
    approved_models,
    billing_class_of,
    check_output_cap,
    find_free_model,
    require_free_model,
)
from open_context.llm.providers import GroqProvider, OpenRouterProvider, groq, openrouter
from open_context.llm.providers._http import HttpRequest, HttpResponse

KEY = "test-key-not-a-real-credential"
APPROVED = APPROVED_FREE_MODELS[0].model_id


class CountingTransport:
    """Counts requests so "zero HTTP" can be asserted rather than assumed."""

    def __init__(self):
        self.calls = 0

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.calls += 1
        return HttpResponse(status=200, body=b'{"choices":[{"message":{"content":"x"}}]}')


# ----------------------------------------------------------------------
# The registry


def test_the_registry_holds_only_free_models():
    assert APPROVED_FREE_MODELS
    for spec in APPROVED_FREE_MODELS:
        assert spec.billing_class is BillingClass.FREE


def test_a_paid_spec_cannot_be_put_in_the_registry():
    """The type refuses it, so a paid entry cannot be added by accident."""
    with pytest.raises(ValueError, match="free models only"):
        FreeModelSpec(
            provider="openrouter",
            model_id="something/paid",
            billing_class=BillingClass.PAID,
            context_window=1000,
            verified_on="2026-08-13",
            verified_by="test",
        )


def test_every_entry_records_how_and_when_it_was_verified():
    """An allowlist whose entries cannot be re-checked is a list of assertions."""
    for spec in APPROVED_FREE_MODELS:
        assert spec.verified_on
        assert spec.verified_by
        assert len(spec.verified_by) > 30, spec.model_id


def test_a_zero_price_entry_cites_the_published_price():
    """The strong evidence has to name where the number came from."""
    for spec in APPROVED_FREE_MODELS:
        if spec.evidence is Evidence.ZERO_PRICE:
            assert "pricing" in spec.verified_by.lower(), spec.model_id


def test_an_account_tier_entry_is_marked_as_weaker_evidence():
    """A claim about one account is not the same kind of fact as a published price.

    Marked in the type rather than only in prose, so a reader can tell at a
    glance which entries anyone can re-check and which rest on somebody's word
    about their own billing.
    """
    groq_specs = approved_models("groq")
    assert groq_specs, "the fixture would be vacuous otherwise"
    for spec in groq_specs:
        assert spec.evidence is Evidence.ACCOUNT_TIER
        assert "account owner confirmed" in spec.verified_by


def test_groq_models_are_approved_on_account_evidence_only():
    """Groq prices every chat model it serves; its free tier is an account allowance.

    Nothing in the API distinguishes "within my allowance" from "billed", so a
    Groq entry can never rest on a published price. These exist because the
    account owner confirmed their plan does not bill for them, which is why they
    are marked as the weaker kind of evidence and are not portable to another
    account.
    """
    specs = approved_models("groq")
    assert specs
    assert all(spec.evidence is Evidence.ACCOUNT_TIER for spec in specs)


def test_a_groq_model_outside_the_allowlist_is_still_refused():
    """Approving some Groq models did not approve the provider wholesale."""
    with pytest.raises(ModelNotApprovedError):
        require_free_model("groq", "meta-llama/llama-prompt-guard-2-86m")


def test_openrouter_models_are_approved():
    assert approved_models("openrouter")
    assert all(spec.provider == "openrouter" for spec in approved_models("openrouter"))


# ----------------------------------------------------------------------
# Validation


def test_an_approved_model_passes():
    spec = require_free_model("openrouter", APPROVED)
    assert spec.billing_class is BillingClass.FREE


def test_an_unapproved_openrouter_model_is_refused():
    with pytest.raises(ModelNotApprovedError, match="not an approved free model"):
        require_free_model("openrouter", "anthropic/claude-opus-4")


def test_an_approved_groq_model_is_allowed():
    assert require_free_model("groq", "openai/gpt-oss-20b").provider == "groq"


def test_a_free_suffix_alone_is_not_enough():
    """OpenRouter's ``:free`` is a naming convention, and conventions drift.

    A model that stopped being free would keep its name, so the suffix is never
    consulted; only the allowlist is.
    """
    with pytest.raises(ModelNotApprovedError):
        require_free_model("openrouter", "someone/invented-model:free")


def test_a_model_without_the_suffix_is_also_refused_when_unapproved():
    with pytest.raises(ModelNotApprovedError):
        require_free_model("openrouter", "someone/invented-model")


def test_the_refusal_names_what_is_allowed():
    with pytest.raises(ModelNotApprovedError) as info:
        require_free_model("openrouter", "nope")
    message = str(info.value)
    assert APPROVED in message
    assert "will not fall back to a paid one" in message


def test_an_unapproved_model_has_unknown_billing_never_free():
    """An absent entry never reads as permission."""
    assert billing_class_of("openrouter", APPROVED) is BillingClass.FREE
    assert billing_class_of("openrouter", "unknown/model") is BillingClass.UNKNOWN
    assert billing_class_of("groq", "meta-llama/llama-prompt-guard-2-86m") is BillingClass.UNKNOWN


def test_lookup_is_provider_scoped():
    """An approved OpenRouter id does not become approved for another provider."""
    assert find_free_model("openrouter", APPROVED) is not None
    assert find_free_model("groq", APPROVED) is None


# ----------------------------------------------------------------------
# Zero HTTP before validation


def test_an_unapproved_model_makes_no_http_request():
    """The cost invariant, proved by counting."""
    transport = CountingTransport()

    with pytest.raises(ModelNotApprovedError):
        OpenRouterProvider(model="expensive/model", api_key=KEY, transport=transport)
    assert transport.calls == 0


def test_an_unapproved_groq_model_makes_no_http_request():
    transport = CountingTransport()

    with pytest.raises(ModelNotApprovedError):
        GroqProvider(model="meta-llama/llama-prompt-guard-2-86m", api_key=KEY, transport=transport)
    assert transport.calls == 0


def test_the_factory_path_makes_no_http_request(monkeypatch):
    """The path the benchmark and CLI actually take."""
    monkeypatch.setenv(openrouter.API_KEY_ENV, KEY)
    monkeypatch.setenv(groq.API_KEY_ENV, KEY)

    with pytest.raises(ModelNotApprovedError):
        create_provider(ProviderConfig(provider="openrouter", model="anthropic/claude-opus-4"))
    with pytest.raises(ModelNotApprovedError):
        create_provider(
            ProviderConfig(provider="groq", model="meta-llama/llama-prompt-guard-2-86m")
        )


def test_an_approved_model_does_reach_the_transport():
    """A guard that the checks above are not passing because nothing ever runs."""
    transport = CountingTransport()
    provider = OpenRouterProvider(model=APPROVED, api_key=KEY, transport=transport)
    provider.generate(GenerationRequest.of(ChatMessage.user("hi")))
    assert transport.calls == 1


# ----------------------------------------------------------------------
# No paid fallback


def test_a_rate_limit_does_not_switch_models():
    """A refused free model fails. It never becomes a paid one."""
    from open_context.llm import ChatMessage, GenerationRequest, RateLimitError

    def limited(request: HttpRequest) -> HttpResponse:
        return HttpResponse(
            status=429,
            body=b'{"error":{"message":"free-models-per-day exceeded"}}',
            headers={"retry-after": "60"},
        )

    provider = OpenRouterProvider(model=APPROVED, api_key=KEY, transport=limited)
    with pytest.raises(RateLimitError):
        provider.generate(GenerationRequest.of(ChatMessage.user("hi")))

    assert provider.model_info().model == APPROVED, "the model did not change"
    assert provider.billing_class is BillingClass.FREE


def test_a_provider_error_does_not_switch_providers():
    from open_context.llm import ChatMessage, GenerationRequest, ProviderUnavailableError

    def broken(request: HttpRequest) -> HttpResponse:
        return HttpResponse(status=503, body=b'{"error":{"message":"down"}}')

    provider = OpenRouterProvider(model=APPROVED, api_key=KEY, transport=broken)
    with pytest.raises(ProviderUnavailableError):
        provider.generate(GenerationRequest.of(ChatMessage.user("hi")))
    assert provider.model_info().provider == "openrouter"


def test_no_module_selects_a_replacement_model():
    """There is no fallback code to review, because there is none.

    A search rather than a behavioural check: the absence of a mechanism is
    stronger than a test of one path through it.
    """
    import ast
    from pathlib import Path

    import open_context

    root = Path(open_context.__file__).parent / "llm"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for suspicious in ("fallback_model", "next_model", "try_paid", "alternate_provider"):
            assert suspicious not in names, f"{path.name} looks like it selects a replacement"


def test_the_gate_is_on_by_default():
    """The escape hatch used by translation tests is never the default."""
    import inspect

    for cls in (GroqProvider, OpenRouterProvider):
        signature = inspect.signature(cls.__init__)
        assert signature.parameters["enforce_free"].default is True


def test_no_factory_disables_the_gate():
    """Every production path leaves the allowlist on."""
    import inspect

    for module in (groq, openrouter):
        source = inspect.getsource(module.build)
        assert "enforce_free" not in source, f"{module.__name__}.build touches the gate"


# ----------------------------------------------------------------------
# Reasoning overhead: the defect that voided a Phase 8.5 benchmark run


def test_every_approved_model_states_whether_it_reasons():
    """Unstated defaults to NONE, which is a claim, so it has to be a measured one."""
    for spec in APPROVED_FREE_MODELS:
        assert isinstance(spec.reasoning, Reasoning)


def test_the_measured_groq_behaviour_is_recorded():
    """Measured against the live endpoint on 2026-08-23, not read from a model card.

    The two llama models were dropped when they disappeared from the account,
    which left no non-reasoning chat model on this provider at all.
    """
    measured = {
        "openai/gpt-oss-20b": Reasoning.SEPARATE_FIELD,
        "openai/gpt-oss-120b": Reasoning.SEPARATE_FIELD,
        "qwen/qwen3.6-27b": Reasoning.INLINE,
    }
    for spec in approved_models("groq"):
        assert spec.reasoning is measured[spec.model_id], spec.model_id


def test_only_a_reasoning_model_carries_an_overhead():
    for spec in APPROVED_FREE_MODELS:
        if spec.reasoning is Reasoning.NONE:
            assert spec.reasoning_overhead == 0
        else:
            assert spec.reasoning_overhead > 0


def test_a_cap_a_reasoning_model_would_spend_entirely_on_thinking_is_refused():
    """Costs nothing to catch here; the alternative was 26 empty-context cells.

    The Phase 8.5 run set a 160-token compaction cap on ``openai/gpt-oss-20b``,
    which spent 158 of them reasoning and returned no content at all.
    """
    with pytest.raises(InvalidRequestError) as caught:
        check_output_cap("groq", "openai/gpt-oss-20b", 160)
    assert "reasons before it answers" in str(caught.value)
    assert "Raise the cap above 512" in str(caught.value)


def test_a_generous_cap_is_allowed():
    check_output_cap("groq", "openai/gpt-oss-20b", 900)


def test_a_model_that_does_not_reason_is_never_refused():
    check_output_cap("openrouter", "google/gemma-4-31b-it:free", 16)


def test_a_provider_with_no_direct_answering_model_says_so_rather_than_suggesting_one():
    """Every approved Groq chat model reasons, so "choose one that does not" is
    advice that cannot be taken."""
    with pytest.raises(InvalidRequestError) as caught:
        check_output_cap("groq", "qwen/qwen3.6-27b", 100)
    assert "no direct-answering alternative" in str(caught.value)


def test_an_unknown_model_is_not_second_guessed():
    """The registry asserts what was measured and nothing else."""
    check_output_cap("groq", "some-model-nobody-measured", 1)
    check_output_cap("groq", "openai/gpt-oss-20b", None)


def test_the_guard_runs_before_the_request_is_sent():
    """Zero HTTP, like every other refusal in this module."""
    transport = CountingTransport()
    provider = GroqProvider(model="openai/gpt-oss-20b", api_key=KEY, transport=transport)
    with pytest.raises(InvalidRequestError):
        provider.generate(GenerationRequest.of(ChatMessage.user("x"), max_output_tokens=160))
    assert transport.calls == 0


# ----------------------------------------------------------------------
# Reasoning is measured, or it is unknown — never assumed


def test_the_measured_openrouter_behaviour_is_recorded():
    """Measured against the live endpoint on 2026-08-24.

    Three of these were previously declared NONE without anyone having called
    them, and turned out to reason — 44, 78, and 91 reasoning tokens on a
    trivial prompt. The gate had been inert for this provider the whole time.
    """
    measured = {
        "google/gemma-4-31b-it:free": Reasoning.NONE,
        "google/gemma-4-26b-a4b-it:free": Reasoning.NONE,
        "nvidia/nemotron-3-nano-30b-a3b:free": Reasoning.SEPARATE_FIELD,
        "nvidia/nemotron-3-super-120b-a12b:free": Reasoning.SEPARATE_FIELD,
        "liquid/lfm-2.5-2.6b:free": Reasoning.SEPARATE_FIELD,
    }
    for spec in approved_models("openrouter"):
        assert spec.reasoning is measured[spec.model_id], spec.model_id


def test_every_approved_entry_carries_the_date_it_was_measured():
    """A value without a date is not a measurement, it is a belief."""
    for spec in APPROVED_FREE_MODELS:
        assert spec.reasoning_measured, f"{spec.model_id} claims {spec.reasoning} unmeasured"
        assert spec.reasoning_measured_on, spec.model_id


def test_an_unmeasured_entry_is_unknown_rather_than_none():
    """The default must be the safe one. It was NONE, which is how five entries
    came to claim they did not reason without anyone having checked."""
    fresh = FreeModelSpec(
        provider="somewhere",
        model_id="brand-new-model",
        billing_class=BillingClass.FREE,
        context_window=1000,
        verified_on="2026-08-24",
        verified_by="test",
    )
    assert fresh.reasoning is Reasoning.UNKNOWN
    assert not fresh.reasoning_measured


def test_an_unmeasured_model_is_protected_not_exempted():
    """Assuming a direct model reasons costs budget; assuming a reasoning model
    answers directly returns an empty completion. Only one of those is safe."""
    unknown = FreeModelSpec(
        provider="somewhere",
        model_id="brand-new-model",
        billing_class=BillingClass.FREE,
        context_window=1000,
        verified_on="2026-08-24",
        verified_by="test",
    )
    assert unknown.reasoning_overhead == REASONING_FLOOR


def test_a_value_without_a_date_does_not_count_as_measured():
    claimed = FreeModelSpec(
        provider="somewhere",
        model_id="asserted-model",
        billing_class=BillingClass.FREE,
        context_window=1000,
        verified_on="2026-08-24",
        verified_by="test",
        reasoning=Reasoning.NONE,
    )
    assert not claimed.reasoning_measured, "a bare value is not an observation"


def test_the_enum_no_longer_claims_every_value_was_measured():
    """The docstring said so while five entries had never been called."""
    import inspect

    text = inspect.getdoc(Reasoning) or ""
    assert "or it is ``UNKNOWN``" in text
    assert "measurement" in text


def test_a_reasoning_model_still_refuses_a_cap_below_its_floor():
    """The gate itself must not have been loosened by any of this."""
    for spec in APPROVED_FREE_MODELS:
        if spec.reasoning is Reasoning.NONE:
            continue
        with pytest.raises(InvalidRequestError):
            check_output_cap(spec.provider, spec.model_id, REASONING_FLOOR - 1)


def test_a_measured_direct_model_is_still_allowed_a_small_cap():
    check_output_cap("openrouter", "google/gemma-4-31b-it:free", 16)
