"""Which models the benchmark is allowed to call.

**Free models only, from an explicit allowlist, checked before any request
leaves the machine.**

An API key opening a door does not make what is behind it free. Groq's key
reaches a dozen models that all cost money; OpenRouter's reaches hundreds, of
which a handful are free. So possession of a credential proves nothing about
billing, and the benchmark treats the allowlist below as the only authority.

**Nothing here is remembered.** Every entry was verified against the provider's
own pricing data on the date recorded, by reading the price per token and
requiring it to be exactly zero. A model whose free status could not be
established that way is absent, however plausible it looks.

**Groq entries rest on different evidence, and the difference is marked.**
Groq's ``/v1/models`` reports a positive ``pricing.prompt`` and
``pricing.completion`` for every chat model it serves — ``llama-3.1-8b-instant``
is priced at $0.00000005 per prompt token, re-checked on 2026-08-14. Groq's free
tier is an account-level allowance against priced models, not a set of
zero-priced ones, and nothing in the API distinguishes "within my free
allowance" from "billed".

That could not be established from inside this repository, so for a time no Groq
model was approved. It was then established the only way it can be: the account
owner confirmed the plan does not bill for them. Those entries carry
``Evidence.ACCOUNT_TIER`` rather than ``Evidence.ZERO_PRICE``, because a claim
about one account is not the same kind of fact as a published price, and
**they are not portable** — a different account, or this one on a different
plan, makes them false.

**The registry goes stale, and this one already has.** Re-verified on 2026-08-23
against both live catalogues, which changed three entries:

* ``openai/gpt-oss-20b:free`` on OpenRouter **stopped being free** and was
  removed. An allowlist naming a model that now costs money is the exact failure
  this file exists to prevent, and it happened within ten days of being written.
* ``llama-3.1-8b-instant`` and ``llama-3.3-70b-versatile`` **no longer exist** on
  the Groq account, taking with them every non-reasoning chat model it had.

The second is worth stating plainly because it changes how the rest of the
project must be used: **every general chat model now reachable on Groq reasons
before it answers**, so `check_output_cap` is not an edge case any more. A
compaction budget that worked in August fails on all three.

Re-verify before trusting an entry — `python -m open_context verify-models` does
it against the live catalogues — and treat a `ModelNotApprovedError` as the
registry needing attention rather than as an obstacle to route around.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from open_context.llm.errors import InvalidRequestError, LLMError

REASONING_FLOOR = 512
"""Output tokens reserved for a model that thinks before it writes.

One number rather than a per-model figure. The observed cost varies with the
prompt — 44 to 332 tokens across the models measured here — so a per-model value
copied from one observation would be a false precision, and the useful question
is not "how much does it usually spend" but "is this cap so small that the
answer cannot survive it".
"""


class BillingClass(StrEnum):
    """What a call to a model costs.

    ``FREE`` is the only value the benchmark accepts. ``PAID`` and ``UNKNOWN``
    exist so a caller can say which one they meant and be refused explicitly,
    rather than having an absent field read as permission.
    """

    FREE = "free"
    PAID = "paid"
    UNKNOWN = "unknown"


class Evidence(StrEnum):
    """How an entry's freeness was established.

    **Two kinds of evidence, and they are not equally strong.** Keeping them
    apart in the type rather than only in prose means a reader can tell at a
    glance which entries rest on a number the provider published and which rest
    on somebody's word about their own account.
    """

    ZERO_PRICE = "zero_price"
    """The provider's own catalogue reported a per-token price of exactly zero.

    Checkable by anyone, from the endpoint named in ``verified_by``.
    """

    ACCOUNT_TIER = "account_tier"
    """The account owner confirmed their plan does not bill for this model.

    **Weaker, and deliberately marked as such.** The model carries a positive
    published price; what makes it free is a property of one account, which the
    API does not expose and nobody else can re-check. It is trusted because the
    person who owns the account and the bill said so, and it stops being true
    the moment that account changes plan.
    """


class Reasoning(StrEnum):
    """Whether a model spends output tokens thinking, and where that thinking goes.

    Recorded because it decides whether a given output cap can work at all, and
    because the reasoning kinds fail in different ways.

    **A value here is a measurement or it is ``UNKNOWN``.** Nothing is read from
    a model card or inferred from a name. ``FreeModelSpec.reasoning_measured_on``
    carries the date the observation was made, and an entry without one is not
    entitled to any value but ``UNKNOWN``.

    That rule exists because the file once broke it. Five OpenRouter entries were
    declared ``NONE`` without ever being called; three of them turned out to
    reason — 44, 78, and 91 reasoning tokens on a trivial prompt — so the output
    cap gate had been silently inert for that provider. An unmeasured model
    defaulting to "does not reason" is the most dangerous possible default,
    because it is the one that disables the check.
    """

    NONE = "none"
    """Writes its answer directly. Any cap that fits the answer is enough."""

    SEPARATE_FIELD = "separate_field"
    """Thinks in its own response field, and reports the token cost.

    The thinking is easy to ignore but is still billed against
    ``max_output_tokens``. Set the cap too low and the response is well-formed
    with an empty ``content`` — measured on ``openai/gpt-oss-20b`` at a cap of
    160, which spent 158 tokens reasoning and emitted nothing.
    """

    UNKNOWN = "unknown"
    """Nobody has called this model to find out.

    **Treated as reasoning**, and given the same overhead, because the failure
    modes are not symmetrical: assuming a direct-answering model reasons costs
    some output budget, while assuming a reasoning model answers directly
    returns an empty completion. A registry entry that has not been measured
    must not be the one that turns the safety gate off.
    """

    INLINE = "inline"
    """Thinks inside ``content``, in ``<think>`` tags, and reports no token cost.

    The harder of the two. Nothing in the accounting shows it happened, and the
    text arrives where an answer is expected — so unstripped it is reasoning
    graded as a conclusion. Measured on ``qwen/qwen3.6-27b``, which at a cap of
    160 spent all 158 tokens inside an unterminated ``<think>`` block.
    """


@dataclass(frozen=True)
class FreeModelSpec:
    """One model approved for benchmark use.

    ``verified_on``, ``verified_by``, and ``evidence`` are not decoration. An
    allowlist whose entries cannot be re-checked is a list of assertions, and
    this one exists precisely because assertions about billing are what must not
    be trusted.
    """

    provider: str
    model_id: str
    billing_class: BillingClass
    context_window: int | None
    verified_on: str
    verified_by: str
    evidence: Evidence = Evidence.ZERO_PRICE
    reasoning: Reasoning = Reasoning.UNKNOWN
    """How this model spends its output budget. **Defaults to unknown.**

    A new entry is unmeasured until somebody calls the model, and the default
    must be the safe one. It previously defaulted to ``NONE``, which is how five
    entries came to claim they did not reason without anyone having checked.
    """

    reasoning_measured_on: str = ""
    """Date the reasoning behaviour was observed, or empty when it was not."""
    note: str = ""

    @property
    def reasoning_measured(self) -> bool:
        """Whether the reasoning value rests on a live observation.

        ``UNKNOWN`` is never measured by definition; anything else must carry the
        date it was seen. The two are kept apart so "we looked and it does not
        reason" cannot be confused with "nobody looked".
        """
        return self.reasoning is not Reasoning.UNKNOWN and bool(self.reasoning_measured_on)

    @property
    def reasoning_overhead(self) -> int:
        """Output tokens to leave for thinking before the model writes anything.

        Zero only for a model measured as answering directly. Everything else —
        including one nobody has called — gets the floor, because a cap at or
        below it buys nothing but deliberation.

        Deliberately generous. The observed cost varies with the prompt (44 to
        332 tokens across the models measured here), so a floor occasionally too
        large costs some budget while one occasionally too small returns an
        empty answer.
        """
        return 0 if self.reasoning is Reasoning.NONE else REASONING_FLOOR

    def __post_init__(self) -> None:
        if self.billing_class is not BillingClass.FREE:
            raise ValueError(
                f"{self.model_id!r} is in the free registry with billing_class "
                f"{self.billing_class!r}; the registry holds free models only"
            )


class ModelNotApprovedError(LLMError):
    """The requested model is not on the free allowlist.

    Raised before a request is built, so an unapproved or paid model costs
    exactly zero network calls and zero money.
    """

    def __init__(self, provider: str, model_id: str, *, reason: str = "") -> None:
        approved = approved_models(provider)
        available = ", ".join(spec.model_id for spec in approved) or "none for this provider"
        super().__init__(
            f"{provider}/{model_id} is not an approved free model"
            f"{f'; {reason}' if reason else ''}. "
            f"The benchmark calls free models only and will not fall back to a paid one. "
            f"Approved for {provider}: {available}"
        )
        self.provider = provider
        self.model_id = model_id


VERIFIED_ON = "2026-08-23"
VERIFIED_BY = "openrouter /api/v1/models reported pricing.prompt == 0 and pricing.completion == 0"

GROQ_VERIFIED_ON = "2026-08-23"
GROQ_REASONING_MEASURED_ON = "2026-08-23"
"""When each Groq model was called to see how it spends its output budget."""

OPENROUTER_REASONING_MEASURED_ON = "2026-08-24"
"""When each OpenRouter model was called for the same reason.

Later than the billing check because it is a different observation: a catalogue
lists a price, and only a real generation shows where the output goes.
"""
GROQ_VERIFIED_BY = (
    "account owner confirmed the Groq account is on the free tier and is not billed for "
    "these models; Groq's catalogue prices them above zero and exposes no plan or billing "
    "field, so this rests on the owner's statement rather than on a published number"
)
"""Why Groq entries exist despite every Groq model carrying a positive price.

**This is an account-scoped exception, not a correction to the pricing finding.**
The finding below still holds: Groq prices every chat model it serves, and no
API response distinguishes "within my allowance" from "billed". What changed is
that the person who owns the account and would receive the bill confirmed the
plan does not charge for them.

It follows that this entry is **not portable**. A different account, or this one
on a different plan, makes it false, and nothing here can detect that happening.
Anyone reusing this repository should delete these entries rather than inherit
somebody else's billing arrangement.
"""

APPROVED_FREE_MODELS: tuple[FreeModelSpec, ...] = (
    FreeModelSpec(
        provider="openrouter",
        model_id="google/gemma-4-31b-it:free",
        billing_class=BillingClass.FREE,
        context_window=262_144,
        verified_on=VERIFIED_ON,
        verified_by=VERIFIED_BY,
        reasoning=Reasoning.NONE,
        reasoning_measured_on=OPENROUTER_REASONING_MEASURED_ON,
        note="General-purpose instruction-tuned model.",
    ),
    FreeModelSpec(
        provider="openrouter",
        model_id="google/gemma-4-26b-a4b-it:free",
        billing_class=BillingClass.FREE,
        context_window=262_144,
        verified_on=VERIFIED_ON,
        verified_by=VERIFIED_BY,
        reasoning=Reasoning.NONE,
        reasoning_measured_on=OPENROUTER_REASONING_MEASURED_ON,
    ),
    FreeModelSpec(
        provider="openrouter",
        model_id="nvidia/nemotron-3-super-120b-a12b:free",
        billing_class=BillingClass.FREE,
        context_window=262_144,
        verified_on=VERIFIED_ON,
        verified_by=VERIFIED_BY,
        reasoning=Reasoning.SEPARATE_FIELD,
        reasoning_measured_on=OPENROUTER_REASONING_MEASURED_ON,
    ),
    FreeModelSpec(
        provider="openrouter",
        model_id="liquid/lfm-2.5-2.6b:free",
        billing_class=BillingClass.FREE,
        context_window=128_000,
        verified_on=VERIFIED_ON,
        verified_by=VERIFIED_BY,
        reasoning=Reasoning.SEPARATE_FIELD,
        reasoning_measured_on=OPENROUTER_REASONING_MEASURED_ON,
    ),
    *(
        FreeModelSpec(
            provider="groq",
            model_id=model_id,
            billing_class=BillingClass.FREE,
            context_window=131_072,
            verified_on=GROQ_VERIFIED_ON,
            verified_by=GROQ_VERIFIED_BY,
            evidence=Evidence.ACCOUNT_TIER,
            reasoning=reasoning,
            reasoning_measured_on=GROQ_REASONING_MEASURED_ON,
            note="Priced in Groq's catalogue; free under the account owner's plan.",
        )
        for model_id, reasoning in (
            ("openai/gpt-oss-20b", Reasoning.SEPARATE_FIELD),
            ("openai/gpt-oss-120b", Reasoning.SEPARATE_FIELD),
            ("qwen/qwen3.6-27b", Reasoning.INLINE),
        )
    ),
)
"""Models the benchmark may call.

Small on purpose. Every one is a general-purpose chat model whose price was read
as zero on the date above; the provider's other zero-priced models are
special-purpose — content safety, code completion, vision — and are left out
because they are unsuitable for a continuation benchmark, not because their
billing is in doubt.
"""


def approved_models(provider: str | None = None) -> tuple[FreeModelSpec, ...]:
    """Everything approved, optionally for one provider."""
    if provider is None:
        return APPROVED_FREE_MODELS
    return tuple(spec for spec in APPROVED_FREE_MODELS if spec.provider == provider)


def find_free_model(provider: str, model_id: str) -> FreeModelSpec | None:
    """The approval for this model, or nothing."""
    for spec in APPROVED_FREE_MODELS:
        if spec.provider == provider and spec.model_id == model_id:
            return spec
    return None


def require_free_model(provider: str, model_id: str) -> FreeModelSpec:
    """The approval for this model, or refuse.

    **Call this before building a request, never after.** The guarantee is that
    an unapproved model produces zero HTTP requests, and a check that ran after
    the request was assembled would only be able to promise that the response
    was ignored.

    A ``:free`` suffix is not sufficient and is not consulted. OpenRouter uses
    it as a naming convention, conventions drift, and a model that stopped being
    free would keep its name. The allowlist is the only authority.
    """
    spec = find_free_model(provider, model_id)
    if spec is None:
        raise ModelNotApprovedError(provider, model_id)
    return spec


def billing_class_of(provider: str, model_id: str) -> BillingClass:
    """What this model costs, as far as the registry knows.

    ``UNKNOWN`` for anything unapproved — never ``FREE`` by default, and never
    inferred from a name.
    """
    spec = find_free_model(provider, model_id)
    return spec.billing_class if spec else BillingClass.UNKNOWN


def check_output_cap(provider: str, model_id: str, max_output_tokens: int | None) -> None:
    """Refuse a cap the model would spend entirely on thinking.

    The response-time check in the provider catches this too, but only after the
    request has been made. Here it costs nothing — which matters, because the
    failure is not one bad answer but a whole run of them: a Phase 8.5 benchmark
    produced 26 cells of empty compacted context before anyone looked at a
    token count, and every one of them scored as a measurement.

    Silent when the model is unknown to the registry. This asserts what has been
    measured and nothing else.
    """
    spec = find_free_model(provider, model_id)
    if spec is None or max_output_tokens is None:
        return
    overhead = spec.reasoning_overhead
    if overhead and max_output_tokens <= overhead:
        direct = [s.model_id for s in approved_models(provider) if not s.reasoning_overhead]
        # Only offer the alternative when there is one. Every approved Groq chat
        # model reasons as of 2026-08-23, and "choose a model that answers
        # directly: none" is advice that cannot be taken.
        remedy = (
            f"Raise the cap above {overhead}, or choose a model that answers directly: "
            f"{', '.join(direct)}"
            if direct
            else f"Raise the cap above {overhead}. Every approved model on this provider "
            f"reasons, so there is no direct-answering alternative to switch to."
        )
        raise InvalidRequestError(
            f"{provider}/{model_id} reasons before it answers ({spec.reasoning}), and a "
            f"max_output_tokens of {max_output_tokens} is within the {overhead} tokens it "
            f"can spend thinking — the answer would be empty or truncated to nothing. "
            f"{remedy}"
        )


__all__ = [
    "APPROVED_FREE_MODELS",
    "BillingClass",
    "Evidence",
    "FreeModelSpec",
    "ModelNotApprovedError",
    "Reasoning",
    "approved_models",
    "billing_class_of",
    "check_output_cap",
    "find_free_model",
    "require_free_model",
]
