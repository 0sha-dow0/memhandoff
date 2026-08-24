"""Provider-neutral errors.

**A vendor exception must never reach the core.** A provider implementation
translates whatever its SDK or HTTP layer raises into one of these, so code
above the boundary can handle "rate limited" or "context too long" without
importing anyone's exception classes or matching on their message strings. A
provider that lets its own exception escape has not finished its job.

**Translation is not swallowing.** The original belongs on ``__cause__`` via
``raise ... from exc``, so nothing is lost and a traceback still names the real
failure. Categories exist so a caller can *decide*, not so detail can be thrown
away: ``RateLimitError`` is worth waiting on, ``InvalidRequestError`` is not, and
``ContextLimitExceededError`` is the one a compactor exists to prevent.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every failure crossing the provider boundary."""


class ProviderUnavailableError(LLMError):
    """The provider could not be reached, started, or resolved.

    Covers a local server that is not running, a host that does not answer, and
    a provider name nothing has registered. All of them mean "this provider
    cannot serve the request right now", which is what a caller acts on.
    """


class AuthenticationError(LLMError):
    """Credentials are missing, malformed, or rejected."""


class ModelUnavailableError(LLMError):
    """The provider is reachable but this model is not available on it.

    Distinct from ``ProviderUnavailableError`` because the fix is different:
    pull the model, or ask for one that exists.
    """


class InvalidRequestError(LLMError):
    """The request was rejected as malformed or unsatisfiable."""


class RateLimitError(LLMError):
    """The provider is refusing further requests for now.

    ``retry_after`` is seconds when the provider says so. Nothing in this phase
    acts on it; it is carried because discarding it at the boundary would mean
    it cannot be recovered later.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ContextLimitExceededError(LLMError):
    """The request exceeded the model's context window.

    The counts are what the provider reported, when it reported them, and are
    not filled in with an estimate: an estimate here would be indistinguishable
    from a fact about why the call failed.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_tokens: int | None = None,
        context_window: int | None = None,
    ) -> None:
        super().__init__(message)
        self.requested_tokens = requested_tokens
        self.context_window = context_window


class UnsupportedCapabilityError(LLMError):
    """The model or provider cannot do what was asked of it.

    Raised rather than approximated. Falling back from structured output to
    "ask nicely and hope for JSON" would make a capability flag a suggestion,
    and a caller who checked it would be misled.
    """

    def __init__(self, capability: str, *, provider: str = "", model: str = "") -> None:
        target = f"{provider}/{model}".strip("/")
        super().__init__(f"{target or 'provider'} does not support {capability}")
        self.capability = capability
        self.provider = provider
        self.model = model


class EmptyCompletionError(LLMError):
    """The model was cut off by the output cap before it emitted any content.

    Distinct from "the model chose to say nothing", and the distinction is the
    whole point. A reasoning model spends part of its output allowance thinking
    before it writes anything, and that spend is billed against the same
    ``max_output_tokens``. Set the cap below the thinking cost and the provider
    returns a well-formed response with a perfectly empty ``content``.

    Returning that as an empty string is what makes it dangerous: an empty
    summary is not obviously wrong to anything downstream, so it propagates as
    a real-looking measurement. A Phase 8.5 benchmark run scored every
    compaction arm near zero and the reference arm at 1.00 on exactly this —
    the arms that called the model got nothing, and the arm that did not was
    untouched. It read as a decisive result and was an empty context.

    ``reasoning_tokens`` is carried because it names the fix: raise the cap
    above what the model spends thinking, or use a model that does not.
    """

    def __init__(
        self,
        message: str,
        *,
        output_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens


class MalformedStructuredOutputError(LLMError):
    """The model returned something that is not the structure that was asked for.

    Carries the text it did return, truncated, because the only useful thing to
    say about an unparseable response is what it actually said.
    """

    def __init__(self, detail: str, *, text: str = "") -> None:
        super().__init__(detail)
        self.detail = detail
        self.text = text


class TokenizationUnavailableError(LLMError):
    """No tokenizer is available for this model.

    A real and common state: a local model whose tokenizer is not installed, or
    a hosted model that exposes no counting endpoint. It is an error rather than
    a silent fallback to an estimate, because a caller that wanted a number
    should have to choose an estimator knowingly.
    """


__all__ = [
    "AuthenticationError",
    "ContextLimitExceededError",
    "EmptyCompletionError",
    "InvalidRequestError",
    "LLMError",
    "MalformedStructuredOutputError",
    "ModelUnavailableError",
    "ProviderUnavailableError",
    "RateLimitError",
    "TokenizationUnavailableError",
    "UnsupportedCapabilityError",
]
