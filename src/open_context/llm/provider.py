"""The provider interface.

```
caller -> LLMProvider -> one provider implementation -> a model
```

Three methods. A provider names its model, generates text, and produces
structured data. Everything a specific vendor needs — base URLs, auth headers,
SDK objects, retries, wire formats — lives inside an implementation and is
invisible above it.

**The dependency direction is the point of this module.** Code above the
boundary imports ``LLMProvider`` and never a vendor SDK, so a compaction engine
written against this interface works with a hosted model, a local one, or a
deterministic fake, and contains no branch on which. A ``if provider ==
"openai"`` anywhere above the boundary means the abstraction has failed.

**Usage is what the provider reported.** ``GenerationResult.usage`` carries the
provider's own accounting, which is authoritative — it is what a bill is
computed from. It is absent when the provider said nothing, and it is never
filled in from a ``Tokenizer`` estimate: a number that might be measured or
might be guessed is worse than a missing one.

**Streaming is not in this interface, and there is no flag claiming otherwise.**
Nothing in the planned architecture needs it: compaction is a batch operation,
and streaming matters for interactive display, which this project does not do.
Designing a streaming contract now, against zero verified provider
implementations, would mean designing it twice. ``Capability`` deliberately has
no ``STREAMING`` member either, since a flag describing something the interface
gives no way to call is a claim a caller cannot act on. Recorded as a future
capability in docs/llm.md.

**Retries are not here either.** No backoff, no attempt budget, no circuit
breaker. ``RateLimitError`` carries ``retry_after`` when a provider supplies it,
which is the information a retry policy would need, and the policy itself waits
until something actually calls this in a loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from open_context.llm.info import ModelInfo
from open_context.llm.messages import ChatMessage


class FinishReason(StrEnum):
    """Why generation stopped.

    ``LENGTH`` is called out because a truncated answer that reads as complete
    is the failure mode a caller most needs to notice.
    """

    STOP = "stop"
    LENGTH = "length"
    FILTERED = "filtered"
    OTHER = "other"
    UNKNOWN = "unknown"


class TokenUsage(BaseModel):
    """Tokens the provider says the call consumed.

    Provider-reported and therefore authoritative, unlike a ``TokenCount`` from
    a ``Tokenizer``, which may be an estimate. The two are separate types so
    they cannot be confused for one another.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Output tokens the model spent thinking before writing, when the "
            "provider reports it. Already included in output_tokens — it is "
            "broken out, not added, because it is the part of the output "
            "allowance the caller never receives."
        ),
    )

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class GenerationRequest(BaseModel):
    """What to ask a model for.

    Only knobs that mean the same thing everywhere. A provider-specific setting
    goes in ``options``, where it is visibly the caller's decision to couple to
    one provider rather than something this interface blessed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0)
    stop: tuple[str, ...] = ()
    options: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific settings, passed through untouched.",
    )

    @classmethod
    def of(cls, *messages: ChatMessage, **kwargs: Any) -> GenerationRequest:  # noqa: ANN401
        return cls(messages=tuple(messages), **kwargs)


class GenerationResult(BaseModel):
    """What a model returned."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    text: str
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1, description="The model that actually answered.")
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: TokenUsage | None = None
    raw: Any = Field(
        default=None,
        description="The provider response as parsed, with unknown fields preserved.",
    )


class StructuredResult(GenerationResult):
    """A model response parsed into data.

    ``data`` is the parsed structure; ``text`` is what the model actually
    emitted, kept so a surprising result can be inspected rather than inferred.
    """

    data: Any = Field(default=None)


@runtime_checkable
class LLMProvider(Protocol):
    """Generates text and structured data from one model.

    An implementation must translate its own failures into ``LLMError``
    subclasses, and must raise ``UnsupportedCapabilityError`` rather than
    approximate a capability its model lacks.
    """

    def model_info(self) -> ModelInfo:
        """Describe the model this provider is configured for.

        A method rather than an attribute because a provider may have to ask
        the service what it is talking to, and may cache the answer.
        """

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce text."""

    def structured_output(
        self, request: GenerationRequest, schema: Mapping[str, Any]
    ) -> StructuredResult:
        """Produce data conforming to a JSON Schema.

        However the provider gets there — native structured output, a JSON
        mode, constrained decoding — the caller receives parsed data or
        ``MalformedStructuredOutputError``.
        """


def messages_of(request: GenerationRequest) -> Sequence[ChatMessage]:
    """The request's messages, for a tokenizer to count."""
    return request.messages


__all__ = [
    "FinishReason",
    "GenerationRequest",
    "GenerationResult",
    "LLMProvider",
    "StructuredResult",
    "TokenUsage",
    "messages_of",
]
