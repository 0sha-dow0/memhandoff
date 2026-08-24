"""What a model is and what it can do.

Provider-neutral, so that code above the boundary can ask "how much room is
there" and "can this do structured output" without knowing whose model answered.

**Unknown is a value.** ``context_window`` and ``max_output_tokens`` are
``None`` when the provider did not say. Filling in a plausible 128,000 would be
worse than admitting ignorance: a budget computed from an invented window is a
number that looks authoritative and is wrong, and the failure surfaces much
later as a truncated request nobody can explain.

**No pricing.** Deliberately. Prices change without notice, vary by region and
contract, and are not provider-neutral in any useful way, so a table here would
be stale before it was read. This phase is about token and context accounting,
not billing, and the two are separable.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Capability(StrEnum):
    """Something a model can do that a caller might have to check first.

    Only capabilities this project will act on. A flag nothing branches on is a
    claim nobody verifies, so the list stays short and grows when a phase needs
    it.

    There is deliberately no ``STREAMING``. ``LLMProvider`` exposes no streaming
    method, so the flag would have described a capability the interface offers no
    way to use, and a caller checking it could do nothing with the answer. It
    returns when there is a streaming call to guard. See docs/llm.md.
    """

    TEXT_GENERATION = "text_generation"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    EXACT_TOKEN_COUNT = "exact_token_count"


class ModelInfo(BaseModel):
    """A provider-neutral description of one model."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    provider: str = Field(min_length=1, description="Registered provider name, such as 'fake'.")
    model: str = Field(min_length=1, description="The provider's own name for the model.")
    context_window: int | None = Field(
        default=None, gt=0, description="Total tokens the model accepts. None when unknown."
    )
    max_output_tokens: int | None = Field(
        default=None, gt=0, description="Cap on generated tokens. None when unknown."
    )
    tokenizer: str | None = Field(
        default=None,
        description="Identity of the tokenizer this model uses, when known. None otherwise.",
    )
    capabilities: frozenset[Capability] = Field(default_factory=frozenset)

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    @property
    def counts_exactly(self) -> bool:
        """Whether an exact token count is obtainable for this model.

        A shorthand for the capability, because it is the one every budgeting
        decision will have to consult.
        """
        return self.supports(Capability.EXACT_TOKEN_COUNT)


__all__ = ["Capability", "ModelInfo"]
