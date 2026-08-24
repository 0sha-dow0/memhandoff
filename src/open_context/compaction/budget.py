"""Splitting a token budget.

Fixed fractions, computed once. There is no adaptive allocation here, no
importance model, and nothing that learns: those are later research, and a
baseline whose allocation moved by itself would be impossible to compare against.

The allocation is reported on every result, so a benchmark can attribute an
outcome to the split that produced it rather than to the default in effect on
the day it ran.
"""

from __future__ import annotations

from dataclasses import dataclass

from open_context.compaction.errors import InvalidConfigurationError, TargetTooSmallError

MINIMUM_TARGET_TOKENS = 50
"""Below this, neither half of the output is worth producing.

Not a tuned number. It is the point at which a summary budget of a few dozen
tokens stops being a summary of anything, and the honest answer is to refuse
rather than return a sentence fragment that looks like a compaction.
"""


@dataclass(frozen=True)
class BaselineConfig:
    """Everything the baseline can be tuned by.

    Small on purpose. Each field is something a benchmark may want to vary; a
    knob nothing will be swept over is a knob that only makes results harder to
    compare.
    """

    recent_fraction: float = 0.40
    """Share of the target spent on verbatim recent events, the rest on the summary."""

    minimum_target_tokens: int = MINIMUM_TARGET_TOKENS

    prompt_reserve_tokens: int = 512
    """Room set aside for the instruction prompt when sizing a model request."""

    context_fill_fraction: float = 0.85
    """How much of a model's context window a single request may occupy.

    Below one because our count of the input is not the provider's count of it,
    and the margin is what absorbs that disagreement.
    """

    summary_retry_fraction: float = 0.70
    """How much smaller to ask for when a summary came back over its budget."""

    truncation_marker: str = " [...truncated]"
    """Appended when text had to be cut, so a reader can see that it was."""

    def __post_init__(self) -> None:
        if not 0.0 < self.recent_fraction < 1.0:
            raise InvalidConfigurationError(
                f"recent_fraction must be between 0 and 1 exclusive, got {self.recent_fraction}"
            )
        if not 0.0 < self.context_fill_fraction <= 1.0:
            raise InvalidConfigurationError(
                f"context_fill_fraction must be in (0, 1], got {self.context_fill_fraction}"
            )
        if not 0.0 < self.summary_retry_fraction < 1.0:
            raise InvalidConfigurationError(
                f"summary_retry_fraction must be between 0 and 1 exclusive, "
                f"got {self.summary_retry_fraction}"
            )
        if self.minimum_target_tokens < 1:
            raise InvalidConfigurationError("minimum_target_tokens must be positive")
        if self.prompt_reserve_tokens < 0:
            raise InvalidConfigurationError("prompt_reserve_tokens must not be negative")


@dataclass(frozen=True)
class BudgetAllocation:
    """How a target was divided."""

    target_tokens: int
    recent_tokens: int
    summary_tokens: int
    recent_fraction: float

    def __post_init__(self) -> None:
        if self.recent_tokens + self.summary_tokens > self.target_tokens:
            raise InvalidConfigurationError(
                "allocation exceeds its own target; this is a bug in allocate()"
            )


def allocate(target_tokens: int, config: BaselineConfig) -> BudgetAllocation:
    """Divide a target between recent events and the historical summary.

    Both halves are guaranteed at least one token, so no configuration produces
    a half that cannot hold anything; ``TargetTooSmallError`` is what happens
    when the target is too small for that to be meaningful.
    """
    if target_tokens < config.minimum_target_tokens:
        raise TargetTooSmallError(target_tokens, config.minimum_target_tokens)

    recent = max(1, int(target_tokens * config.recent_fraction))
    summary = max(1, target_tokens - recent)
    if recent + summary > target_tokens:
        recent = target_tokens - summary
    return BudgetAllocation(
        target_tokens=target_tokens,
        recent_tokens=recent,
        summary_tokens=summary,
        recent_fraction=config.recent_fraction,
    )


__all__ = [
    "MINIMUM_TARGET_TOKENS",
    "BaselineConfig",
    "BudgetAllocation",
    "allocate",
]
