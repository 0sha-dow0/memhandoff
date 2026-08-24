"""Compaction errors.

Failures a caller can act on. Everything the baseline can survive is a warning
in the result instead, because a compaction that quietly did something different
from what was asked is worse than one that stopped.
"""

from __future__ import annotations


class CompactionError(Exception):
    """Base class for compaction failures."""


class TargetTooSmallError(CompactionError):
    """The token budget is below the smallest representation worth producing.

    Raised rather than returning a result, because at a few dozen tokens the
    output is neither a usable summary nor a usable recent window, and returning
    one anyway would imply the compaction succeeded.
    """

    def __init__(self, target_tokens: int, minimum: int) -> None:
        super().__init__(
            f"target of {target_tokens} tokens is below the minimum of {minimum}; "
            "there is no useful representation at that size"
        )
        self.target_tokens = target_tokens
        self.minimum = minimum


class InvalidConfigurationError(CompactionError):
    """A configuration value that cannot produce a coherent allocation."""


__all__ = ["CompactionError", "InvalidConfigurationError", "TargetTooSmallError"]
