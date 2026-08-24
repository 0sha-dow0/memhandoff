"""Counting tokens, and being honest about it.

**The invariant: an estimate is never presented as an exact count.**

Every count carries how it was produced. That is not decoration. A compactor
that fills a 128,000 token window to 127,900 on the strength of a character
heuristic will overflow, and the failure will look like a provider bug rather
than a measurement one. Knowing a number is approximate is what lets a caller
leave headroom.

The invariant is enforced by arithmetic rather than by discipline: adding an
exact count to an estimated one yields an estimated count, because a total is
only as trustworthy as its worst part. There is no way to add your way back to
``exact=True``.

**Tokenization is separate from generation.** Not every provider exposes an
exact tokenizer, and some expose none at all. A count may be local or remote,
model-specific or generic, exact or approximate, or simply unavailable — so
``Tokenizer`` is its own interface rather than a method on ``LLMProvider``, and
a provider is free to hand back a tokenizer it did not implement.

**Messages cost more than their text.** A chat model wraps each message in
role markers and delimiters defined by its own template. A tokenizer that does
not know that template cannot count the framing exactly, which is why
``count_messages`` may be estimated even where ``count_text`` is exact. Saying
so is more useful than a number that is quietly short by a few tokens per turn.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from open_context.llm.errors import TokenizationUnavailableError
from open_context.llm.info import Capability, ModelInfo
from open_context.llm.messages import ChatMessage

MIXED_METHOD = "mixed"
"""``TokenCount.method`` once counts from different methods have been combined."""


@dataclass(frozen=True)
class TokenCount:
    """A number of tokens, and how much that number can be trusted.

    ``method`` names the technique, such as a tokenizer's encoding name or the
    heuristic used. It is for reporting and debugging; ``exact`` is what code
    should branch on.
    """

    count: int
    exact: bool
    method: str

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError(f"token count must not be negative, got {self.count}")
        if not self.method:
            raise ValueError("token count must say how it was produced")

    def __add__(self, other: TokenCount) -> TokenCount:
        """Combine two counts, keeping the weaker guarantee.

        Exactness does not survive contact with an estimate. This is the whole
        enforcement mechanism for the invariant at the top of this module, so it
        lives in the type rather than in every caller's head.
        """
        if not isinstance(other, TokenCount):
            return NotImplemented
        return TokenCount(
            count=self.count + other.count,
            exact=self.exact and other.exact,
            method=self.method if self.method == other.method else MIXED_METHOD,
        )

    def __str__(self) -> str:
        return f"{self.count} tokens ({'exact' if self.exact else 'estimated'}, {self.method})"

    @classmethod
    def total(cls, counts: Iterable[TokenCount]) -> TokenCount:
        """Sum counts. Empty is zero, exactly, by no method in particular."""
        running = cls(count=0, exact=True, method="empty")
        seen = False
        for count in counts:
            running = count if not seen else running + count
            seen = True
        return running


@runtime_checkable
class Tokenizer(Protocol):
    """Counts tokens for one model.

    Implementations must never round an estimate into ``exact=True``, and must
    raise ``TokenizationUnavailableError`` rather than fall back to a guess the
    caller did not ask for.
    """

    def model_info(self) -> ModelInfo:
        """The model this tokenizer counts for, including its context window."""

    def count_text(self, text: str) -> TokenCount:
        """Tokens in a plain string."""

    def count_messages(self, messages: Sequence[ChatMessage]) -> TokenCount:
        """Tokens in a sequence of messages, including framing where known."""


class UnavailableTokenizer:
    """A tokenizer for a model whose tokenizer is not available.

    Exists so "we cannot count this" is a real object that can be passed around
    and handled, rather than a ``None`` that every call site has to remember to
    check. Every count raises, with a reason.
    """

    def __init__(self, info: ModelInfo, reason: str = "no tokenizer is available") -> None:
        self._info = info
        self.reason = reason

    def model_info(self) -> ModelInfo:
        return self._info

    def _fail(self) -> TokenCount:
        raise TokenizationUnavailableError(
            f"{self._info.provider}/{self._info.model}: {self.reason}"
        )

    def count_text(self, text: str) -> TokenCount:
        return self._fail()

    def count_messages(self, messages: Sequence[ChatMessage]) -> TokenCount:
        return self._fail()


class CharacterRatioTokenizer:
    """Estimates tokens from character count. Never exact, and says so.

    A deliberately crude fallback for a model whose real tokenizer is not
    installed or not exposed. The ratio is an English-prose rule of thumb; it
    runs long on code, punctuation, and non-Latin scripts, and it knows nothing
    about any model's chat template. It is offered because a rough number a
    caller knows is rough beats no number at all, and because it makes the
    estimated path testable without shipping a tokenizer library.

    ``message_overhead`` is an allowance for the role markers and delimiters a
    chat template adds per message. It is a guess, which is one more reason
    nothing this class returns is ever exact.
    """

    def __init__(
        self,
        info: ModelInfo,
        *,
        characters_per_token: float = 4.0,
        message_overhead: int = 4,
    ) -> None:
        if characters_per_token <= 0:
            raise ValueError("characters_per_token must be positive")
        if message_overhead < 0:
            raise ValueError("message_overhead must not be negative")
        self._info = info
        self.characters_per_token = characters_per_token
        self.message_overhead = message_overhead
        self.method = f"chars/{characters_per_token:g}"

    def model_info(self) -> ModelInfo:
        return self._info

    def _estimate(self, characters: int, overhead: int = 0) -> TokenCount:
        return TokenCount(
            count=math.ceil(characters / self.characters_per_token) + overhead,
            exact=False,
            method=self.method,
        )

    def count_text(self, text: str) -> TokenCount:
        return self._estimate(len(text))

    def count_messages(self, messages: Sequence[ChatMessage]) -> TokenCount:
        characters = sum(len(m.content) + len(m.role) + len(m.name or "") for m in messages)
        return self._estimate(characters, overhead=self.message_overhead * len(messages))


def exact_counting_available(tokenizer: Tokenizer) -> bool:
    """Whether this tokenizer's model claims exact counting.

    A convenience over the capability flag. It reads the model's claim; whether
    a given count is exact is still answered by that count's own ``exact``.
    """
    return tokenizer.model_info().supports(Capability.EXACT_TOKEN_COUNT)


__all__ = [
    "MIXED_METHOD",
    "CharacterRatioTokenizer",
    "TokenCount",
    "Tokenizer",
    "UnavailableTokenizer",
    "exact_counting_available",
]
