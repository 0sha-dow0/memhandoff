"""What a compaction produced, and everything needed to judge it later.

**This is an internal experimental representation.** It is not the portable
context package, not `.ctx`, and not a MemHandoff package. It is what Phase 5
returns so Phase 5.5 has something to measure. Serializing one for a test or a
debugging session is fine; treating one as a durable interchange format is not.

The metadata is deliberately heavier than the output. A benchmark result whose
provenance is missing cannot be reproduced or attributed, and the questions
Phase 5.5 exists to answer are all of the form "which configuration produced
this, and how does it compare".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from open_context.importers import RawEvent


class TokenCountInfo(BaseModel):
    """A token count with its provenance, flattened for serialization.

    Mirrors ``llm.TokenCount``. It exists separately so a result can be written
    to JSON and read back without the LLM layer, and so ``exact`` survives that
    round trip: a stored number that lost its exactness flag would be indis-
    tinguishable from a measured one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(ge=0)
    exact: bool
    method: str


class TokenizerInfo(BaseModel):
    """Which tokenizer produced the counts on this result."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    provider: str
    model: str
    tokenizer: str | None = None
    exact_available: bool = False


class ModelUsageInfo(BaseModel):
    """Which model produced the summary, and what it reported."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    provider: str
    model: str
    context_window: int | None = None
    llm_calls: int = 0
    input_tokens_reported: int | None = None
    output_tokens_reported: int | None = None


class CompactionResult(BaseModel):
    """The outcome of one compaction, whatever strategy produced it.

    Named for the baseline until Phase 8, when a second strategy needed it and
    the name became wrong. The shape was always strategy-agnostic — the
    ``CompactionStrategy`` protocol has returned it for any implementation since
    Phase 5 — so this is a rename rather than a widening.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    # Identity and reproducibility
    session_id: str
    strategy: str
    strategy_version: int
    prompt_id: str | None = None
    prompt_hash: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)

    # The output
    state_section: str = ""
    """Structured state rendered as text, when a strategy produces any.

    Empty for every strategy that does not extract, which is what keeps a
    baseline result unchanged by this field existing. See Phase 8.
    """

    historical_summary: str = ""
    recent_events: tuple[RawEvent, ...] = ()

    # Budget
    target_tokens: int
    allocated_recent_tokens: int
    allocated_summary_tokens: int
    input_tokens: TokenCountInfo
    output_tokens: TokenCountInfo
    recent_tokens: TokenCountInfo
    summary_tokens: TokenCountInfo
    allocated_state_tokens: int = 0
    state_tokens: TokenCountInfo | None = None
    """What the state section cost. ``None`` when there is no state section,
    which is not the same as a state section that measured zero."""

    # What was considered
    archive_event_count: int = 0
    historical_event_count: int = 0
    recent_event_count: int = 0
    historical_range: tuple[int, int] | None = None
    """``[start, stop)`` archive positions summarized. None when nothing was."""

    recent_range: tuple[int, int] | None = None
    """``[start, stop)`` archive positions kept verbatim. None when the window is empty."""

    dropped_recent_event_count: int = 0
    """Events dropped from the recent window by the emergency fallback.

    These are **lost from the output entirely**, not condensed into the summary:
    the summary covers ``historical_range`` only, which ends where the recent
    window began. When this is non-zero the two ranges no longer meet, and the
    gap between ``historical_range[1]`` and ``recent_range[0]`` is exactly what
    is missing.
    """

    recent_event_ids: tuple[str, ...] = ()

    # What happened
    compaction_needed: bool = True
    fallbacks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    tokenizer: TokenizerInfo
    model: ModelUsageInfo
    elapsed_seconds: float = 0.0
    """Wall clock. The one field that will not reproduce between runs."""

    @property
    def fallback_used(self) -> bool:
        return bool(self.fallbacks)

    @property
    def within_budget(self) -> bool:
        return self.output_tokens.count <= self.target_tokens

    @property
    def lost_event_count(self) -> int:
        """Events represented nowhere in the output.

        Only the emergency fallback produces any. Everything else in the archive
        is either summarized or kept verbatim.
        """
        return self.dropped_recent_event_count

    @property
    def counts_exact(self) -> bool:
        """Whether the budget was enforced against measured counts.

        False means the output was sized with an estimate. The number may still
        be right; nothing here knows that it is.
        """
        return self.output_tokens.exact

    @property
    def compression_ratio(self) -> float:
        """Input tokens per output token.

        **Not a quality measure.** A summary that discards the one constraint
        that mattered compresses beautifully. Whether compression cost anything
        is what Phase 5.5 exists to find out.

        Zero output gives ``inf``: nothing was kept, which is an infinite ratio
        rather than an error.
        """
        if self.output_tokens.count == 0:
            return float("inf") if self.input_tokens.count else 1.0
        return self.input_tokens.count / self.output_tokens.count

    @property
    def retention_ratio(self) -> float:
        """Output tokens per input token. The reciprocal, with the same caveat."""
        if self.input_tokens.count == 0:
            return 1.0 if self.output_tokens.count == 0 else float("inf")
        return self.output_tokens.count / self.input_tokens.count

    def rendered(self) -> str:
        """Summary and recent events as one block of text.

        A convenience for inspection and for measuring the whole output. Not a
        package format and not an interchange format.
        """
        from open_context.compaction.rendering import render_events

        parts = []
        if self.state_section:
            parts.append(self.state_section)
        if self.historical_summary:
            parts.append(self.historical_summary)
        if self.recent_events:
            parts.append(render_events(self.recent_events))
        return "\n\n".join(parts)


BaselineCompactionResult = CompactionResult
"""The name this type had through Phase 7. Kept so existing callers still work."""


__all__ = [
    "BaselineCompactionResult",
    "CompactionResult",
    "ModelUsageInfo",
    "TokenCountInfo",
    "TokenizerInfo",
]
