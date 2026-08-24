"""Baseline compaction.

```
archive -> historical summary + verbatim recent window -> baseline context
```

**Phase 5 is a baseline intended for measurement. It is not the final MemHandoff
compaction algorithm.**

It is roughly what a competent developer would build in an afternoon: summarize
the older part of a session with a model, keep the most recent part verbatim,
and hold the whole thing to a token budget. That is the point. It is the control
that later work has to beat, and the places where it loses a constraint stated
once in message 40 are the evidence that later work is needed.

What it deliberately does not do: extract structure, rank importance, retrieve,
embed, resolve contradictions, or produce a portable package. A conversation
saying "we chose Postgres because the infra already runs it" yields a summary
that says so, and never a ``Decision`` record.

Reads the archive and never writes it. Reaches a model only through
``LLMProvider`` and counts only through ``Tokenizer``, so it works offline
against the deterministic fakes and knows nothing about any vendor.
"""

from open_context.compaction.baseline import (
    FALLBACK_CHUNK_SUMMARIES_TRUNCATED,
    FALLBACK_MULTI_PASS,
    FALLBACK_RECENT_DROPPED,
    FALLBACK_SUMMARY_RETRY,
    FALLBACK_SUMMARY_TRUNCATED,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    WARN_ESTIMATED_COUNTS,
    WARN_NO_HISTORY,
    WARN_RECENT_DROPPED_LOST,
    WARN_UNKNOWN_WINDOW,
    BaselineRecentPlusSummary,
)
from open_context.compaction.budget import (
    MINIMUM_TARGET_TOKENS,
    BaselineConfig,
    BudgetAllocation,
    allocate,
)
from open_context.compaction.errors import (
    CompactionError,
    InvalidConfigurationError,
    TargetTooSmallError,
)
from open_context.compaction.prompts import BASELINE_SUMMARY_V1, Prompt
from open_context.compaction.rendering import render_event, render_events, to_message
from open_context.compaction.result import (
    BaselineCompactionResult,
    CompactionResult,
    ModelUsageInfo,
    TokenCountInfo,
    TokenizerInfo,
)
from open_context.compaction.strategy import CompactionRequest, CompactionStrategy
from open_context.compaction.window import (
    WINDOW_EMPTY_OVERSIZED,
    RecentWindow,
    count_conversation,
    select_recent_window,
    stream_archive,
)

__all__ = [
    "BASELINE_SUMMARY_V1",
    "FALLBACK_CHUNK_SUMMARIES_TRUNCATED",
    "FALLBACK_MULTI_PASS",
    "FALLBACK_RECENT_DROPPED",
    "FALLBACK_SUMMARY_RETRY",
    "FALLBACK_SUMMARY_TRUNCATED",
    "MINIMUM_TARGET_TOKENS",
    "STRATEGY_NAME",
    "STRATEGY_VERSION",
    "WARN_ESTIMATED_COUNTS",
    "WARN_NO_HISTORY",
    "WARN_RECENT_DROPPED_LOST",
    "WARN_UNKNOWN_WINDOW",
    "WINDOW_EMPTY_OVERSIZED",
    "BaselineCompactionResult",
    "BaselineConfig",
    "BaselineRecentPlusSummary",
    "BudgetAllocation",
    "CompactionError",
    "CompactionRequest",
    "CompactionResult",
    "CompactionStrategy",
    "InvalidConfigurationError",
    "ModelUsageInfo",
    "Prompt",
    "RecentWindow",
    "TargetTooSmallError",
    "TokenCountInfo",
    "TokenizerInfo",
    "allocate",
    "count_conversation",
    "render_event",
    "render_events",
    "select_recent_window",
    "stream_archive",
    "to_message",
]
