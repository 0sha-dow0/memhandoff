"""Hybrid compaction: structured state plus recent context plus history.

Phase 8. Composes `compaction` and `extraction`, which are peers, so it sits
above both rather than inside either.

```
RAW ARCHIVE -> {STATE, RECENT, HISTORY} -> BUDGET -> PORTABLE CONTEXT
```

It exists to fix one measured failure. Phase 5.8 found the Phase 5 baseline and
a plain one-call summary scoring identically, and found that what compaction
lost was **exact values** — a port number among similar ports, a row count from
a tool result — which full context kept. Structured state is the mechanism that
can carry an exact value through compaction intact.
"""

from open_context.hybrid.compactor import (
    STATE_ORDER,
    HybridCompaction,
    HybridConfig,
    fit_state,
    render_state,
    state_of,
)

__all__ = [
    "STATE_ORDER",
    "HybridCompaction",
    "HybridConfig",
    "fit_state",
    "render_state",
    "state_of",
]
