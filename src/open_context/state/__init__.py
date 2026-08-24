"""State and retention management: durable, incrementally updateable context.

Phase 7. What makes extracted state survive a long session rather than being
rebuilt from scratch on every change.

```
OLD STATE  +  NEW EVENTS  ->  UPDATED STATE
```

Three things carry the phase. **Nothing is deleted**: an overturned item is
marked superseded and keeps its place, because the record that a decision
changed is part of the state. **Reconciliation returns a plan, not an effect**,
so an update carrying a conflict can be refused before it is half-applied.
**A watermark is earned**: it advances only when an extraction actually
succeeded, since one moved on a failure would skip a region of the conversation
permanently and leave nothing to say so.
"""

from open_context.state.conflict import (
    Conflict,
    ConflictKind,
    detect_conflicts,
    format_conflicts,
)
from open_context.state.incremental import (
    IncrementalUpdate,
    Watermark,
    extract_incremental,
    pending,
)
from open_context.state.reconcile import (
    Reconciliation,
    apply_supersession,
    fingerprint,
    reconcile,
)

__all__ = [
    "Conflict",
    "ConflictKind",
    "IncrementalUpdate",
    "Reconciliation",
    "Watermark",
    "apply_supersession",
    "detect_conflicts",
    "extract_incremental",
    "fingerprint",
    "format_conflicts",
    "pending",
    "reconcile",
]
