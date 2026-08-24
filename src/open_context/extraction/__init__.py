"""Structured context extraction: conversation to typed work state.

Phase 6. The first layer that interprets rather than transports.

```
RAW ARCHIVE -> StructuredExtractor -> Goal / Constraint / Decision / Task / ...
```

Import answers "what happened". Compaction answers "what should be kept".
Extraction answers "what does it mean" — that this sentence is a decision, that
one a prohibition, and that the second overturns the first.

What it does not do: write to storage, reconcile against existing state, resolve
conflicts, or process a session incrementally. Those need supersession and
conflict machinery that does not exist, and belong to Phase 7.
"""

from open_context.extraction.extractor import (
    ExtractionConfig,
    StructuredExtractor,
    parse_json_object,
)
from open_context.extraction.prompts import EXTRACTION_PROMPT_V1
from open_context.extraction.result import (
    MAX_SAMPLES,
    ExtractedItem,
    ExtractionReport,
    ExtractionResult,
    RejectedItem,
)
from open_context.extraction.validation import (
    MIN_OVERLAP,
    Finding,
    Severity,
    format_findings,
    validate,
)

__all__ = [
    "EXTRACTION_PROMPT_V1",
    "MAX_SAMPLES",
    "MIN_OVERLAP",
    "ExtractedItem",
    "ExtractionConfig",
    "ExtractionReport",
    "ExtractionResult",
    "Finding",
    "RejectedItem",
    "Severity",
    "StructuredExtractor",
    "format_findings",
    "parse_json_object",
    "validate",
]
