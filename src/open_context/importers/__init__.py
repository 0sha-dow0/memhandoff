"""Conversation import adapters.

Gets conversation history from a provider's export into the archive without the
rest of the runtime learning which provider it came from.

```
provider export -> importer -> RawEvent -> archive -> later phases
```

Ingestion, not understanding. An importer answers "what happened in this
conversation". It does not answer "what does this conversation mean": no
summary, no extraction, no decisions, no tasks, no embeddings, no model call.
"We should use PostgreSQL" imports as a message saying that, and turning it into
a ``Decision`` is Phase 6's job.

Provider knowledge lives in adapters and nowhere else. The event model, the
pipeline, and the archive contain no branch on provider and no place to put one.
"""

from open_context.importers.base import ConversationImporter, ReadResult
from open_context.importers.claude_code import ClaudeCodeImporter, active_thread, segments
from open_context.importers.errors import (
    DuplicateSourceIdError,
    ImporterError,
    IncompleteImportError,
    MalformedRecordError,
    UnsupportedSourceError,
)
from open_context.importers.events import EventType, RawEvent
from open_context.importers.generic import (
    DEFAULT_ROLES,
    FieldMap,
    JsonLinesImporter,
    parse_timestamp,
)
from open_context.importers.pipeline import (
    FINGERPRINT_LENGTH,
    SYNTHETIC_PREFIX,
    DuplicatePolicy,
    MalformedPolicy,
    check_source_key,
    content_digest_for_bytes,
    content_digest_for_path,
    default_source_key,
    detect_importer,
    import_events,
    import_path,
    source_key_fingerprint,
    stream_events,
    synthetic_id,
)
from open_context.importers.registry import DEFAULT_IMPORTERS
from open_context.importers.report import (
    SAMPLE_LIMIT,
    ImportReport,
    MalformedRecord,
)

__all__ = [
    "DEFAULT_IMPORTERS",
    "DEFAULT_ROLES",
    "FINGERPRINT_LENGTH",
    "SAMPLE_LIMIT",
    "SYNTHETIC_PREFIX",
    "ClaudeCodeImporter",
    "ConversationImporter",
    "DuplicatePolicy",
    "DuplicateSourceIdError",
    "EventType",
    "FieldMap",
    "ImportReport",
    "ImporterError",
    "IncompleteImportError",
    "JsonLinesImporter",
    "MalformedPolicy",
    "MalformedRecord",
    "MalformedRecordError",
    "RawEvent",
    "ReadResult",
    "UnsupportedSourceError",
    "active_thread",
    "check_source_key",
    "content_digest_for_bytes",
    "content_digest_for_path",
    "default_source_key",
    "detect_importer",
    "import_events",
    "import_path",
    "parse_timestamp",
    "segments",
    "source_key_fingerprint",
    "stream_events",
    "synthetic_id",
]
