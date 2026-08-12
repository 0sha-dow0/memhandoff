"""Local immutable archive.

The durable record of what was actually said. Append-only JSONL on the local
filesystem, one log per session, with a rebuildable offset index.

This is the source material compaction reads from. SQLite holds indexed metadata
and interpreted structure; the archive holds the raw bytes those were derived
from. The conversation is not duplicated into SQLite.
"""

from open_context.archive.errors import (
    ArchiveError,
    ArchiveExistsError,
    ArchiveFormatError,
    ArchiveNotFoundError,
    CorruptRecordError,
    DuplicateSeqError,
    RecordNotFoundError,
)
from open_context.archive.index import ENTRY_SIZE, OffsetIndex
from open_context.archive.log import IntegrityReport, RecoveryReport, SessionLog
from open_context.archive.records import (
    FORMAT,
    FORMAT_VERSION,
    ArchiveRecord,
    canonical_json,
    hash_body,
)
from open_context.archive.store import Archive

__all__ = [
    "ENTRY_SIZE",
    "FORMAT",
    "FORMAT_VERSION",
    "Archive",
    "ArchiveError",
    "ArchiveExistsError",
    "ArchiveFormatError",
    "ArchiveNotFoundError",
    "ArchiveRecord",
    "CorruptRecordError",
    "DuplicateSeqError",
    "IntegrityReport",
    "OffsetIndex",
    "RecordNotFoundError",
    "RecoveryReport",
    "SessionLog",
    "canonical_json",
    "hash_body",
]
