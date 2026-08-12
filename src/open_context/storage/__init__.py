"""SQLite persistence for the Phase 1 data model.

Append and read only. Records are immutable, so there is no update and no
delete: a change is a new record whose ``supersedes`` points at the old one.
"""

from open_context.storage.database import (
    MEMORY,
    Database,
    current_version,
    migrate,
    split_statements,
)
from open_context.storage.errors import (
    CrossSessionReferenceError,
    DanglingReferenceError,
    DuplicateRecordError,
    RecordNotFoundError,
    SchemaVersionError,
    SequenceConflictError,
    SerializationError,
    StorageError,
    SupersessionCycleError,
)
from open_context.storage.repository import Repository
from open_context.storage.schema import MIGRATIONS, SCHEMA_VERSION, latest_version

__all__ = [
    "MEMORY",
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "CrossSessionReferenceError",
    "DanglingReferenceError",
    "Database",
    "DuplicateRecordError",
    "RecordNotFoundError",
    "Repository",
    "SchemaVersionError",
    "SequenceConflictError",
    "SerializationError",
    "StorageError",
    "SupersessionCycleError",
    "current_version",
    "latest_version",
    "migrate",
    "split_statements",
]
