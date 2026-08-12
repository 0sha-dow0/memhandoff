"""Storage errors.

Distinct types so callers can tell a missing record from a broken reference from
a genuine bug, without matching on message text.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for every storage failure."""


class RecordNotFoundError(StorageError):
    """A record was requested by id and does not exist."""

    def __init__(self, record_id: str) -> None:
        super().__init__(f"no record with id {record_id!r}")
        self.record_id = record_id


class DuplicateRecordError(StorageError):
    """A record with this id is already stored.

    Records are immutable, so writing over one is never the intent. This is
    raised instead of silently replacing it.
    """

    def __init__(self, record_id: str) -> None:
        super().__init__(f"a record with id {record_id!r} is already stored")
        self.record_id = record_id


class DanglingReferenceError(StorageError):
    """A record refers to something that is not stored.

    Phase 1 could only check the format of a reference. The database checks that
    the target exists.
    """

    def __init__(self, source_id: str, reference: str) -> None:
        super().__init__(f"{source_id!r} refers to {reference!r}, which is not stored")
        self.source_id = source_id
        self.reference = reference


class SequenceConflictError(StorageError):
    """Two messages in one session claim the same position."""

    def __init__(self, session_id: str, seq: int) -> None:
        super().__init__(f"session {session_id!r} already has a message at seq {seq}")
        self.session_id = session_id
        self.seq = seq


class SerializationError(StorageError):
    """A record holds a value that cannot be written as JSON.

    ``metadata`` is typed ``dict[str, Any]``, so a value such as a set passes
    model validation and then fails here.
    """


class SchemaVersionError(StorageError):
    """The database was written by a different schema version."""

    def __init__(self, found: int, expected: int) -> None:
        super().__init__(
            f"database is at schema version {found}, this build expects {expected}. "
            "Run the migrations, or open a database created by a matching build."
        )
        self.found = found
        self.expected = expected


class CrossSessionReferenceError(StorageError):
    """A record refers to something in a different session.

    The session state graph is session-local. Knowledge that should outlive one
    session belongs in the memory layer, which is a later phase, rather than in
    a reference reaching sideways out of one session's history into another's.
    """

    def __init__(self, source_id: str, reference: str, expected: str, found: str) -> None:
        super().__init__(
            f"{source_id!r} in session {expected!r} refers to {reference!r}, "
            f"which belongs to session {found!r}"
        )
        self.source_id = source_id
        self.reference = reference
        self.expected_session = expected
        self.found_session = found


class SupersessionCycleError(StorageError):
    """A supersession chain would loop.

    Supersession says which record replaced which, so it has to be acyclic or
    there is no current state to resolve to.
    """

    def __init__(self, state_id: str) -> None:
        super().__init__(f"storing {state_id!r} would create a supersession cycle")
        self.state_id = state_id
