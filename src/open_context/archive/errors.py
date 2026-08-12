"""Archive errors."""

from __future__ import annotations


class ArchiveError(Exception):
    """Base class for archive failures."""


class ArchiveNotFoundError(ArchiveError):
    """No archive exists at the requested location."""

    def __init__(self, name: str) -> None:
        super().__init__(f"no archive for {name!r}")
        self.name = name


class ArchiveExistsError(ArchiveError):
    """An archive already exists and would be overwritten.

    The archive is append-only, so replacing one is never the intent.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"an archive for {name!r} already exists")
        self.name = name


class RecordNotFoundError(ArchiveError):
    """No record at that position."""

    def __init__(self, seq: int) -> None:
        super().__init__(f"no record at seq {seq}")
        self.seq = seq


class DuplicateSeqError(ArchiveError):
    """A record was appended at a position that is already written.

    Positions are assigned by the archive and increase by one. This means the
    caller tried to rewrite history.
    """

    def __init__(self, seq: int) -> None:
        super().__init__(f"seq {seq} is already written; the archive is append-only")
        self.seq = seq


class CorruptRecordError(ArchiveError):
    """A record does not match its stored hash, or cannot be parsed.

    Distinct from a truncated tail, which is recoverable. This is damage in the
    body of the archive and is never repaired silently.
    """

    def __init__(self, seq: int, detail: str) -> None:
        super().__init__(f"record at seq {seq} is corrupt: {detail}")
        self.seq = seq
        self.detail = detail


class ArchiveFormatError(ArchiveError):
    """The archive was written in a format this build does not understand."""

    def __init__(self, found: str, expected: str) -> None:
        super().__init__(f"archive format is {found!r}, this build expects {expected!r}")
        self.found = found
        self.expected = expected
