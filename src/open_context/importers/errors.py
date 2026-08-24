"""Import errors.

Named ``ImporterError`` rather than ``ImportError`` on purpose: the builtin of
that name means something else entirely, and shadowing it in a package called
``importers`` would be a trap for every reader after this one.

**Every failure that can stop a running import carries the partial report.**
An import is not atomic, so the interesting question after a failure is never
"what went wrong" alone; it is "what is in my archive now". ``IncompleteImportError``
exists so that question is answered by the exception itself rather than by
guessing, and so a caller cannot catch the specific failure without meeting the
type whose name says the import stopped partway.
"""

from __future__ import annotations

from open_context.importers.report import ImportReport, MalformedRecord


class ImporterError(Exception):
    """Base class for import failures."""


class UnsupportedSourceError(ImporterError):
    """No importer recognises this source, or it cannot be read at all.

    Raised before anything is written, so the destination is untouched.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class IncompleteImportError(ImporterError):
    """An import stopped partway. Records written before it remain in the archive.

    The archive is append-only: it has ``append`` and ``extend`` and no delete,
    no truncate, and no rollback. Nothing here can take a written record back,
    so the honest contract is to say exactly what landed. ``report`` is the same
    ``ImportReport`` a successful import returns, with ``completed=False``, and
    ``report.last_seq`` is the position of the final record that was written.

    The record that caused the failure is never among them.
    """

    def __init__(self, message: str, report: ImportReport) -> None:
        super().__init__(f"{message}; {report.events_written} records were already written")
        self.report = report


class MalformedRecordError(IncompleteImportError):
    """A source record could not be read, and the import stopped.

    Carries the record itself so the caller can locate the problem in the source
    file rather than being told only that something, somewhere, failed.
    """

    def __init__(self, record: MalformedRecord, report: ImportReport) -> None:
        super().__init__(f"{record.location}: {record.detail}", report)
        self.record = record

    @property
    def position(self) -> int:
        return self.record.position

    @property
    def location(self) -> str:
        return self.record.location


class DuplicateSourceIdError(IncompleteImportError):
    """A source id appeared more than once, and the import stopped.

    Not an error by default. The archive stores two records sharing an id as two
    distinct records, which is correct for a message and a later revision of it,
    so a duplicate is reported rather than refused unless the caller asks.
    """

    def __init__(self, source_id: str, position: int, report: ImportReport) -> None:
        super().__init__(f"source id {source_id!r} appears again at position {position}", report)
        self.source_id = source_id
        self.position = position


__all__ = [
    "DuplicateSourceIdError",
    "ImporterError",
    "IncompleteImportError",
    "MalformedRecordError",
    "UnsupportedSourceError",
]
