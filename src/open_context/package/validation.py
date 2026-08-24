"""Validating a `.ctx` package.

**Independent of whoever built it.** Validation takes a package and, optionally,
the archive it points at. It never consults the database the package was built
from, because the interesting case is a package that arrived from somewhere else
and the receiver has no such database. A validator that needed the producer's
state store would only ever confirm that a package matches itself.

**Two depths, and the difference is stated rather than blurred.** Without the
archive, a package can be checked for internal consistency: it hashes correctly,
its references point at items it contains, nothing claims a source it also omits.
With the archive, the claims about provenance can actually be tested — that the
cited records exist, and that they are the records the package hashed.

A package that passes the shallow check and fails the deep one is the dangerous
case: internally impeccable and describing a different archive. Saying "valid"
without naming which depth was reached would hide exactly that.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from open_context.archive.records import ArchiveRecord
from open_context.models.enums import StateStatus
from open_context.package.format import FORMAT_VERSION, STATE_SCHEMA_VERSION, CtxPackage


class Severity(StrEnum):
    """How much a finding costs the reader."""

    ERROR = "error"
    """The package cannot be trusted for the purpose it exists for."""

    WARNING = "warning"
    """Usable, with something the reader should know before relying on it."""


class Depth(StrEnum):
    """How far the check could go."""

    STRUCTURE = "structure"
    """Internal consistency only. The archive was not available."""

    PROVENANCE = "provenance"
    """Claims about the archive were tested against the archive."""


@dataclass(frozen=True)
class Finding:
    """One thing wrong, or worth knowing."""

    severity: Severity
    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.detail}"


@dataclass
class ValidationReport:
    """The verdict, and how far it was earned."""

    depth: Depth
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def valid(self) -> bool:
        """No errors. **Only as strong as ``depth``.**

        A structure-depth pass says nothing about whether the package describes
        the archive it names.
        """
        return not self.errors

    def __str__(self) -> str:
        verdict = "valid" if self.valid else "INVALID"
        head = (
            f"{verdict} ({self.depth} check, {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings)"
        )
        return "\n".join([head, *(f"  {finding}" for finding in self.findings)])


def validate(
    package: CtxPackage,
    *,
    archive_records: Mapping[int, ArchiveRecord] | None = None,
) -> ValidationReport:
    """Check a package, as deeply as the inputs allow."""
    depth = Depth.PROVENANCE if archive_records is not None else Depth.STRUCTURE
    report = ValidationReport(depth=depth)
    add = report.findings.append

    # -- the two fields a reader must trust before anything else -------
    if not package.manifest.readable:
        add(
            Finding(
                Severity.ERROR,
                "unreadable-version",
                f"package is format version {package.manifest.format_version}; this code "
                f"reads up to {FORMAT_VERSION}",
            )
        )

    if not package.manifest.state_readable:
        add(
            Finding(
                Severity.ERROR,
                "unreadable-state-schema",
                f"the package holds state schema version "
                f"{package.manifest.state_schema_version}; this code reads up to "
                f"{STATE_SCHEMA_VERSION}. The envelope is readable and its contents are not",
            )
        )

    if not package.intact():
        add(
            Finding(
                Severity.ERROR,
                "hash-mismatch",
                "content does not match content_hash; the package was altered after it "
                "was built, or was built by something that hashes differently",
            )
        )

    # -- internal consistency -----------------------------------------
    state_ids = {item.id for item in package.state}
    if len(state_ids) != len(package.state):
        add(Finding(Severity.ERROR, "duplicate-state", "the same state id appears more than once"))

    for item in package.state:
        if item.session_id != package.manifest.session_id:
            add(
                Finding(
                    Severity.ERROR,
                    "foreign-state",
                    f"{item.id} belongs to session {item.session_id}, not "
                    f"{package.manifest.session_id}",
                )
            )
        if item.status is StateStatus.SUPERSEDED:
            add(
                Finding(
                    Severity.ERROR,
                    "superseded-included",
                    f"{item.id} is superseded; a package presents what is believed now, and "
                    f"a reader given both sides of a reversal cannot tell which won",
                )
            )

    for reference in package.evidence:
        if reference.state_id not in state_ids:
            add(
                Finding(
                    Severity.ERROR,
                    "orphan-evidence",
                    f"evidence cites {reference.state_id}, which the package does not contain",
                )
            )

    described = {reference.state_id for reference in package.evidence}
    missing = state_ids - described
    if missing:
        add(
            Finding(
                Severity.WARNING,
                "state-without-evidence",
                f"{len(missing)} state items carry no evidence reference at all",
            )
        )

    unresolvable = [r.state_id for r in package.evidence if not r.resolvable]
    if unresolvable:
        add(
            Finding(
                Severity.WARNING,
                "unresolvable-provenance",
                f"{len(unresolvable)} state items name no source that could be followed; "
                f"their claims cannot be checked against the conversation",
            )
        )

    # -- supersession pointing outside the package ---------------------
    for item in package.state:
        target = getattr(item, "supersedes", None)
        if target and target not in state_ids:
            add(
                Finding(
                    Severity.WARNING,
                    "supersedes-absent",
                    f"{item.id} supersedes {target}, which is not in the package — expected, "
                    f"since superseded items are excluded, and recorded so a reader knows "
                    f"the item replaced something",
                )
            )

    if package.archive is None:
        add(
            Finding(
                Severity.WARNING,
                "no-archive-reference",
                "the package names no archive, so nothing in it can ever be traced to a source",
            )
        )

    if archive_records is None:
        return report

    # -- provenance, against the real archive --------------------------
    _check_against_archive(package, archive_records, add)
    return report


def _check_against_archive(
    package: CtxPackage,
    records: Mapping[int, ArchiveRecord],
    emit: Callable[[Finding], None],
) -> None:
    """Test the package's claims about the archive against that archive.

    This is the check that distinguishes a package describing *this* archive from
    one describing a similar archive. A hash mismatch here means the package is
    internally perfect and about something else.
    """
    reference = package.archive
    if reference is None:
        return

    if reference.session_id != package.manifest.session_id:
        emit(
            Finding(
                Severity.ERROR,
                "archive-session-mismatch",
                f"archive reference names session {reference.session_id}, package names "
                f"{package.manifest.session_id}",
            )
        )

    for seq, expected in sorted(reference.record_hashes.items()):
        record = records.get(seq)
        if record is None:
            emit(
                Finding(
                    Severity.ERROR,
                    "cited-record-missing",
                    f"the package cites archive record {seq}, which this archive does not have",
                )
            )
            continue
        if record.hash != expected:
            emit(
                Finding(
                    Severity.ERROR,
                    "cited-record-changed",
                    f"archive record {seq} hashes differently than when the package was "
                    f"built; this package describes a different archive, or that record "
                    f"was rewritten",
                )
            )

    for evidence in package.evidence:
        for seq in evidence.archive_seqs:
            if seq not in records:
                emit(
                    Finding(
                        Severity.WARNING,
                        "provenance-unreachable",
                        f"{evidence.state_id} cites archive record {seq}, absent from this archive",
                    )
                )


def format_report(report: ValidationReport) -> str:
    """The report as text, for a person."""
    return str(report)


__all__ = [
    "Depth",
    "Finding",
    "Severity",
    "ValidationReport",
    "format_report",
    "validate",
]
