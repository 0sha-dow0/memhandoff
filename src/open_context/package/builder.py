"""Building a `.ctx` package from a session.

**Nothing here interprets.** State items arrive already extracted and already
reconciled; this assembles them, resolves what provenance it can, and records
what it could not. A builder that filtered or rewrote state would be a second,
undocumented compaction step sitting between the state store and the artifact
that claims to represent it.

**Superseded items are excluded, and the exclusion is counted.** A package is
what is believed *now*: handing a receiving agent both sides of a reversal
without saying which won leaves it worse off than handing it neither. The count
survives in the manifest metadata so a reader can see that history existed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from open_context.archive.records import FORMAT as ARCHIVE_FORMAT
from open_context.archive.records import FORMAT_VERSION as ARCHIVE_FORMAT_VERSION
from open_context.archive.records import ArchiveRecord
from open_context.models import ids
from open_context.models.enums import StateStatus
from open_context.models.state import StateItem
from open_context.package.format import (
    ArchiveReference,
    CtxPackage,
    EvidenceReference,
    Manifest,
    RecentContext,
)

EXCERPT_LIMIT = 240
"""Characters of source text carried per evidence reference.

Long enough to show what an item rests on, short enough that evidence does not
become the archive by accident. A package whose excerpts outweigh its state has
inlined the history it was designed to reference.
"""


@dataclass
class PackageBuildReport:
    """What the build did, including what it could not do.

    Returned beside the package rather than logged. A caller deciding whether to
    hand this artifact to another agent needs to know that half its state has no
    resolvable provenance, and a warning printed to a terminal somewhere does not
    reach that decision.
    """

    state_items: int = 0
    superseded_excluded: int = 0
    evidence_resolved: int = 0
    evidence_unresolved: int = 0
    recent_messages: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def fully_traceable(self) -> bool:
        """Whether every included item can be traced back to a source."""
        return self.state_items > 0 and self.evidence_unresolved == 0


def _sources_to_seqs(item: StateItem, by_record_id: dict[str, int]) -> tuple[int, ...]:
    """Resolve an item's ``sources`` against the archive, by id.

    **This is a lookup, not a guess, and the distinction is the whole reason it
    is allowed here.** Phase 6 recorded that ``sources`` speaks the SQLite
    ``msg_``/``ev_`` scheme while the archive numbers its records by sequence,
    and concluded that no correspondence had been established. That is true of
    *sequence numbers*. It is not true of ids: ``ArchiveRecord.id`` is defined as
    "identity of the underlying item, preserved as given", so an archive fed the
    same messages the state was extracted from already holds those very ids.

    So the match is on an identifier both sides agreed on, and a source naming
    something this archive does not contain simply does not resolve. Nothing is
    minted, and an id that means something else somewhere cannot collide — the
    prefix scheme exists to make ids self-describing.

    An explicit ``provenance`` mapping still wins, because a caller who tracked
    the derivation knows more than this inference does.
    """
    return tuple(
        sorted({by_record_id[source] for source in item.sources if source in by_record_id})
    )


def _excerpt(record: ArchiveRecord) -> str:
    """A short readable quotation from a record, if it has readable text."""
    payload = record.payload
    text = ""
    if isinstance(payload, dict):
        for key in ("content", "text", "body"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = value
                break
    elif isinstance(payload, str):
        text = payload
    text = " ".join(text.split())
    return text[:EXCERPT_LIMIT]


def build(
    *,
    session_id: str,
    state: Iterable[StateItem],
    inherited: Sequence[StateItem] = (),
    package_id: str | None = None,
    title: str | None = None,
    source: str | None = None,
    created_by: str = "",
    generator_version: str = "",
    created_at: datetime | None = None,
    archive_records: Sequence[ArchiveRecord] | None = None,
    archive_reference: ArchiveReference | None = None,
    archive_location: str = "",
    recent: Sequence[dict[str, Any]] | None = None,
    recent_reason: str = "",
    provenance: dict[str, Sequence[int]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[CtxPackage, PackageBuildReport]:
    """Assemble a package, and report what the assembly could not establish.

    ``provenance`` maps a state item id to the archive sequence numbers it was
    derived from, for a caller that tracked the derivation. It is optional:
    where it says nothing, ``sources`` is resolved against the archive by record
    id, which is a lookup rather than an inference — see ``_sources_to_seqs``.

    An item neither route can place is packaged with an empty reference and
    counted. That is the truthful outcome; minting a plausible archive id would
    produce a reference indistinguishable from one that resolves.

    ``inherited`` is project-level state from *other* sessions. It is a separate
    argument, and a separate field on the package, because the session check
    above is right: state from another session is not this session's account of
    its work. Labelling it keeps both facts — that it applies here, and that it
    was concluded elsewhere.

    ``archive_reference`` is for a caller that already knows its archive's shape.
    Deriving one requires every record in hand, and holding a large archive in
    memory to describe it is exactly the proportional cost the archive exists to
    avoid — so a caller that streamed it may hand the summary over instead.

    ``created_at`` defaults to now. It is exposed so a caller who needs a
    byte-identical rebuild — a test, or a reproducibility check — can pin the one
    field that legitimately varies between two builds of the same state.
    """
    report = PackageBuildReport()
    items = list(state)

    active = [item for item in items if item.status is not StateStatus.SUPERSEDED]
    report.superseded_excluded = len(items) - len(active)
    report.state_items = len(active)

    wrong_session = {item.id for item in active if item.session_id != session_id}
    if wrong_session:
        raise ValueError(
            f"state items belong to a different session than {session_id!r}: "
            f"{sorted(wrong_session)[:5]}. A package mixing sessions would "
            f"present one agent's work as another's."
        )

    by_seq = {record.seq: record for record in (archive_records or ())}
    by_record_id = {record.id: record.seq for record in (archive_records or ())}
    provenance = provenance or {}

    evidence: list[EvidenceReference] = []
    for item in active:
        seqs = tuple(sorted(provenance.get(item.id, ()) or _sources_to_seqs(item, by_record_id)))
        known = [seq for seq in seqs if seq in by_seq]
        reference = EvidenceReference(
            state_id=item.id,
            archive_seqs=seqs,
            archive_record_ids=tuple(by_seq[seq].id for seq in known),
            excerpt=_excerpt(by_seq[known[0]]) if known else "",
        )
        evidence.append(reference)
        if reference.resolvable:
            report.evidence_resolved += 1
        else:
            report.evidence_unresolved += 1

    if report.evidence_unresolved:
        report.warnings.append(
            f"{report.evidence_unresolved} of {report.state_items} state items have no "
            f"archive provenance; they are packaged with an empty reference rather than "
            f"an invented one, and cannot be checked against the source"
        )

    archive_reference = archive_reference or _archive_reference(
        session_id=session_id,
        records=archive_records or (),
        # From the evidence actually built, not from the ``provenance`` argument.
        # Those differ whenever provenance was inferred from ``sources``, and
        # taking the argument would leave the package citing records it never
        # hashed — so the deep validation would have nothing to check and would
        # pass for want of anything to fail on.
        cited=sorted({seq for reference in evidence for seq in reference.archive_seqs}),
        location=archive_location,
    )
    if archive_reference is None and archive_records:
        report.warnings.append("archive records were supplied but none could be referenced")

    recent_section = None
    if recent:
        message_seqs = [m["seq"] for m in recent if isinstance(m.get("seq"), int)]
        recent_section = RecentContext(
            messages=tuple(recent),
            from_seq=min(message_seqs) if message_seqs else None,
            to_seq=max(message_seqs) if message_seqs else None,
            reason=recent_reason,
        )
        report.recent_messages = len(recent)

    manifest_metadata = dict(metadata or {})
    if report.superseded_excluded:
        manifest_metadata["superseded_excluded"] = report.superseded_excluded

    package = CtxPackage(
        manifest=Manifest(
            package_id=package_id or ids.new_id(ids.PACKAGE),
            session_id=session_id,
            title=title,
            source=source,
            created_by=created_by,
            generator_version=generator_version,
            **({"created_at": created_at} if created_at is not None else {}),
            metadata=manifest_metadata,
        ),
        state=tuple(active),
        inherited=tuple(inherited),
        evidence=tuple(evidence),
        recent=recent_section,
        archive=archive_reference,
    )
    return package, report


def _archive_reference(
    *,
    session_id: str,
    records: Sequence[ArchiveRecord],
    cited: Sequence[int],
    location: str,
) -> ArchiveReference | None:
    """A pointer to the archive, hashing only the records actually cited.

    Hashing every record would grow the reference with the conversation and turn
    a pointer into a manifest of the whole history. Hashing none would leave a
    receiver unable to tell one archive from another. The cited records are the
    ones the package's claims depend on, so they are the ones worth proving.
    """
    if not records:
        return None
    seqs = [record.seq for record in records]
    by_seq = {record.seq: record for record in records}
    return ArchiveReference(
        archive_format=ARCHIVE_FORMAT,
        archive_format_version=ARCHIVE_FORMAT_VERSION,
        session_id=session_id,
        first_seq=min(seqs),
        last_seq=max(seqs),
        record_count=len(records),
        location=location,
        record_hashes={seq: by_seq[seq].hash for seq in cited if seq in by_seq},
    )


__all__ = ["EXCERPT_LIMIT", "PackageBuildReport", "build"]
