"""Driving an importer into the archive.

```
path -> importer.read() -> RawEvent -> log.append() -> ImportReport
```

Everything here is provider-neutral. There is no branch on which adapter
produced an event and no place to add one: the pipeline sees ``RawEvent`` and
``MalformedRecord`` and could not name a provider if it wanted to. Provider
knowledge stops at the adapter, which is the whole point of having adapters.

**Ordering.** Events are appended one at a time in the order the importer yields
them, and the archive assigns ``seq`` on append. Source order is therefore
archive order, and no sorting step exists to get that wrong.

**Memory.** One event is held at a time. The two exceptions are the sets used to
notice a repeated source id and an unmatched tool result, which hold identifiers
rather than content: they grow with the number of events, not with how much was
said, and a conversation large enough for that to matter would have a set orders
of magnitude smaller than the text it came from.

**Why ``append`` and not ``extend``.** ``SessionLog.extend`` writes one ``kind``
for a whole batch and buffers an offset per record, and imported events have
mixed kinds. Appending per event keeps memory flat and archive ``kind`` faithful
to the event type, at the cost of one ``fsync`` per record. That trade is
recorded in docs/import.md as the open performance question for this phase.

**Imports are not atomic, explicitly.** See ``import_events``. The archive has
``append`` and ``extend`` and no delete, truncate, or rollback, so a failure
partway cannot un-write what already landed. Rather than pretend otherwise, a
failure raises ``IncompleteImportError`` carrying the report of exactly what was
written.

**No semantic state.** This module writes archive records and nothing else. It
does not create a ``Decision``, a ``Goal``, a ``Task``, or any other state item,
and it does not write conversation into SQLite. "We should use PostgreSQL"
imports as a message saying that, never as a decision to do it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from open_context.archive.log import SessionLog
from open_context.archive.store import Archive
from open_context.importers.base import ConversationImporter, ReadResult
from open_context.importers.errors import (
    DuplicateSourceIdError,
    MalformedRecordError,
    UnsupportedSourceError,
)
from open_context.importers.events import EventType, RawEvent
from open_context.importers.registry import DEFAULT_IMPORTERS
from open_context.importers.report import (
    BLANK_LINE,
    SAMPLE_LIMIT,
    ImportReport,
    MalformedRecord,
)

MalformedPolicy = Literal["abort", "skip"]
DuplicatePolicy = Literal["report", "abort"]

DETECT_SAMPLE_BYTES = 8192
"""How much of a source is read to choose an importer."""

SYNTHETIC_PREFIX: Final = "synth"
"""Marks an archive record id this runtime made up rather than read from a source."""

FINGERPRINT_LENGTH: Final = 24
"""Hex characters of the encoded source key kept in a synthetic id."""

DIGEST_CHUNK_BYTES: Final = 1 << 16
"""Read size while digesting a source. Small enough to keep the import streaming."""


# ----------------------------------------------------------------------
# Identity
#
# Four distinct concepts, and conflating any two of them produces a wrong id:
#
#   source identity   which logical source this is. A caller's judgement, not a
#                     property of any bytes. Two byte-identical files may be one
#                     logical source or two, and only the caller knows which.
#   content digest    a fingerprint of what a source contains. Detects that two
#                     sources hold the same bytes; cannot decide whether they
#                     are the same source.
#   archive sequence  where a record sits in one local archive. Ordering, not
#                     identity, and it changes with what else was imported.
#   synthetic id      the archive record id for an event whose provider gave
#                     none: a deterministic encoding of source identity and
#                     position within that source.


def content_digest_for_bytes(data: bytes) -> str:
    """Fingerprint of content already in memory.

    A fingerprint of *what a source contains*. It is not a source identity: two
    separate exports that happen to hold identical bytes share a fingerprint and
    are still two sources.
    """
    return hashlib.sha256(data).hexdigest()


def content_digest_for_path(path: str | Path) -> str:
    """Fingerprint of a file's content, read in bounded chunks.

    One extra sequential pass over the source, which costs far less than the
    ``fsync`` per record the import itself performs.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(DIGEST_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def default_source_key(path: str | Path) -> str:
    """The logical source identity ``import_path`` assumes when told nothing.

    Reads as **"this file, holding this content, at this location"**:

    ```
    file:/home/dana/export.jsonl#9f2c...e41
    ```

    Both halves are load-bearing.

    *Location* distinguishes two files that happen to hold identical bytes. They
    are two files, so by default they are two logical sources, and the content
    alone could never tell them apart.

    *Content* means an edited file is a different logical source. Once the bytes
    change, position 7 no longer names the event it used to, and carrying the
    old identity forward would hand one id to two unrelated events, which is the
    exact failure this scheme exists to prevent.

    The consequence is that copying a source to another machine changes its
    default identity, because a path is local to a machine. That is a default,
    not a rule: a caller who means "these two copies are the same logical
    source" passes ``source_key`` explicitly and says so. No default can infer
    that intent, so this one is documented rather than clever.
    """
    return f"file:{Path(path).resolve()}#{content_digest_for_path(path)}"


def check_source_key(source_key: str) -> str:
    """Reject a source key that identifies nothing."""
    if not source_key.strip():
        raise ValueError("source key must not be empty")
    return source_key


def source_key_fingerprint(source_key: str) -> str:
    """Encode a logical source key for use inside an archive record id.

    Hashed rather than embedded so that a key of any length, containing any
    characters a caller finds meaningful, still yields a short fixed-width id.
    The hash is an encoding of the identity, never a substitute for it.
    """
    return hashlib.sha256(check_source_key(source_key).encode("utf-8")).hexdigest()[
        :FINGERPRINT_LENGTH
    ]


def synthetic_id(source_key: str, position: int) -> str:
    """The archive record id for an event whose source gave none.

    ```
    synth-<fingerprint of the logical source key>-<position within that source>
    ```

    Archive ids mean "identity of the underlying item", and two records sharing
    one are read as the same item written twice, a message and a later revision
    of it. An invented id that landed on an unrelated event would assert that
    relationship falsely, so identity is scoped to the logical source and the
    position within it, and to nothing else.

    * The same logical source and position always give the same id, on every
      machine and in every archive, so two imports of one source are comparable.
    * Distinct logical source keys give distinct ids, collision-resistantly:
      the fingerprint is a truncated SHA-256, so this is a cryptographic
      argument about how hard it is to find a clash, not a proof that none
      exists. Nothing here should be read as guaranteeing uniqueness.
    * Nothing depends on ``seq``. Archive ordering is where a record landed, not
      what it is, and an id that moved with it would rename an event according
      to what else happened to be imported first.
    * Nothing is random, so re-running an import reproduces the ids exactly.

    The prefix marks the id as generated by this runtime. The event still
    records ``source_id=None``, so the stored data never claims otherwise.
    """
    if position < 0:
        raise ValueError(f"position must not be negative, got {position}")
    return f"{SYNTHETIC_PREFIX}-{source_key_fingerprint(source_key)}-{position:08d}"


# ----------------------------------------------------------------------
# Import


@dataclass(frozen=True)
class _Failure:
    """Why the loop stopped. Turned into the caller's exception once the report exists."""

    malformed: MalformedRecord | None = None
    duplicate: tuple[str, int] | None = None


class _Counters:
    """Mutable tallies for one import.

    Separate from ``ImportReport`` so the report can stay frozen and be the only
    thing a caller ever sees.
    """

    def __init__(self) -> None:
        self.read = 0
        self.written = 0
        self.first_seq: int | None = None
        self.last_seq: int | None = None
        self.by_type: dict[str, int] = {}
        self.without_source_id = 0
        self.without_timestamp = 0
        self.unusable_timestamps = 0
        self.unmatched_tool_results = 0
        self.tool_events_without_name = 0
        self.blank_lines = 0
        self.unknown_types = 0
        self.unknown_names: list[str] = []
        self.duplicates = 0
        self.duplicate_ids: list[str] = []
        self.malformed = 0
        self.malformed_sample: list[MalformedRecord] = []
        self.seen_ids: set[str] = set()
        self.open_tool_calls: set[str] = set()

    def sample(self, bucket: list[str], value: str) -> None:
        if value not in bucket and len(bucket) < SAMPLE_LIMIT:
            bucket.append(value)


def import_events(
    log: SessionLog,
    results: Iterable[ReadResult],
    *,
    provider: str,
    source_key: str,
    on_malformed: MalformedPolicy = "abort",
    on_duplicate: DuplicatePolicy = "report",
) -> ImportReport:
    """Append normalized events to a session log and report what happened.

    **An import is not atomic and does not roll back.** The archive is
    append-only; it offers no delete, no truncate, and no undo, and building a
    staging area on top would mean writing every record twice, renumbering
    every ``seq`` on commit, and holding a second copy on disk, to buy a
    guarantee that a live agent stream could never use anyway. So the guarantee
    this phase makes is the honest one: records written before a failure stay
    written, and the failure says exactly which.

    On failure the raised ``IncompleteImportError`` carries an ``ImportReport``
    with ``completed=False``, ``events_written`` counting what landed and
    ``last_seq`` naming the final record. The record that caused the failure is
    never written. Callers wanting all-or-nothing import into a session of their
    own and delete its directory if the report is not ``completed``; that is one
    ``rmtree`` and needs nothing from this runtime.

    ``on_malformed="abort"`` is the default because a source this build cannot
    read is more likely an adapter bug than a broken file, and importing three
    quarters of a conversation while believing it whole is the worse failure.
    ``"skip"`` continues, and the record still appears in the report; there is
    no setting that discards one quietly.

    ``on_duplicate="report"`` is the default because a repeated id is legitimate
    in the archive, which stores a message and a later revision of it as two
    records distinguished by position. ``"abort"`` is for a caller who knows
    their source assigns ids uniquely and wants to hear about it when it does
    not.

    ``source_key`` names the **logical source** these events came from, and is
    required because an event whose provider gave no id is identified by it. It
    is a caller's judgement rather than a property of any bytes: whether two
    exports holding identical content are one source or two is a question only
    the caller can answer, so nothing here tries to infer it. A live agent
    stream passes something stable and meaningful, such as a run id;
    ``import_path`` defaults it from the file. See ``synthetic_id``.
    """
    check_source_key(source_key)
    counters = _Counters()
    started_at = log.count

    for result in results:
        failure = (
            _record_malformed(counters, result, on_malformed)
            if isinstance(result, MalformedRecord)
            else _write(log, counters, result, source_key, on_duplicate)
        )
        if failure is not None:
            raise _abort(
                failure,
                _build(log, counters, provider, source_key, started_at, completed=False),
            )

    return _build(log, counters, provider, source_key, started_at, completed=True)


def _abort(failure: _Failure, report: ImportReport) -> Exception:
    if failure.malformed is not None:
        return MalformedRecordError(failure.malformed, report)
    assert failure.duplicate is not None
    source_id, position = failure.duplicate
    return DuplicateSourceIdError(source_id, position, report)


def _build(
    log: SessionLog,
    counters: _Counters,
    provider: str,
    source_key: str,
    started_at: int,
    *,
    completed: bool,
) -> ImportReport:
    return ImportReport(
        session_id=log.session_id,
        provider=provider,
        source_key=source_key,
        completed=completed,
        events_read=counters.read,
        events_written=counters.written,
        first_seq=counters.first_seq,
        last_seq=counters.last_seq,
        appended_to_existing=started_at > 0,
        by_type=dict(counters.by_type),
        without_source_id=counters.without_source_id,
        without_timestamp=counters.without_timestamp,
        unusable_timestamps=counters.unusable_timestamps,
        unmatched_tool_results=counters.unmatched_tool_results,
        tool_events_without_name=counters.tool_events_without_name,
        blank_lines_skipped=counters.blank_lines,
        unknown_source_types=counters.unknown_types,
        unknown_source_type_sample=tuple(counters.unknown_names),
        duplicate_source_ids=counters.duplicates,
        duplicate_source_id_sample=tuple(counters.duplicate_ids),
        malformed_records=counters.malformed,
        malformed_sample=tuple(counters.malformed_sample),
    )


def _record_malformed(
    counters: _Counters, record: MalformedRecord, policy: MalformedPolicy
) -> _Failure | None:
    if record.reason == BLANK_LINE:
        counters.blank_lines += 1
        return None
    counters.malformed += 1
    if len(counters.malformed_sample) < SAMPLE_LIMIT:
        counters.malformed_sample.append(record)
    return _Failure(malformed=record) if policy == "abort" else None


def _write(
    log: SessionLog,
    counters: _Counters,
    event: RawEvent,
    source_key: str,
    on_duplicate: DuplicatePolicy,
) -> _Failure | None:
    """Append one event, tallying everything worth telling the caller about.

    A refused duplicate is counted and then reported as a failure without being
    written, so the partial report names the problem and ``events_written``
    stays true to what is on disk.
    """
    position = counters.read
    repeated = event.source_id is not None and event.source_id in counters.seen_ids
    if repeated and on_duplicate == "abort":
        assert event.source_id is not None
        counters.duplicates += 1
        counters.sample(counters.duplicate_ids, event.source_id)
        return _Failure(duplicate=(event.source_id, position))

    counters.read += 1
    counters.by_type[event.type.value] = counters.by_type.get(event.type.value, 0) + 1

    if event.source_id is None:
        counters.without_source_id += 1
    else:
        if repeated:
            counters.duplicates += 1
            counters.sample(counters.duplicate_ids, event.source_id)
        counters.seen_ids.add(event.source_id)

    if event.timestamp is None:
        counters.without_timestamp += 1
        if _claimed_a_timestamp(event):
            counters.unusable_timestamps += 1

    if event.type is EventType.OTHER:
        counters.unknown_types += 1
        if event.source_type is not None:
            counters.sample(counters.unknown_names, event.source_type)

    _tally_tool_pairing(counters, event)

    record = log.append(
        event.source_id or synthetic_id(source_key, position),
        event.to_payload(),
        kind=event.type.value,
    )
    counters.written += 1
    if counters.first_seq is None:
        counters.first_seq = record.seq
    counters.last_seq = record.seq
    return None


def _claimed_a_timestamp(event: RawEvent) -> bool:
    """Whether the source offered a timestamp the adapter could not use.

    Distinguishes "this export has no timestamps" from "this export has
    timestamps in a form we refused", which are different problems with
    different fixes. Answered from ``raw`` because that is the only place the
    original value still exists.
    """
    if not isinstance(event.raw, dict):
        return False
    return any(key in event.raw for key in ("timestamp", "created_at", "time"))


def _tally_tool_pairing(counters: _Counters, event: RawEvent) -> None:
    """Note tool calls with no name and tool results with no matching call.

    Recorded, never repaired. Pairing a result to a call it does not name would
    be a guess about what the agent did, and this phase does not guess.
    """
    if event.type is EventType.TOOL_CALL:
        if event.tool_name is None:
            counters.tool_events_without_name += 1
        identifier = event.tool_call_id or event.source_id
        if identifier is not None:
            counters.open_tool_calls.add(identifier)
    elif event.type is EventType.TOOL_RESULT:
        if event.tool_name is None:
            counters.tool_events_without_name += 1
        identifier = event.tool_call_id
        if identifier is None or identifier not in counters.open_tool_calls:
            counters.unmatched_tool_results += 1
        else:
            counters.open_tool_calls.discard(identifier)


# ----------------------------------------------------------------------
# Sources


def detect_importer(
    sample: str, importers: Iterable[ConversationImporter] = DEFAULT_IMPORTERS
) -> ConversationImporter:
    """The first importer that recognises this source."""
    for importer in importers:
        if importer.detect(sample):
            return importer
    raise UnsupportedSourceError("no importer recognises this source")


def import_path(
    archive: Archive,
    session_id: str,
    path: str | Path,
    *,
    source_key: str | None = None,
    importer: ConversationImporter | None = None,
    encoding: str = "utf-8",
    on_malformed: MalformedPolicy = "abort",
    on_duplicate: DuplicatePolicy = "report",
) -> ImportReport:
    """Import a conversation file into a session archive.

    Opens the session if it exists and creates it otherwise, so importing twice
    appends a second copy rather than failing or replacing the first. The
    archive is append-only and has no notion of the same conversation arriving
    again; a caller wanting one copy imports into a session of its own, and the
    report says ``appended_to_existing`` when that is not what happened.

    ``source_key`` names the logical source these events came from, and is what
    an event with no id of its own is identified by. Passing it is how a caller
    states an identity the filesystem cannot: that two copies of one export are
    the same source, or that two byte-identical files are not. Omitting it takes
    ``default_source_key``, whose exact meaning is documented there.

    The key is reported back as ``ImportReport.source_key`` and is not written
    into the archive; only its fingerprint appears, inside synthetic record ids.

    Not atomic: see ``import_events``.
    """
    source = Path(path)
    if not source.is_file():
        raise UnsupportedSourceError(f"no file at {source}")

    chosen = importer
    if chosen is None:
        with source.open("r", encoding=encoding, errors="strict") as handle:
            try:
                sample = handle.read(DETECT_SAMPLE_BYTES)
            except UnicodeDecodeError as exc:
                raise UnsupportedSourceError(f"{source} is not valid {encoding}: {exc}") from exc
        chosen = detect_importer(sample)

    resolved_key = check_source_key(source_key) if source_key else default_source_key(source)
    log = archive.open_or_create(session_id)
    with source.open("r", encoding=encoding, errors="strict") as handle:
        try:
            return import_events(
                log,
                chosen.read(handle),
                provider=chosen.provider,
                source_key=resolved_key,
                on_malformed=on_malformed,
                on_duplicate=on_duplicate,
            )
        except UnicodeDecodeError as exc:
            raise UnsupportedSourceError(f"{source} is not valid {encoding}: {exc}") from exc


def stream_events(log: SessionLog, start: int = 0, stop: int | None = None) -> Iterable[RawEvent]:
    """Read normalized events back out of an archive, one at a time.

    The inverse of the import, and the only supported way to get events back:
    there is no read-everything call here for the same reason there is none on
    the archive.
    """
    for record in log.range(start, stop):
        yield RawEvent.from_payload(record.payload)


__all__ = [
    "DETECT_SAMPLE_BYTES",
    "FINGERPRINT_LENGTH",
    "SYNTHETIC_PREFIX",
    "DuplicatePolicy",
    "MalformedPolicy",
    "check_source_key",
    "content_digest_for_bytes",
    "content_digest_for_path",
    "default_source_key",
    "detect_importer",
    "import_events",
    "import_path",
    "source_key_fingerprint",
    "stream_events",
    "synthetic_id",
]
