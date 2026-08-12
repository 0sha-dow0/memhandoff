"""One session's append-only log.

Two files plus a manifest:

```
<root>/sessions/<session_id>/
    manifest.json     format, version, session id, creation time
    records.jsonl     the source of truth, one record per line
    records.idx       rebuildable offsets, 12 bytes per record
```

**Write order matters.** A record is written to the data file and synced before
its index entry is written. If the process dies between the two, the index is
short and the data file is whole, which is repaired by scanning the tail. The
reverse order would leave an index entry pointing at bytes that were never
written, which is not repairable.

**Recovery is bounded.** Opening an archive scans only from the end of the last
indexed record to the end of the file, not the whole history. A clean archive
reads nothing.

**A torn tail is truncated, never accepted.** An interrupted append leaves a
partial final line. That line is removed and reported through the recovery
report. Damage anywhere other than the tail is corruption, and corruption is
raised rather than repaired.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from open_context.archive.errors import (
    ArchiveFormatError,
    CorruptRecordError,
    DuplicateSeqError,
    RecordNotFoundError,
)
from open_context.archive.index import OffsetIndex
from open_context.archive.records import FORMAT, FORMAT_VERSION, ArchiveRecord

DATA_FILE = "records.jsonl"
INDEX_FILE = "records.idx"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class RecoveryReport:
    """What opening the archive had to fix.

    ``clean`` is the normal case. Anything else is reported rather than logged
    quietly, because a truncated record is lost conversation and the caller
    should be able to say so.
    """

    indexed: int = 0
    truncated_bytes: int = 0
    recovered_records: int = 0

    @property
    def clean(self) -> bool:
        return self.truncated_bytes == 0 and self.recovered_records == 0


@dataclass(frozen=True)
class IntegrityReport:
    """Result of verifying an archive."""

    records: int
    corrupt: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.corrupt


class SessionLog:
    """Append-only log for one session.

    Not safe to share across processes for writing. Reads are safe alongside
    other readers.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.data_path = directory / DATA_FILE
        self.index = OffsetIndex(directory / INDEX_FILE)
        self.manifest_path = directory / MANIFEST_FILE
        self._count = 0
        self.recovery = RecoveryReport()

    # ------------------------------------------------------------------
    # Lifecycle

    @property
    def session_id(self) -> str:
        manifest: dict[str, Any] = json.loads(self.manifest_path.read_text())
        return str(manifest["session_id"])

    def _write_manifest(self, session_id: str, created_at: str) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "format": FORMAT,
                    "version": FORMAT_VERSION,
                    "session_id": session_id,
                    "created_at": created_at,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _check_manifest(self) -> None:
        manifest: dict[str, Any] = json.loads(self.manifest_path.read_text())
        found = str(manifest.get("format", "unknown"))
        if found != FORMAT:
            raise ArchiveFormatError(found, FORMAT)
        version = int(manifest.get("version", 0))
        if version > FORMAT_VERSION:
            raise ArchiveFormatError(f"{FORMAT} v{version}", f"{FORMAT} v{FORMAT_VERSION}")

    def _open_existing(self) -> None:
        self._check_manifest()
        self.recovery = self._recover()
        self._count = self.index.count()

    # ------------------------------------------------------------------
    # Recovery

    def _recover(self) -> RecoveryReport:
        """Reconcile the index with the data file, scanning only the tail."""
        file_size = self.data_path.stat().st_size
        start = self.index.last_end()

        if start > file_size:
            # The index describes records the data file does not contain. The
            # data file is the source of truth, so the index is rebuilt.
            self.index.clear()
            start = 0

        if start == file_size:
            return RecoveryReport(indexed=self.index.count())

        recovered = 0
        truncated = 0
        offset = start
        with self.data_path.open("rb") as handle:
            handle.seek(start)
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # An append that never finished. The bytes are removed.
                    truncated = len(line)
                    break
                try:
                    ArchiveRecord.from_line(line)
                except (ValueError, UnicodeDecodeError):
                    # A complete but unparseable line at the tail is also a
                    # failed write, so it goes the same way.
                    truncated = len(line)
                    break
                self.index.append(offset, len(line), sync=False)
                offset += len(line)
                recovered += 1

        if truncated:
            with self.data_path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())

        return RecoveryReport(
            indexed=self.index.count(), truncated_bytes=truncated, recovered_records=recovered
        )

    # ------------------------------------------------------------------
    # Writing

    def __len__(self) -> int:
        return self._count

    @property
    def count(self) -> int:
        return self._count

    def append(
        self,
        record_id: str,
        payload: Any,  # noqa: ANN401  the archive stores shapes we do not model
        *,
        kind: str = "message",
    ) -> ArchiveRecord:
        """Append one record and return it with its assigned position.

        The data file is synced before the index entry is written, so an
        interrupted append can always be repaired from the data file.
        """
        record = ArchiveRecord(seq=self._count, id=record_id, kind=kind, payload=payload)
        line = record.to_line()
        with self.data_path.open("ab") as handle:
            offset = handle.tell()
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self.index.append(offset, len(line), sync=True)
        self._count += 1
        return record

    def extend(self, records: Iterable[tuple[str, Any]], *, kind: str = "message") -> int:
        """Append many records, syncing once at the end.

        Faster than repeated ``append`` and no less safe: a crash mid-batch
        leaves a prefix of the batch plus at most one torn line, which recovery
        removes. It never leaves a half-written record accepted as valid.
        """
        written = 0
        offsets: list[tuple[int, int]] = []
        with self.data_path.open("ab") as handle:
            for record_id, payload in records:
                record = ArchiveRecord(
                    seq=self._count + written, id=record_id, kind=kind, payload=payload
                )
                line = record.to_line()
                offset = handle.tell()
                handle.write(line)
                offsets.append((offset, len(line)))
                written += 1
            handle.flush()
            os.fsync(handle.fileno())
        for offset, length in offsets:
            self.index.append(offset, length, sync=False)
        self._count += written
        return written

    def _reject_rewrite(self, seq: int) -> None:
        if seq < self._count:
            raise DuplicateSeqError(seq)

    # ------------------------------------------------------------------
    # Reading

    def read(self, seq: int) -> ArchiveRecord:
        """Read one record by position. Constant time, one record in memory."""
        entry = self.index.get(seq)
        if entry is None:
            raise RecordNotFoundError(seq)
        offset, length = entry
        with self.data_path.open("rb") as handle:
            handle.seek(offset)
            line = handle.read(length)
        return self._parse(line.rstrip(b"\n"), seq)

    def _parse(self, line: bytes, seq: int) -> ArchiveRecord:
        try:
            record = ArchiveRecord.from_line(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise CorruptRecordError(seq, str(exc)) from exc
        if not record.verify():
            raise CorruptRecordError(seq, "content does not match its stored hash")
        return record

    def range(self, start: int = 0, stop: int | None = None) -> Iterator[ArchiveRecord]:
        """Yield records in ``[start, stop)``, one at a time.

        Reads sequentially from the start offset rather than seeking per record,
        and holds one record in memory regardless of how many are requested.
        """
        end = self._count if stop is None else min(stop, self._count)
        if start < 0:
            raise ValueError("start must not be negative")
        if start >= end:
            return
        entry = self.index.get(start)
        if entry is None:
            raise RecordNotFoundError(start)
        with self.data_path.open("rb") as handle:
            handle.seek(entry[0])
            for seq in range(start, end):
                line = handle.readline()
                if not line:
                    raise RecordNotFoundError(seq)
                yield self._parse(line.rstrip(b"\n"), seq)

    def __iter__(self) -> Iterator[ArchiveRecord]:
        """Stream every record. Memory stays flat regardless of archive size."""
        return self.range()

    def window(self, seq: int, *, before: int = 0, after: int = 0) -> Iterator[ArchiveRecord]:
        """Records around a position, clamped to the archive."""
        return self.range(max(0, seq - before), min(self._count, seq + after + 1))

    def tail(self, limit: int) -> Iterator[ArchiveRecord]:
        """The last ``limit`` records."""
        if limit <= 0:
            return iter(())
        return self.range(max(0, self._count - limit))

    # ------------------------------------------------------------------
    # Integrity

    def verify(self) -> IntegrityReport:
        """Check every record against its hash by streaming the file.

        Returns the positions of damaged records rather than raising, so a
        caller can report the extent of the damage instead of stopping at the
        first bad line.
        """
        corrupt: list[int] = []
        seq = 0
        with self.data_path.open("rb") as handle:
            for raw in handle:
                line = raw.rstrip(b"\n")
                try:
                    record = ArchiveRecord.from_line(line)
                except (ValueError, UnicodeDecodeError):
                    corrupt.append(seq)
                else:
                    if not record.verify() or record.seq != seq:
                        corrupt.append(seq)
                seq += 1
        return IntegrityReport(records=seq, corrupt=tuple(corrupt))

    def rebuild_index(self) -> int:
        """Discard and rebuild the index from the data file."""
        self.index.clear()
        self.recovery = self._recover()
        self._count = self.index.count()
        return self._count

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Present for symmetry. No handle is held open between operations."""
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
