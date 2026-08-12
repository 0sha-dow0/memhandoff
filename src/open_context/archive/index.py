"""The offset index.

Twelve bytes per record: an unsigned 64-bit offset and an unsigned 32-bit
length, little-endian. The length includes the terminating newline, so
``offset + length`` is exactly where the next record begins. Excluding it makes
``last_end`` point at the newline instead of past it, and recovery then eats a
byte of the last good record. Entry N describes record N, so reading record N is a seek
to ``N * 12`` in the index and a seek to the offset it names in the data file.
Nothing scans.

The index is derived data. It can be deleted at any time and rebuilt from the
JSONL, which is the only source of truth. That is deliberate: an index that
could not be rebuilt would be a second thing to keep consistent, and a second
thing to lose.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Final

ENTRY: Final = struct.Struct("<QI")
ENTRY_SIZE: Final = ENTRY.size


class OffsetIndex:
    """Fixed-width offsets into the data file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def exists(self) -> bool:
        return self.path.exists()

    def count(self) -> int:
        """Number of complete entries.

        A trailing partial entry is ignored rather than trusted. The index is
        rebuildable, so a torn write here costs a rescan of the tail, not data.
        """
        if not self.path.exists():
            return 0
        return self.path.stat().st_size // ENTRY_SIZE

    def get(self, seq: int) -> tuple[int, int] | None:
        """Offset and length of record ``seq``, or None if not indexed."""
        if seq < 0 or seq >= self.count():
            return None
        with self.path.open("rb") as handle:
            handle.seek(seq * ENTRY_SIZE)
            raw = handle.read(ENTRY_SIZE)
        if len(raw) != ENTRY_SIZE:
            return None
        offset, length = ENTRY.unpack(raw)
        return int(offset), int(length)

    def append(self, offset: int, length: int, *, sync: bool) -> None:
        with self.path.open("ab") as handle:
            handle.write(ENTRY.pack(offset, length))
            handle.flush()
            if sync:
                import os

                os.fsync(handle.fileno())

    def truncate(self, entries: int) -> None:
        """Cut the index to a whole number of entries."""
        with self.path.open("r+b") as handle:
            handle.truncate(entries * ENTRY_SIZE)

    def clear(self) -> None:
        self.path.write_bytes(b"")

    def last_end(self) -> int:
        """Byte offset just past the last indexed record, or 0 if empty."""
        total = self.count()
        if total == 0:
            return 0
        entry = self.get(total - 1)
        if entry is None:
            return 0
        offset, length = entry
        return offset + length
