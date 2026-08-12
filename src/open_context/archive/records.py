"""The archive record.

One record is one line of JSON. It carries its own hash, so a line can be
checked without consulting anything else. That property is what lets the archive
be verified by streaming, and lets a damaged record be identified precisely
rather than invalidating the whole file.

The archive stores raw material: messages, tool results, events, and whatever
else a provider export contains. It deliberately does not import the Phase 1
models. Those describe interpreted structure and live in SQLite; the archive
holds the bytes they were derived from and must be able to store a payload whose
shape nothing in this codebase understands yet.

That is why ``payload`` is typed ``Any`` and the annotation checks are suppressed
where it appears. Narrowing it would mean the archive could only store shapes we
have already anticipated, which defeats its purpose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field

FORMAT: Final = "open-context-jsonl"
FORMAT_VERSION: Final = 1


def canonical_json(value: Any) -> str:  # noqa: ANN401
    """Serialise deterministically.

    Sorted keys and fixed separators, so the same content always produces the
    same bytes and therefore the same hash. ``ensure_ascii`` stays on: escaping
    non-ASCII makes the file safe to move between machines whose default
    encodings differ, at the cost of some readability.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_body(seq: int, record_id: str, kind: str, payload: Any) -> str:  # noqa: ANN401
    """Hash covering every field of a record except the hash itself.

    Deliberately byte-exact, with no Unicode normalisation. ``Evidence`` hashes
    NFC-normalised text because its hash is used to recognise duplicate content.
    The archive hash exists to detect corruption in stored originals, so it must
    notice every byte difference, including one composed form replacing another.
    Normalising here would hide a real change in what was written.
    """
    body = canonical_json({"seq": seq, "id": record_id, "kind": kind, "payload": payload})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class ArchiveRecord(BaseModel):
    """An immutable entry in the archive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0, description="Position in the archive. Assigned by the archive.")
    id: str = Field(
        min_length=1, description="Identity of the underlying item, preserved as given."
    )
    kind: str = Field(default="message", min_length=1)
    payload: Any = Field(default=None, description="The raw content, stored as provided.")
    hash: str = Field(default="", description="sha256 over the other fields.")

    def model_post_init(self, _context: object, /) -> None:
        if not self.hash:
            object.__setattr__(self, "hash", self.expected_hash())

    def expected_hash(self) -> str:
        return hash_body(self.seq, self.id, self.kind, self.payload)

    def verify(self) -> bool:
        """Whether the stored hash still matches the content."""
        return self.hash == self.expected_hash()

    def to_line(self) -> bytes:
        """Serialise to one newline-terminated line."""
        return (
            canonical_json(
                {
                    "seq": self.seq,
                    "id": self.id,
                    "kind": self.kind,
                    "payload": self.payload,
                    "hash": self.hash,
                }
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_line(cls, line: bytes) -> Self:
        """Parse one line. Raises ValueError if it is not a well-formed record."""
        decoded = json.loads(line.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("record is not a JSON object")
        missing = {"seq", "id", "hash"} - set(decoded)
        if missing:
            raise ValueError(f"record is missing {', '.join(sorted(missing))}")
        return cls(**decoded)
