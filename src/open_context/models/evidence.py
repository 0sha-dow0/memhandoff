"""Evidence: the exact source text behind a piece of derived state.

Evidence is what makes compaction reversible. A state item says "we chose
RocksDB"; the evidence behind it holds the words that were actually written, so
the claim can be checked rather than trusted.

The content hash is computed at construction. Supplying a hash that does not
match the content raises, which turns a corrupted archive read into an error at
parse time rather than a wrong answer later.
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any

from pydantic import Field, field_validator, model_validator

from open_context.models import ids
from open_context.models.base import Record, Timestamp, utc_now
from open_context.models.enums import SourceType, TrustLevel

_SOURCE_PREFIXES = frozenset({ids.MESSAGE, ids.ARTIFACT, ids.EVENT})


def content_hash(content: str) -> str:
    """Stable content hash.

    Normalises to Unicode NFC before hashing so that text which is canonically
    identical but differently encoded produces one hash. Without that, the same
    passage imported from two providers would deduplicate inconsistently.
    """
    normalised = unicodedata.normalize("NFC", content)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


class Evidence(Record):
    """An immutable excerpt of source material."""

    id: str = Field(default_factory=lambda: ids.new_id(ids.EVIDENCE))
    session_id: str
    source_ids: list[str] = Field(
        min_length=1,
        description="Records this excerpt was taken from. Must not be empty.",
    )
    source_type: SourceType
    trust: TrustLevel = TrustLevel.AGENT_GENERATED
    content: str = Field(min_length=1)
    content_hash: str = ""
    timestamp: Timestamp = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return ids.validate_id(value, expected=ids.EVIDENCE)

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: str) -> str:
        return ids.validate_id(value, expected=ids.SESSION)

    @field_validator("source_ids")
    @classmethod
    def _check_source_ids(cls, value: list[str]) -> list[str]:
        for ref in value:
            ids.validate_ref(ref, allowed=_SOURCE_PREFIXES)
        if len(set(value)) != len(value):
            raise ValueError("source_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _check_hash(self) -> Evidence:
        expected = content_hash(self.content)
        if not self.content_hash:
            object.__setattr__(self, "content_hash", expected)
            return self
        if self.content_hash != expected:
            raise ValueError(
                "content_hash does not match content; the record has been altered or corrupted"
            )
        return self

    def verify(self) -> bool:
        """Recompute the hash and compare. Used when reading from the archive."""
        return self.content_hash == content_hash(self.content)
