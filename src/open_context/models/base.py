"""Shared model configuration.

Two rules apply to every record in the data model.

**Records are frozen.** A changing fact produces a new record that points at the
old one through ``supersedes``; it never edits the old one in place. The runtime
has to answer "what did we believe in March, and why do we believe something
else now", and in-place mutation destroys the first half of that question.

**Timestamps are timezone-aware UTC.** A naive datetime is rejected rather than
assumed to be local time. Conversation history arrives from exports produced in
other timezones, and a silent assumption there corrupts ordering in a way that
surfaces much later as an unexplainable retrieval bug.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict


def _require_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalise aware ones to UTC."""
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware; naive datetimes are ambiguous")
    return value.astimezone(UTC)


Timestamp = Annotated[datetime, AfterValidator(_require_utc)]
"""A timezone-aware datetime, normalised to UTC."""


def utc_now() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


class Record(BaseModel):
    """Base class for every persisted record.

    ``frozen`` makes instances immutable. It does not make them hashable in
    practice: records carry a ``metadata`` dict, and hashing one raises. Use the
    ``id`` field as a set or dict key.

    ``extra="forbid"`` turns an unexpected field into an error instead of
    silently dropping it, which matters when parsing provider exports whose
    formats change without notice.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )
