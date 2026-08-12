"""Frozen records and UTC timestamp enforcement."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from open_context.models import Session, utc_now


def test_records_are_frozen():
    session = Session()
    with pytest.raises(ValidationError):
        session.title = "changed"


def test_records_are_not_hashable_because_of_metadata():
    """frozen=True implies hashable, but the metadata dict makes hashing raise.

    Recorded as a test so nobody builds a set of records and discovers this in
    a later phase. Key collections by ``id``.
    """
    session = Session()
    with pytest.raises(TypeError):
        hash(session)


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        Session(unexpected="value")


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        Session(created_at=datetime(2026, 1, 1, 12, 0, 0))


def test_aware_timestamp_normalised_to_utc():
    ist = timezone(timedelta(hours=5, minutes=30))
    session = Session(created_at=datetime(2026, 1, 1, 17, 30, 0, tzinfo=ist))
    assert session.created_at.tzinfo is UTC
    assert session.created_at == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_utc_now_is_aware():
    assert utc_now().tzinfo is UTC
