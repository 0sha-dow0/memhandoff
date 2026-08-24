"""The normalized event model.

Ingestion, not interpretation: these tests pin what an event is allowed to carry
and, just as importantly, what it must never invent.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from open_context.importers import EventType, RawEvent


def test_the_six_event_kinds_exist():
    """The vocabulary ingestion has to be able to express."""
    assert {t.value for t in EventType} == {
        "user_message",
        "assistant_message",
        "system_message",
        "developer_message",
        "tool_call",
        "tool_result",
        "other",
    }


def test_message_kinds_line_up_with_the_data_model_roles():
    """A later phase turning events into Messages must not need a mapping table."""
    from open_context.models import Role

    for role in (Role.USER, Role.ASSISTANT, Role.SYSTEM, Role.DEVELOPER):
        assert EventType(f"{role.value}_message")


def test_an_event_needs_only_a_type_and_a_provider():
    event = RawEvent(type=EventType.USER_MESSAGE, provider="generic-jsonl")
    assert event.source_id is None
    assert event.timestamp is None
    assert event.text is None
    assert event.metadata == {}


def test_events_are_frozen():
    event = RawEvent(type=EventType.USER_MESSAGE, provider="p")
    with pytest.raises(ValidationError):
        event.text = "rewritten"


def test_unknown_fields_are_refused_rather_than_dropped():
    """Provider fields belong in raw. A typo'd model field is a bug, not data."""
    with pytest.raises(ValidationError):
        RawEvent(type=EventType.USER_MESSAGE, provider="p", tool_nmae="search")


def test_a_naive_timestamp_is_rejected():
    """Ordering is the one thing this project cannot get wrong."""
    with pytest.raises(ValidationError):
        RawEvent(
            type=EventType.USER_MESSAGE,
            provider="p",
            timestamp=datetime(2024, 3, 1, 10, 0, 0),
        )


def test_an_aware_timestamp_is_normalised_to_utc():
    from datetime import timedelta, timezone

    event = RawEvent(
        type=EventType.USER_MESSAGE,
        provider="p",
        timestamp=datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2))),
    )
    assert event.timestamp == datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def test_payload_round_trip_preserves_every_field():
    event = RawEvent(
        type=EventType.TOOL_RESULT,
        provider="generic-jsonl",
        raw={"anything": [1, {"nested": True}], "unicode": "é中"},
        source_id="evt_7",
        source_type="function_response",
        timestamp=datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC),
        text="3 matches",
        tool_name="search",
        tool_call_id="call_9",
        parent_id="evt_6",
        metadata={"note": "x"},
    )
    assert RawEvent.from_payload(event.to_payload()) == event


def test_a_payload_is_plain_json_types():
    """The archive hashes the payload canonically and knows nothing about pydantic."""
    import json

    event = RawEvent(
        type=EventType.USER_MESSAGE,
        provider="p",
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
    )
    payload = event.to_payload()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["type"] == "user_message"
    assert isinstance(payload["timestamp"], str)


def test_a_non_object_payload_is_refused():
    with pytest.raises(ValueError, match="must be a JSON object"):
        RawEvent.from_payload(["not", "an", "object"])


def test_raw_holds_shapes_the_model_does_not_understand():
    """The point of raw: a provider field with no home here still survives."""
    raw = {"role": "user", "provider_only": {"weights": [0.1, 0.2]}, "v": 3}
    event = RawEvent(type=EventType.USER_MESSAGE, provider="p", raw=raw)
    assert RawEvent.from_payload(event.to_payload()).raw == raw


def test_raw_preserves_values_not_representation():
    """What `raw` promises, stated exactly.

    It is the provider record *as parsed*. Values survive a round trip; the
    original text does not. Key order is normalised by canonical serialisation
    on the way into the archive, and anything the parser already resolved, such
    as a repeated key or the spelling of a number, was gone before it arrived.
    Byte-level preservation of the source is a different feature and is not
    claimed anywhere.
    """
    import json

    from open_context.archive import canonical_json

    source_text = '{"z": 1, "a": 2, "dup": 1, "dup": 2}'
    raw = json.loads(source_text)
    event = RawEvent(type=EventType.USER_MESSAGE, provider="p", raw=raw)

    assert RawEvent.from_payload(event.to_payload()).raw == raw, "values survive"
    assert canonical_json(event.to_payload()["raw"]) == '{"a":2,"dup":2,"z":1}'
    assert raw["dup"] == 2, "the parser resolved the repeated key before we ever saw it"
