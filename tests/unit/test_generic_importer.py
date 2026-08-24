"""The generic JSON Lines adapter.

Parsing only. Nothing here touches the archive.
"""

import io
import json
from datetime import UTC, datetime

import pytest

from open_context.importers import (
    FINGERPRINT_LENGTH,
    SYNTHETIC_PREFIX,
    ConversationImporter,
    EventType,
    FieldMap,
    JsonLinesImporter,
    MalformedRecord,
    RawEvent,
    content_digest_for_bytes,
    content_digest_for_path,
    default_source_key,
    parse_timestamp,
    source_key_fingerprint,
    synthetic_id,
)
from open_context.importers.pipeline import DIGEST_CHUNK_BYTES


@pytest.fixture
def importer():
    return JsonLinesImporter()


def read(importer, records):
    """Feed records through the importer as a JSON Lines stream."""
    text = "".join(f"{json.dumps(r)}\n" if not isinstance(r, str) else r for r in records)
    return list(importer.read(io.StringIO(text)))


def test_it_satisfies_the_importer_protocol(importer):
    assert isinstance(importer, ConversationImporter)
    assert importer.provider == "generic-jsonl"


def test_a_user_message(importer):
    (event,) = read(importer, [{"id": "1", "role": "user", "content": "hello"}])
    assert event.type is EventType.USER_MESSAGE
    assert event.source_id == "1"
    assert event.source_type == "user"
    assert event.text == "hello"
    assert event.provider == "generic-jsonl"


def test_an_assistant_message(importer):
    (event,) = read(importer, [{"role": "assistant", "content": "hi"}])
    assert event.type is EventType.ASSISTANT_MESSAGE
    assert event.text == "hi"


def test_a_system_message(importer):
    (event,) = read(importer, [{"role": "system", "content": "be brief"}])
    assert event.type is EventType.SYSTEM_MESSAGE


def test_a_developer_message(importer):
    (event,) = read(importer, [{"role": "developer", "content": "internal"}])
    assert event.type is EventType.DEVELOPER_MESSAGE


def test_a_tool_call(importer):
    (event,) = read(
        importer,
        [
            {
                "type": "tool_call",
                "tool_name": "search",
                "tool_call_id": "c1",
                "arguments": {"q": "x"},
            }
        ],
    )
    assert event.type is EventType.TOOL_CALL
    assert event.tool_name == "search"
    assert event.tool_call_id == "c1"
    assert event.raw["arguments"] == {"q": "x"}, (
        "arguments are not a modelled field, so raw holds them"
    )


def test_a_tool_result(importer):
    (event,) = read(
        importer,
        [{"type": "tool_result", "tool_name": "search", "tool_call_id": "c1", "content": "3 hits"}],
    )
    assert event.type is EventType.TOOL_RESULT
    assert event.text == "3 hits"


def test_the_tool_role_reads_as_a_tool_result(importer):
    """A generic convention, not a fact about any provider.

    In JSON conversation formats a tool-role message generally carries what a
    tool returned rather than a request to run one, and that is the reading
    ``Role.TOOL`` was defined against. It is the default for sources nobody has
    verified; a provider adapter classifies tool events from its own format.
    """
    (event,) = read(importer, [{"role": "tool", "tool_name": "grep", "content": "2 matches"}])
    assert event.type is EventType.TOOL_RESULT
    assert event.source_type == "tool", "the source's own word is kept either way"


def test_the_tool_convention_is_a_default_not_a_rule():
    """A source where a tool-role record is the call, not the result, says so."""
    importer = JsonLinesImporter(roles={"tool": EventType.TOOL_CALL})
    (event,) = read(importer, [{"role": "tool", "tool_name": "grep"}])
    assert event.type is EventType.TOOL_CALL


def test_an_explicit_type_beats_a_role(importer):
    (event,) = read(importer, [{"type": "tool_call", "role": "assistant", "tool_name": "run"}])
    assert event.type is EventType.TOOL_CALL
    assert event.source_type == "tool_call"


def test_an_unknown_role_becomes_other_and_keeps_its_name(importer):
    """Unrecognised is not the same as unimportant."""
    (event,) = read(importer, [{"role": "critic", "content": "needs work"}])
    assert event.type is EventType.OTHER
    assert event.source_type == "critic"
    assert event.text == "needs work"
    assert event.raw["role"] == "critic"


def test_extra_roles_are_configured_not_hardcoded():
    """A source saying 'human' is handled by the caller, not by a branch in the adapter."""
    importer = JsonLinesImporter(roles={"human": EventType.USER_MESSAGE})
    (event,) = read(importer, [{"role": "human", "content": "hi"}])
    assert event.type is EventType.USER_MESSAGE


def test_unknown_fields_are_preserved_verbatim(importer):
    record = {"role": "user", "content": "hi", "weird": {"deep": [1, {"x": None}]}, "n": 7}
    (event,) = read(importer, [record])
    assert event.raw == record


def test_ordering_is_file_order(importer):
    events = read(importer, [{"role": "user", "content": str(i)} for i in range(20)])
    assert [e.text for e in events] == [str(i) for i in range(20)]


def test_an_iso_timestamp_with_an_offset(importer):
    (event,) = read(importer, [{"role": "user", "timestamp": "2024-03-01T12:00:00+02:00"}])
    assert event.timestamp == datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def test_a_zulu_timestamp(importer):
    (event,) = read(importer, [{"role": "user", "timestamp": "2024-03-01T10:00:00Z"}])
    assert event.timestamp == datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def test_epoch_seconds(importer):
    (event,) = read(importer, [{"role": "user", "created_at": 1709287200}])
    assert event.timestamp == datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def test_a_naive_timestamp_is_refused_not_guessed(importer):
    """It could be any timezone, and guessing silently reorders history."""
    (event,) = read(importer, [{"role": "user", "timestamp": "2024-03-01T10:00:00"}])
    assert event.timestamp is None
    assert event.raw["timestamp"] == "2024-03-01T10:00:00", "the original is still readable"


@pytest.mark.parametrize("value", ["", "not a date", None, True, [], {}])
def test_unusable_timestamps_become_none(value):
    assert parse_timestamp(value) is None


def test_a_missing_timestamp_is_none_never_now(importer):
    """Import time is a fact about the import, not about the conversation."""
    (event,) = read(importer, [{"role": "user", "content": "hi"}])
    assert event.timestamp is None


def test_a_missing_id_stays_missing_on_the_event(importer):
    """The adapter does not invent identity. The writer names the archive record."""
    (event,) = read(importer, [{"role": "user", "content": "hi"}])
    assert event.source_id is None


def test_a_numeric_id_is_stringified(importer):
    (event,) = read(importer, [{"id": 42, "role": "user"}])
    assert event.source_id == "42"


def test_an_unusable_id_is_left_alone(importer):
    (event,) = read(importer, [{"id": {"nested": "id"}, "role": "user"}])
    assert event.source_id is None
    assert event.raw["id"] == {"nested": "id"}


def test_non_text_content_is_not_forced_into_text(importer):
    """Content blocks are provider-specific. Guessing their shape is how you lose data."""
    blocks = [{"type": "text", "text": "a"}, {"type": "image", "url": "b"}]
    (event,) = read(importer, [{"role": "user", "content": blocks}])
    assert event.text is None
    assert event.raw["content"] == blocks


def test_parent_references_are_preserved_but_not_validated(importer):
    (event,) = read(importer, [{"id": "b", "parent_id": "a", "role": "user"}])
    assert event.parent_id == "a"


def test_field_names_are_configurable():
    fields = FieldMap(id=("msg_uuid",), content=("payload",), role=("kind",))
    importer = JsonLinesImporter(fields)
    (event,) = read(importer, [{"msg_uuid": "z", "kind": "user", "payload": "hi"}])
    assert (event.source_id, event.type, event.text) == ("z", EventType.USER_MESSAGE, "hi")


def test_the_first_present_key_wins(importer):
    (event,) = read(importer, [{"uuid": "u", "role": "user", "text": "from text"}])
    assert event.source_id == "u"
    assert event.text == "from text"


def test_invalid_json_is_reported_with_its_line(importer):
    results = read(importer, ['{"role":"user","content":"ok"}\n', "{not json}\n"])
    assert isinstance(results[0], RawEvent)
    bad = results[1]
    assert isinstance(bad, MalformedRecord)
    assert bad.position == 1
    assert bad.location == "line 2"
    assert bad.reason == "invalid_json"
    assert "{not json}" in bad.text


def test_a_non_object_line_is_reported(importer):
    (bad,) = read(importer, ["[1, 2, 3]\n"])
    assert isinstance(bad, MalformedRecord)
    assert bad.reason == "not_an_object"
    assert "list" in bad.detail


def test_a_bad_line_does_not_stop_the_read(importer):
    """The generator stays alive because a failure is a value, not an exception."""
    results = read(
        importer,
        ['{"role":"user","content":"a"}\n', "broken\n", '{"role":"user","content":"b"}\n'],
    )
    assert [type(r).__name__ for r in results] == ["RawEvent", "MalformedRecord", "RawEvent"]
    assert results[2].text == "b"


def test_a_truncated_final_line_is_located(importer):
    """A cut-off export ends mid-record, which is reported at that record."""
    results = read(importer, ['{"role":"user","content":"a"}\n', '{"role":"user","cont'])
    assert isinstance(results[1], MalformedRecord)
    assert results[1].location == "line 2"


def test_blank_lines_are_reported_separately(importer):
    results = read(importer, ['{"role":"user"}\n', "\n", "   \n"])
    assert [r.reason for r in results[1:]] == ["blank", "blank"]


def test_an_empty_source_yields_nothing(importer):
    assert list(importer.read(io.StringIO(""))) == []


def test_detection_accepts_json_lines(importer):
    assert importer.detect('{"role": "user"}\n{"role": "assistant"}')
    assert importer.detect('\n\n{"a": 1}')


def test_detection_rejects_what_it_cannot_read(importer):
    assert not importer.detect("[{'role': 'user'}]")
    assert not importer.detect("role,content\nuser,hi")
    assert not importer.detect("")


def test_detection_accepts_a_record_cut_by_the_sample_boundary(importer):
    """A sample ends where it ends. An unclosed opening object is still JSON Lines."""
    assert importer.detect('{"role": "user", "content": "a very long message that is c')


# ----------------------------------------------------------------------
# Identity
#
# Four concepts that must not be confused: the identity of a logical source,
# a fingerprint of what a source contains, the archive's own ordering, and the
# id given to an event whose provider supplied none.


def test_a_content_digest_fingerprints_content():
    assert content_digest_for_bytes(b"same") == content_digest_for_bytes(b"same")
    assert content_digest_for_bytes(b"same") != content_digest_for_bytes(b"different")


def test_a_content_digest_is_not_a_source_identity(tmp_path):
    """The distinction this correction turns on.

    Two separate exports holding identical bytes share a fingerprint. They are
    still two sources, and only a caller can say otherwise.
    """
    content = b'{"role":"user"}\n'
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first.write_bytes(content)
    second.write_bytes(content)

    assert content_digest_for_path(first) == content_digest_for_path(second)
    assert default_source_key(first) != default_source_key(second)


def test_a_content_digest_reads_in_bounded_chunks(tmp_path):
    """Digesting must not undo the streaming the import exists to preserve."""
    path = tmp_path / "big.jsonl"
    payload = b"x" * (DIGEST_CHUNK_BYTES * 4)
    path.write_bytes(payload)

    assert content_digest_for_path(path) == content_digest_for_bytes(payload)


def test_the_default_source_key_survives_an_equivalent_path(tmp_path):
    """A relative and an absolute route to one file are one file."""
    path = tmp_path / "nested" / "export.jsonl"
    path.parent.mkdir()
    path.write_bytes(b'{"role":"user"}\n')

    indirect = tmp_path / "nested" / ".." / "nested" / "export.jsonl"
    assert default_source_key(indirect) == default_source_key(path)


def test_a_synthetic_id_is_scoped_to_a_logical_source_and_position():
    assert synthetic_id("source-a", 0) != synthetic_id("source-b", 0)
    assert synthetic_id("source-a", 0) != synthetic_id("source-a", 1)
    assert synthetic_id("source-a", 7) == synthetic_id("source-a", 7)
    assert synthetic_id("source-a", 7).startswith(f"{SYNTHETIC_PREFIX}-")


def test_a_synthetic_id_encodes_rather_than_embeds_the_key():
    """The key may be long or awkward; the id stays fixed width."""
    identifier = synthetic_id("a very long logical source key with spaces and /slashes/", 3)
    prefix, fingerprint, position = identifier.split("-")

    assert prefix == SYNTHETIC_PREFIX
    assert len(fingerprint) == FINGERPRINT_LENGTH
    assert position == "00000003"


def test_a_fingerprint_is_deterministic_and_differs_between_keys():
    assert source_key_fingerprint("k") == source_key_fingerprint("k")
    assert source_key_fingerprint("k") != source_key_fingerprint("k2")
    assert len(source_key_fingerprint("k")) == FINGERPRINT_LENGTH


def test_synthetic_ids_sort_in_position_order():
    """Zero padding, so an id ordering is never a lexicographic surprise."""
    ids = [synthetic_id("k", i) for i in (0, 2, 10, 100)]
    assert ids == sorted(ids)


def test_a_negative_position_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        synthetic_id("k", -1)
