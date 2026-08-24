"""Importing a conversation into the archive.

End to end: a file on disk becomes archive records, and everything the import
could not read cleanly comes back in the report rather than disappearing.
"""

import json
import shutil

import pytest

from open_context.archive import Archive
from open_context.importers import (
    SYNTHETIC_PREFIX,
    DuplicateSourceIdError,
    EventType,
    IncompleteImportError,
    JsonLinesImporter,
    MalformedRecordError,
    RawEvent,
    UnsupportedSourceError,
    content_digest_for_path,
    default_source_key,
    import_events,
    import_path,
    stream_events,
    synthetic_id,
)

pytestmark = pytest.mark.integration

SESSION = "ses_" + "e" * 24


@pytest.fixture
def archive(tmp_path):
    return Archive(tmp_path / "archive")


@pytest.fixture
def write_source(tmp_path):
    def write(records, name="conversation.jsonl"):
        path = tmp_path / name
        path.write_text(
            "".join(r if isinstance(r, str) else f"{json.dumps(r)}\n" for r in records),
            encoding="utf-8",
        )
        return path

    return write


CONVERSATION = [
    {"id": "m1", "role": "system", "content": "Be brief.", "timestamp": "2024-03-01T10:00:00Z"},
    {
        "id": "m2",
        "role": "user",
        "content": "We should use PostgreSQL.",
        "timestamp": "2024-03-01T10:00:05Z",
    },
    {"id": "m3", "role": "assistant", "content": "Noted.", "timestamp": "2024-03-01T10:00:09Z"},
    {
        "id": "m4",
        "type": "tool_call",
        "tool_name": "search",
        "tool_call_id": "c1",
        "arguments": {"q": "pg"},
    },
    {
        "id": "m5",
        "type": "tool_result",
        "tool_name": "search",
        "tool_call_id": "c1",
        "content": "3 hits",
    },
]


# ----------------------------------------------------------------------
# The happy path


def test_a_conversation_imports_into_the_archive(archive, write_source):
    report = import_path(archive, SESSION, write_source(CONVERSATION))

    assert report.events_read == 5
    assert report.events_written == 5
    assert report.session_id == SESSION
    assert report.provider == "generic-jsonl"
    assert (report.first_seq, report.last_seq) == (0, 4)
    assert not report.appended_to_existing
    assert archive.open(SESSION).count == 5


def test_every_event_kind_survives_the_round_trip(archive, write_source):
    import_path(archive, SESSION, write_source(CONVERSATION))
    events = list(stream_events(archive.open(SESSION)))

    assert [e.type for e in events] == [
        EventType.SYSTEM_MESSAGE,
        EventType.USER_MESSAGE,
        EventType.ASSISTANT_MESSAGE,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
    ]
    assert events[1].text == "We should use PostgreSQL."
    assert events[3].tool_name == "search"
    assert events[4].tool_call_id == "c1"


def test_the_report_counts_what_it_stored(archive, write_source):
    report = import_path(archive, SESSION, write_source(CONVERSATION))
    assert report.by_type == {
        "system_message": 1,
        "user_message": 1,
        "assistant_message": 1,
        "tool_call": 1,
        "tool_result": 1,
    }


def test_ordering_is_preserved_exactly(archive, write_source):
    """Source order is archive order. There is no sort step to get this wrong."""
    records = [{"id": f"m{i}", "role": "user", "content": f"turn {i}"} for i in range(200)]
    import_path(archive, SESSION, write_source(records))

    log = archive.open(SESSION)
    assert [r.seq for r in log] == list(range(200))
    assert [e.text for e in stream_events(log)] == [f"turn {i}" for i in range(200)]


def test_timestamps_are_preserved(archive, write_source):
    from datetime import UTC, datetime

    import_path(archive, SESSION, write_source(CONVERSATION))
    events = list(stream_events(archive.open(SESSION)))
    assert events[0].timestamp == datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)
    assert events[3].timestamp is None, "the source gave none, so none is stored"


def test_provider_metadata_is_recorded_on_every_event(archive, write_source):
    import_path(archive, SESSION, write_source(CONVERSATION))
    events = list(stream_events(archive.open(SESSION)))
    assert {e.provider for e in events} == {"generic-jsonl"}
    assert [e.source_type for e in events[:3]] == ["system", "user", "assistant"]


def test_the_archive_kind_is_the_event_type(archive, write_source):
    """So an archive can be filtered without parsing every payload."""
    import_path(archive, SESSION, write_source(CONVERSATION))
    kinds = [r.kind for r in archive.open(SESSION)]
    assert kinds == [
        "system_message",
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_result",
    ]


# ----------------------------------------------------------------------
# Raw payload preservation


def test_unknown_provider_fields_survive_in_the_archive(archive, write_source):
    """The normalized event is an interpretation, not a replacement."""
    record = {
        "id": "m1",
        "role": "user",
        "content": "hi",
        "provider_only": {"weights": [0.1, 0.2], "flag": None},
        "schema_version": 9,
    }
    import_path(archive, SESSION, write_source([record]))

    stored = archive.open(SESSION).read(0)
    assert stored.payload["raw"] == record
    assert RawEvent.from_payload(stored.payload).raw["provider_only"]["weights"] == [0.1, 0.2]


def test_an_entirely_unrecognised_record_still_imports(archive, write_source):
    """A shape nothing here anticipated is stored, not refused."""
    record = {"kind": "checkpoint", "state": {"branch": "main"}, "n": 3}
    report = import_path(archive, SESSION, write_source([record]))

    assert report.events_written == 1
    assert report.unknown_source_types == 1
    event = next(iter(stream_events(archive.open(SESSION))))
    assert event.type is EventType.OTHER
    assert event.raw == record


def test_unknown_event_kinds_are_named_in_the_report(archive, write_source):
    report = import_path(
        archive,
        SESSION,
        write_source([{"role": "critic", "content": "a"}, {"role": "judge", "content": "b"}]),
    )
    assert report.unknown_source_types == 2
    assert set(report.unknown_source_type_sample) == {"critic", "judge"}


def test_the_archive_stays_verifiable_after_import(archive, write_source):
    import_path(archive, SESSION, write_source(CONVERSATION))
    log = archive.open(SESSION)
    assert log.verify().ok
    assert log.recovery.clean


# ----------------------------------------------------------------------
# Identity


def test_events_without_an_id_get_a_synthetic_one(archive, write_source):
    source = write_source([{"role": "user", "content": f"m{i}"} for i in range(3)])
    report = import_path(archive, SESSION, source)

    assert report.without_source_id == 3
    assert report.source_key == default_source_key(source), "the report names the identity used"
    assert [r.id for r in archive.open(SESSION)] == [
        synthetic_id(report.source_key, i) for i in range(3)
    ]
    assert all(r.id.startswith(f"{SYNTHETIC_PREFIX}-") for r in archive.open(SESSION))


def test_a_synthesised_id_is_not_claimed_to_be_the_providers(archive, write_source):
    """The prefix marks it as made up, and the event still says so."""
    import_path(archive, SESSION, write_source([{"role": "user", "content": "hi"}]))
    stored = archive.open(SESSION).read(0)

    assert stored.id.startswith(f"{SYNTHETIC_PREFIX}-")
    assert stored.payload["source_id"] is None, "the event still says the provider gave none"


def test_synthetic_ids_are_deterministic_for_the_same_source(archive, write_source, tmp_path):
    """Two archives of one export must be comparable, so nothing may be random."""
    source = write_source([{"role": "user", "content": f"m{i}"} for i in range(3)])
    import_path(archive, SESSION, source)

    other = Archive(tmp_path / "second-archive")
    import_path(other, SESSION, source)

    assert [r.id for r in other.open(SESSION)] == [r.id for r in archive.open(SESSION)]


def test_a_second_import_into_one_session_does_not_reuse_ids(archive, write_source):
    """The bug this scheme exists to prevent.

    A position counted from the start of each import gave the first unidentified
    event of every import the same id, which the archive reads as one item
    written twice.
    """
    first = write_source([{"role": "user", "content": "from the first file"}], name="a.jsonl")
    second = write_source([{"role": "user", "content": "from the second file"}], name="b.jsonl")

    import_path(archive, SESSION, first)
    import_path(archive, SESSION, second)

    ids = [r.id for r in archive.open(SESSION)]
    assert len(set(ids)) == 2, f"unrelated events share an identity: {ids}"


def test_distinct_logical_sources_get_distinct_ids(archive, write_source):
    """Same shape, same positions: the ids must still differ."""
    left = write_source(
        [{"role": "user", "content": f"left {i}"} for i in range(5)], name="l.jsonl"
    )
    right = write_source(
        [{"role": "user", "content": f"right {i}"} for i in range(5)], name="r.jsonl"
    )

    import_path(archive, SESSION, left)
    import_path(archive, SESSION, right)

    ids = [r.id for r in archive.open(SESSION)]
    assert len(set(ids)) == 10
    assert default_source_key(left) != default_source_key(right)


def test_byte_identical_files_are_two_sources_by_default(archive, write_source):
    """Content cannot decide this. Two files are two sources until a caller says otherwise.

    The bug this correction fixes: a key derived from content alone gave these
    the same identity, which the archive reads as one item written twice.
    """
    records = [{"role": "user", "content": "hello"}]
    left = write_source(records, name="copy-a.jsonl")
    right = write_source(records, name="copy-b.jsonl")
    assert left.read_bytes() == right.read_bytes()

    import_path(archive, SESSION, left)
    import_path(archive, SESSION, right)

    ids = [r.id for r in archive.open(SESSION)]
    assert len(set(ids)) == 2, f"identical content is not identical identity: {ids}"


def test_a_caller_can_declare_two_copies_to_be_one_source(archive, write_source):
    """The judgement the filesystem cannot make, made explicitly."""
    records = [{"role": "user", "content": "hello"}]
    left = write_source(records, name="laptop.jsonl")
    right = write_source(records, name="desktop.jsonl")

    import_path(archive, SESSION, left, source_key="export:2024-03-01")
    import_path(archive, SESSION, right, source_key="export:2024-03-01")

    ids = [r.id for r in archive.open(SESSION)]
    assert ids[0] == ids[1], "declared the same source, so the same items"


def test_a_caller_can_declare_one_file_to_be_two_sources(archive, write_source):
    """And the other way: the same bytes imported as two distinct ingestions."""
    source = write_source([{"role": "user", "content": "hello"}])

    import_path(archive, SESSION, source, source_key="run:1")
    import_path(archive, SESSION, source, source_key="run:2")

    ids = [r.id for r in archive.open(SESSION)]
    assert ids[0] != ids[1]


def test_an_explicit_key_overrides_the_default(archive, write_source):
    source = write_source([{"role": "user", "content": "hi"}])
    report = import_path(archive, SESSION, source, source_key="chosen")

    assert report.source_key == "chosen"
    assert archive.open(SESSION).read(0).id == synthetic_id("chosen", 0)


def test_the_default_key_names_the_file_and_its_content(archive, write_source):
    """Documented precisely, because a defaulted identity is still an identity."""
    source = write_source([{"role": "user", "content": "hi"}])
    key = default_source_key(source)

    assert key.startswith("file:")
    assert str(source.resolve()) in key
    assert content_digest_for_path(source) in key


def test_editing_a_source_makes_it_a_different_logical_source(archive, write_source, tmp_path):
    """Once the bytes change, position 0 no longer names the event it used to."""
    source = write_source([{"role": "user", "content": "first version"}])
    before = default_source_key(source)

    source.write_text('{"role":"user","content":"second version"}\n', encoding="utf-8")
    assert default_source_key(source) != before


def test_re_importing_one_source_repeats_its_ids(archive, write_source):
    """Identity is scoped to the source, so the same source is the same items.

    Two records sharing an id is exactly what the archive means by a repeated
    item, and this is the one case where that claim is true.
    """
    source = write_source([{"role": "user", "content": "hi"}])
    import_path(archive, SESSION, source)
    import_path(archive, SESSION, source)

    ids = [r.id for r in archive.open(SESSION)]
    assert ids[0] == ids[1]
    assert len(ids) == 2, "both records are kept; the archive is append-only"


def test_identity_does_not_come_from_the_archive_sequence(archive, write_source, tmp_path):
    """A source imported into a busy session keeps the ids it has when alone."""
    source = write_source([{"role": "user", "content": "hi"}])
    prior = write_source([{"id": f"m{i}", "role": "user"} for i in range(7)], name="prior.jsonl")

    import_path(archive, SESSION, prior)
    import_path(archive, SESSION, source)
    appended = archive.open(SESSION).read(7).id

    alone = Archive(tmp_path / "alone")
    import_path(alone, SESSION, source)
    assert appended == alone.open(SESSION).read(0).id


def test_provider_ids_are_untouched_by_the_synthetic_scheme(archive, write_source):
    """A source that names its events still names them."""
    import_path(archive, SESSION, write_source(CONVERSATION))
    assert [r.id for r in archive.open(SESSION)] == ["m1", "m2", "m3", "m4", "m5"]


def test_mixed_sources_keep_provider_ids_and_synthesise_only_the_rest(archive, write_source):
    records = [
        {"id": "given", "role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    import_path(archive, SESSION, write_source(records))
    ids = [r.id for r in archive.open(SESSION)]

    assert ids[0] == "given"
    assert ids[1].startswith(f"{SYNTHETIC_PREFIX}-")


def test_a_source_key_must_identify_something():
    for unusable in ["", "   ", "\t\n"]:
        with pytest.raises(ValueError, match="source key"):
            synthetic_id(unusable, 0)


def test_a_source_key_may_be_any_meaningful_string():
    """It is hashed into the id, so a caller is free to make it readable."""
    for usable in ["run 47", "export:2024-03-01#2", "/tmp/a b/c.jsonl", "ünïcode"]:
        assert synthetic_id(usable, 0).startswith(f"{SYNTHETIC_PREFIX}-")


def test_source_identity_and_content_digest_are_different_concepts(write_source):
    """Two files, same content: one digest, two identities."""
    records = [{"role": "user", "content": "hello"}]
    left = write_source(records, name="one.jsonl")
    right = write_source(records, name="two.jsonl")

    assert content_digest_for_path(left) == content_digest_for_path(right)
    assert default_source_key(left) != default_source_key(right)


def test_duplicate_ids_are_reported_and_both_are_kept(archive, write_source):
    """The archive stores a message and a later revision of it as two records."""
    records = [
        {"id": "m1", "role": "user", "content": "first"},
        {"id": "m1", "role": "user", "content": "revised"},
    ]
    report = import_path(archive, SESSION, write_source(records))

    assert report.duplicate_source_ids == 1
    assert report.duplicate_source_id_sample == ("m1",)
    assert report.events_written == 2, "reported, not discarded"
    log = archive.open(SESSION)
    assert [e.text for e in stream_events(log)] == ["first", "revised"]


def test_duplicate_ids_can_be_made_fatal(archive, write_source):
    records = [{"id": "m1", "role": "user"}, {"id": "m1", "role": "user"}]
    with pytest.raises(DuplicateSourceIdError) as info:
        import_path(archive, SESSION, write_source(records), on_duplicate="abort")
    assert info.value.source_id == "m1"
    assert info.value.position == 1


# ----------------------------------------------------------------------
# Malformed input


def test_a_malformed_record_stops_the_import_by_default(archive, write_source):
    """Importing three quarters of a conversation while believing it whole is worse."""
    source = write_source(['{"role":"user","content":"a"}\n', "{broken\n"])
    with pytest.raises(MalformedRecordError) as info:
        import_path(archive, SESSION, source)
    assert info.value.location == "line 2"
    assert info.value.position == 1


# ----------------------------------------------------------------------
# Partial imports
#
# An import is not atomic. The archive has append and extend and no delete,
# truncate, or rollback, so a failure partway cannot un-write what landed. These
# pin the contract that replaces the guarantee we cannot make.


def test_an_aborted_import_keeps_what_it_already_wrote(archive, write_source):
    source = write_source(
        [
            '{"id":"m1","role":"user","content":"a"}\n',
            '{"id":"m2","role":"user","content":"b"}\n',
            "{broken\n",
            '{"id":"m4","role":"user","content":"d"}\n',
        ]
    )
    with pytest.raises(MalformedRecordError):
        import_path(archive, SESSION, source)

    log = archive.open(SESSION)
    assert log.count == 2, "the records before the failure are still there"
    assert [e.text for e in stream_events(log)] == ["a", "b"]
    assert log.verify().ok, "and the archive is intact, not half-written"


def test_the_failure_says_exactly_what_was_written(archive, write_source):
    source = write_source(
        [
            '{"id":"m1","role":"user","content":"a"}\n',
            '{"id":"m2","role":"user","content":"b"}\n',
            "{broken\n",
        ]
    )
    with pytest.raises(MalformedRecordError) as info:
        import_path(archive, SESSION, source)

    report = info.value.report
    assert not report.completed, "the report never claims a whole import"
    assert not report.clean
    assert report.events_written == 2
    assert (report.first_seq, report.last_seq) == (0, 1)
    assert report.malformed_records == 1
    assert report.malformed_sample[0].location == "line 3"


def test_an_aborted_import_is_catchable_as_one_thing(archive, write_source):
    """A caller should not have to enumerate failure types to ask 'is my archive partial'."""
    source = write_source(['{"id":"m1","role":"user"}\n', "{broken\n"])
    with pytest.raises(IncompleteImportError) as info:
        import_path(archive, SESSION, source)
    assert info.value.report.events_written == 1
    assert "1 records were already written" in str(info.value)


def test_a_duplicate_abort_keeps_the_records_before_it(archive, write_source):
    records = [
        {"id": "m1", "role": "user", "content": "a"},
        {"id": "m2", "role": "user", "content": "b"},
        {"id": "m1", "role": "user", "content": "repeat"},
        {"id": "m4", "role": "user", "content": "d"},
    ]
    with pytest.raises(DuplicateSourceIdError) as info:
        import_path(archive, SESSION, write_source(records), on_duplicate="abort")

    report = info.value.report
    assert not report.completed
    assert report.events_written == 2
    assert report.duplicate_source_ids == 1
    assert archive.open(SESSION).count == 2


def test_the_offending_record_is_never_written(archive, write_source):
    """Whatever stopped the import is not in the archive; everything before it is."""
    records = [
        {"id": "m1", "role": "user", "content": "kept"},
        {"id": "m1", "role": "user", "content": "the duplicate"},
    ]
    with pytest.raises(DuplicateSourceIdError):
        import_path(archive, SESSION, write_source(records), on_duplicate="abort")

    assert [e.text for e in stream_events(archive.open(SESSION))] == ["kept"]


def test_a_failure_before_any_write_leaves_the_session_empty(archive, write_source):
    source = write_source(["{broken\n", '{"id":"m1","role":"user"}\n'])
    with pytest.raises(MalformedRecordError) as info:
        import_path(archive, SESSION, source)

    assert info.value.report.events_written == 0
    assert info.value.report.first_seq is None
    assert archive.open(SESSION).count == 0


def test_a_failed_import_can_be_continued_or_discarded(archive, write_source, tmp_path):
    """The documented ways out: append the rest, or drop the session directory.

    Atomicity is a filesystem operation on a session of its own, not a
    transaction system inside the runtime.
    """
    broken = write_source(['{"id":"m1","role":"user","content":"a"}\n', "{broken\n"])
    with pytest.raises(MalformedRecordError):
        import_path(archive, SESSION, broken)

    fixed = write_source([{"id": "m2", "role": "user", "content": "b"}], name="fixed.jsonl")
    resumed = import_path(archive, SESSION, fixed)
    assert resumed.completed
    assert resumed.appended_to_existing
    assert [e.text for e in stream_events(archive.open(SESSION))] == ["a", "b"]

    shutil.rmtree(tmp_path / "archive" / "sessions" / SESSION)
    assert not archive.exists(SESSION), "discarding a partial session is one rmtree"


def test_a_completed_import_says_so(archive, write_source):
    records = [
        {"id": f"m{i}", "role": "user", "content": "hi", "timestamp": "2024-03-01T10:00:00Z"}
        for i in range(3)
    ]
    report = import_path(archive, SESSION, write_source(records))
    assert report.completed
    assert report.clean, "a source with ids and timestamps and nothing broken"


def test_a_source_with_reservations_is_complete_but_not_clean(archive, write_source):
    """Two different questions: did it finish, and is there anything to look at."""
    report = import_path(archive, SESSION, write_source(CONVERSATION))
    assert report.completed
    assert not report.clean, "the two tool events carry no timestamp"
    assert report.without_timestamp == 2


def test_malformed_records_can_be_skipped_but_never_silently(archive, write_source):
    source = write_source(
        ['{"role":"user","content":"a"}\n', "{broken\n", '{"role":"user","content":"b"}\n']
    )
    report = import_path(archive, SESSION, source, on_malformed="skip")

    assert report.events_written == 2
    assert report.malformed_records == 1
    assert report.malformed_sample[0].location == "line 2"
    assert not report.clean


def test_a_truncated_source_is_reported_at_the_cut(archive, write_source):
    source = write_source(['{"id":"m1","role":"user","content":"a"}\n', '{"id":"m2","rol'])
    report = import_path(archive, SESSION, source, on_malformed="skip")

    assert report.events_written == 1
    assert report.malformed_records == 1
    assert report.malformed_sample[0].location == "line 2"


def test_blank_lines_are_counted_not_treated_as_records(archive, write_source):
    source = write_source(
        ['{"role":"user","content":"a"}\n', "\n", '{"role":"user","content":"b"}\n']
    )
    report = import_path(archive, SESSION, source)

    assert report.events_written == 2
    assert report.blank_lines_skipped == 1
    assert report.malformed_records == 0


def test_a_missing_file_is_refused(archive, tmp_path):
    with pytest.raises(UnsupportedSourceError, match="no file at"):
        import_path(archive, SESSION, tmp_path / "absent.jsonl")


def test_a_source_no_importer_recognises_is_refused(archive, tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("role,content\nuser,hi\n", encoding="utf-8")
    with pytest.raises(UnsupportedSourceError, match="no importer recognises"):
        import_path(archive, SESSION, path)


def test_a_non_utf8_source_is_refused_with_its_path(archive, tmp_path):
    path = tmp_path / "latin.jsonl"
    path.write_bytes(b'{"role":"user","content":"caf\xe9"}\n')
    with pytest.raises(UnsupportedSourceError, match="not valid utf-8"):
        import_path(archive, SESSION, path)


# ----------------------------------------------------------------------
# Timestamps and tool pairing


def test_missing_and_unusable_timestamps_are_distinguished(archive, write_source):
    """Different problems with different fixes."""
    records = [
        {"role": "user", "content": "no timestamp field at all"},
        {"role": "user", "content": "naive", "timestamp": "2024-03-01T10:00:00"},
    ]
    report = import_path(archive, SESSION, write_source(records))

    assert report.without_timestamp == 2
    assert report.unusable_timestamps == 1, "only the second offered one we refused"


def test_an_unmatched_tool_result_is_reported_not_repaired(archive, write_source):
    """Pairing a result to a call it does not name would be a guess about what happened."""
    records = [
        {"id": "t1", "type": "tool_result", "tool_name": "search", "tool_call_id": "missing"},
        {"id": "t2", "type": "tool_result", "tool_name": "search"},
    ]
    report = import_path(archive, SESSION, write_source(records))

    assert report.unmatched_tool_results == 2
    assert report.events_written == 2


def test_a_matched_tool_pair_is_not_flagged(archive, write_source):
    report = import_path(archive, SESSION, write_source(CONVERSATION))
    assert report.unmatched_tool_results == 0


def test_a_tool_event_without_a_name_is_reported(archive, write_source):
    records = [{"type": "tool_call", "tool_call_id": "c1"}]
    report = import_path(archive, SESSION, write_source(records))
    assert report.tool_events_without_name == 1
    assert report.events_written == 1


# ----------------------------------------------------------------------
# Empty and repeated imports


def test_an_empty_conversation_imports_as_an_empty_archive(archive, write_source):
    report = import_path(archive, SESSION, write_source([]), importer=JsonLinesImporter())

    assert report.events_read == 0
    assert report.events_written == 0
    assert (report.first_seq, report.last_seq) == (None, None)
    assert report.clean
    assert archive.open(SESSION).count == 0


def test_a_file_of_only_blank_lines_is_not_a_conversation(archive, write_source):
    report = import_path(archive, SESSION, write_source(["\n", "\n"]), importer=JsonLinesImporter())
    assert report.events_written == 0
    assert report.blank_lines_skipped == 2


def test_re_importing_appends_a_second_copy(archive, write_source):
    """The archive is append-only and has no idea the same file arrived twice."""
    source = write_source(CONVERSATION)
    first = import_path(archive, SESSION, source)
    second = import_path(archive, SESSION, source)

    assert not first.appended_to_existing
    assert second.appended_to_existing, "the caller is told the session was not empty"
    assert (second.first_seq, second.last_seq) == (5, 9)
    assert archive.open(SESSION).count == 10


def test_importing_into_a_new_session_is_how_you_get_one_copy(archive, write_source):
    source = write_source(CONVERSATION)
    import_path(archive, SESSION, source)
    import_path(archive, "ses_" + "f" * 24, source)

    assert archive.open(SESSION).count == 5
    assert archive.open("ses_" + "f" * 24).count == 5


def test_an_import_can_continue_an_existing_session(archive, write_source):
    import_path(archive, SESSION, write_source(CONVERSATION[:2]))
    report = import_path(archive, SESSION, write_source(CONVERSATION[2:], name="rest.jsonl"))

    assert report.first_seq == 2
    assert [e.type for e in stream_events(archive.open(SESSION))][-1] is EventType.TOOL_RESULT


# ----------------------------------------------------------------------
# Seams


def test_an_importer_can_be_named_explicitly(archive, write_source):
    report = import_path(archive, SESSION, write_source(CONVERSATION), importer=JsonLinesImporter())
    assert report.events_written == 5


def test_events_can_be_imported_without_a_file(archive):
    """The pipeline takes events, so an adapter need not be file-shaped."""
    log = archive.create(SESSION)
    events = [
        RawEvent(type=EventType.USER_MESSAGE, provider="in-memory", text="hi", source_id="a"),
        RawEvent(type=EventType.ASSISTANT_MESSAGE, provider="in-memory", text="hello"),
    ]
    report = import_events(log, events, provider="in-memory", source_key="run-4718")

    assert report.events_written == 2
    assert report.provider == "in-memory"
    assert [e.text for e in stream_events(archive.open(SESSION))] == ["hi", "hello"]


def test_a_source_key_is_required_so_it_cannot_be_forgotten(archive):
    """An event with no id of its own is identified by it, and that cannot be optional."""
    log = archive.create(SESSION)
    with pytest.raises(TypeError, match="source_key"):
        import_events(log, [], provider="in-memory")


def test_a_caller_supplied_source_key_scopes_synthetic_ids(archive):
    """A live stream has no file to digest, so it names its own source."""
    log = archive.create(SESSION)
    event = RawEvent(type=EventType.USER_MESSAGE, provider="live", text="hi")
    import_events(log, [event], provider="live", source_key="run-1")
    import_events(log, [event], provider="live", source_key="run-2")

    ids = [r.id for r in archive.open(SESSION)]
    assert ids == [synthetic_id("run-1", 0), synthetic_id("run-2", 0)]
    assert len(set(ids)) == 2


def test_the_pipeline_never_sees_a_provider_name(archive, write_source):
    """Provider knowledge stops at the adapter.

    A custom adapter with a name nothing in the core has heard of imports with
    no special handling, which is the property that keeps the core neutral.
    """

    class WeatherImporter:
        provider = "weather-station-v3"

        def detect(self, sample: str) -> bool:
            return True

        def read(self, stream):
            for line in stream:
                yield RawEvent(
                    type=EventType.OTHER,
                    provider=self.provider,
                    raw={"line": line.strip()},
                    source_type="reading",
                )

    path = write_source(['{"unused": 1}\n'], name="weather.txt")
    log = archive.create(SESSION)
    with path.open(encoding="utf-8") as handle:
        report = import_events(
            log,
            WeatherImporter().read(handle),
            provider="weather-station-v3",
            source_key=default_source_key(path),
        )

    assert report.provider == "weather-station-v3"
    assert report.events_written == 1


# ----------------------------------------------------------------------
# What import must not do


def test_import_creates_no_semantic_state(archive, write_source, repo):
    """Phase 4 is ingestion. 'We should use PostgreSQL' is a message, not a Decision."""
    from open_context.models import Session

    repo.add_session(Session(id=SESSION, source="generic-jsonl"))
    import_path(archive, SESSION, write_source(CONVERSATION))

    counts = repo.counts()
    assert counts["state_items"] == 0
    assert counts["evidence"] == 0
    assert counts["snapshots"] == 0


def test_import_does_not_duplicate_the_conversation_into_sqlite(archive, write_source, repo):
    """The archive is the source of truth. SQLite holds the session, not the history."""
    from open_context.models import Session

    repo.add_session(Session(id=SESSION, source="generic-jsonl"))
    import_path(archive, SESSION, write_source(CONVERSATION))

    assert repo.counts()["messages"] == 0
    assert archive.open(SESSION).count == 5
    assert repo.get_session(SESSION).source == "generic-jsonl"
