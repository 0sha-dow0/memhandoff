"""Archive lifecycle, reads, and ordering."""

import pytest

from open_context.archive import (
    Archive,
    ArchiveExistsError,
    ArchiveNotFoundError,
    RecordNotFoundError,
)

pytestmark = pytest.mark.integration

SESSION = "ses_" + "a" * 24


@pytest.fixture
def archive(tmp_path):
    return Archive(tmp_path / "archive")


@pytest.fixture
def log(archive):
    return archive.create(SESSION)


def fill(log, count, *, prefix="msg"):
    for i in range(count):
        log.append(f"{prefix}_{i}", {"role": "user", "content": f"message {i}"})
    return log


def test_create_then_reopen(archive, log):
    fill(log, 3)
    reopened = archive.open(SESSION)
    assert reopened.count == 3
    assert reopened.session_id == SESSION
    assert reopened.recovery.clean


def test_creating_twice_is_refused(archive, log):
    with pytest.raises(ArchiveExistsError):
        archive.create(SESSION)


def test_opening_a_missing_archive_raises(archive):
    with pytest.raises(ArchiveNotFoundError):
        archive.open("ses_nope")


def test_open_or_create(archive):
    first = archive.open_or_create(SESSION)
    first.append("msg_0", {"content": "x"})
    assert archive.open_or_create(SESSION).count == 1


def test_empty_archive(archive, log):
    assert log.count == 0
    assert list(log) == []
    assert list(log.range(0, 10)) == []
    assert list(log.tail(5)) == []
    assert log.verify().ok
    assert archive.open(SESSION).count == 0


def test_append_assigns_sequential_positions(log):
    for i in range(4):
        assert log.append(f"msg_{i}", {"content": i}).seq == i
    assert log.count == 4


def test_read_by_position(log):
    fill(log, 5)
    assert log.read(0).payload["content"] == "message 0"
    assert log.read(4).payload["content"] == "message 4"
    assert log.read(2).id == "msg_2"


def test_read_out_of_range_raises(log):
    fill(log, 2)
    with pytest.raises(RecordNotFoundError):
        log.read(2)
    with pytest.raises(RecordNotFoundError):
        log.read(-1)


def test_iteration_preserves_order(archive, log):
    fill(log, 50)
    seqs = [record.seq for record in archive.open(SESSION)]
    assert seqs == list(range(50))


def test_range_reads_a_window(log):
    fill(log, 20)
    assert [r.seq for r in log.range(5, 9)] == [5, 6, 7, 8]
    assert [r.seq for r in log.range(18)] == [18, 19]
    assert [r.seq for r in log.range(5, 5)] == []
    assert [r.seq for r in log.range(15, 999)] == [15, 16, 17, 18, 19], "stop is clamped"


def test_range_rejects_negative_start(log):
    fill(log, 3)
    with pytest.raises(ValueError, match="must not be negative"):
        list(log.range(-1))


def test_window_around_a_position(log):
    fill(log, 20)
    assert [r.seq for r in log.window(10, before=2, after=2)] == [8, 9, 10, 11, 12]
    assert [r.seq for r in log.window(0, before=5, after=1)] == [0, 1], "clamped at the start"
    assert [r.seq for r in log.window(19, before=1, after=5)] == [18, 19], "clamped at the end"


def test_tail(log):
    fill(log, 10)
    assert [r.seq for r in log.tail(3)] == [7, 8, 9]
    assert [r.seq for r in log.tail(99)] == list(range(10))
    assert list(log.tail(0)) == []


def test_extend_writes_a_batch(archive, log):
    written = log.extend((f"msg_{i}", {"content": i}) for i in range(100))
    assert written == 100
    reopened = archive.open(SESSION)
    assert reopened.count == 100
    assert [r.seq for r in reopened.range(0, 3)] == [0, 1, 2]
    assert reopened.verify().ok


def test_append_after_reopen_continues_the_sequence(archive, log):
    fill(log, 3)
    reopened = archive.open(SESSION)
    assert reopened.append("msg_3", {"content": "later"}).seq == 3
    assert archive.open(SESSION).count == 4


def test_duplicate_ids_are_stored_as_distinct_records(log):
    """Identity is preserved as given. Two records may share an id, for instance
    a message and a later revision of it. Position is what distinguishes them."""
    log.append("msg_same", {"content": "first"})
    log.append("msg_same", {"content": "second"})
    assert log.count == 2
    assert log.read(0).id == log.read(1).id == "msg_same"
    assert log.read(0).payload["content"] == "first"


def test_payloads_are_stored_verbatim(log):
    payload = {"content": 'line one\nline two\t"quoted"', "unicode": "\u00e9\u4e2d", "n": None}
    log.append("msg_0", payload)
    assert log.read(0).payload == payload


def test_kinds_are_preserved(log):
    log.append("msg_0", {"content": "hi"}, kind="message")
    log.append("tool_0", {"stdout": "3 matches"}, kind="tool_result")
    assert [r.kind for r in log] == ["message", "tool_result"]


def test_sessions_are_listed_and_isolated(archive):
    first = archive.create("ses_" + "1" * 24)
    second = archive.create("ses_" + "2" * 24)
    first.append("msg_0", {"content": "mine"})
    second.append("msg_0", {"content": "theirs"})
    assert list(archive.sessions()) == ["ses_" + "1" * 24, "ses_" + "2" * 24]
    assert first.read(0).payload["content"] == "mine"
    assert second.count == 1


def test_unsafe_session_ids_are_refused(archive):
    """A session id becomes a directory name."""
    for unsafe in ["../escape", "a/b", "", ".", ".."]:
        with pytest.raises(ValueError, match="unsafe session id"):
            archive.create(unsafe)


def test_archive_is_plain_jsonl_on_disk(archive, log, tmp_path):
    """Inspectable without our tooling. That is why JSONL was chosen."""
    import json

    fill(log, 3)
    path = tmp_path / "archive" / "sessions" / SESSION / "records.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[1])["payload"]["content"] == "message 1"


def test_size_bytes_reports_raw_conversation_size(archive, log):
    assert archive.size_bytes(SESSION) == 0
    fill(log, 10)
    assert archive.size_bytes(SESSION) > 0
    assert archive.size_bytes("ses_missing") == 0
