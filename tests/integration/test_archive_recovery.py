"""Crash safety, incomplete writes, and corruption detection.

The archive must never treat a partial write as valid conversation history.
These tests simulate the failures rather than assuming filesystem behaviour.
"""

import pytest

from open_context.archive import (
    ENTRY_SIZE,
    Archive,
    ArchiveFormatError,
    CorruptRecordError,
)

pytestmark = pytest.mark.integration

SESSION = "ses_" + "b" * 24


@pytest.fixture
def archive(tmp_path):
    return Archive(tmp_path / "archive")


@pytest.fixture
def paths(tmp_path):
    base = tmp_path / "archive" / "sessions" / SESSION
    return base / "records.jsonl", base / "records.idx"


def seed(archive, count=5):
    log = archive.create(SESSION)
    for i in range(count):
        log.append(f"msg_{i}", {"content": f"m{i}"})
    return log


def test_torn_final_record_is_removed_and_reported(archive, paths):
    """A process that died mid-append leaves a line with no terminator."""
    seed(archive)
    data, _ = paths
    intact = data.read_bytes()
    with data.open("ab") as handle:
        handle.write(b'{"seq":5,"id":"msg_5","kind":"mes')

    log = archive.open(SESSION)
    assert log.count == 5, "the torn record is not counted as history"
    assert log.recovery.truncated_bytes == 33
    assert not log.recovery.clean, "the loss is reported, not swallowed"
    assert data.read_bytes() == intact, "the preceding record must survive byte for byte"
    assert log.verify().ok


def test_complete_but_unparseable_final_line_is_removed(archive, paths):
    """A finished write can still be garbage if the disk lost bytes mid-line."""
    seed(archive)
    data, _ = paths
    with data.open("ab") as handle:
        handle.write(b"{not valid json at all}\n")

    log = archive.open(SESSION)
    assert log.count == 5
    assert log.recovery.truncated_bytes > 0
    assert log.verify().ok


def test_crash_between_data_sync_and_index_write_is_recovered(archive, paths):
    """The data file is synced before the index entry is written, so this gap is
    the expected failure window and must be repairable."""
    seed(archive)
    _, index = paths
    log = archive.open(SESSION)
    log.append("msg_5", {"content": "m5"})
    index.write_bytes(index.read_bytes()[:-ENTRY_SIZE])

    reopened = archive.open(SESSION)
    assert reopened.count == 6, "the record survived because the data file is the source of truth"
    assert reopened.recovery.recovered_records == 1
    assert reopened.recovery.truncated_bytes == 0
    assert reopened.read(5).payload["content"] == "m5"


def test_a_lost_index_is_rebuilt_from_the_data_file(archive, paths):
    seed(archive, 20)
    _, index = paths
    index.write_bytes(b"")

    log = archive.open(SESSION)
    assert log.count == 20
    assert log.recovery.recovered_records == 20
    assert log.read(19).payload["content"] == "m19"


def test_a_deleted_index_is_rebuilt(archive, paths):
    seed(archive, 5)
    _, index = paths
    index.unlink()
    assert archive.open(SESSION).count == 5


def test_a_torn_index_entry_is_ignored(archive, paths):
    """A partial index entry is not trusted; the tail is rescanned instead."""
    seed(archive, 5)
    _, index = paths
    index.write_bytes(index.read_bytes() + b"\x00\x01\x02")
    log = archive.open(SESSION)
    assert log.count == 5
    assert log.read(4).payload["content"] == "m4"


def test_an_index_longer_than_the_data_file_is_discarded(archive, paths):
    """If the index describes records the data file does not contain, the data
    file wins. It is the source of truth and the index is derived."""
    seed(archive, 5)
    data, _ = paths
    lines = data.read_bytes().split(b"\n")
    data.write_bytes(b"\n".join(lines[:3]))

    log = archive.open(SESSION)
    assert log.count == 2
    assert log.verify().ok


def test_mid_file_corruption_is_detected_not_repaired(archive, paths):
    """Damage in the body is not a failed write, so it is never truncated away."""
    seed(archive, 6)
    data, _ = paths
    lines = data.read_bytes().split(b"\n")
    lines[2] = lines[2].replace(b'"m2"', b'"XX"')
    data.write_bytes(b"\n".join(lines))

    log = archive.open(SESSION)
    report = log.verify()
    assert not report.ok
    assert report.corrupt == (2,), "the damage is located precisely"
    assert report.records == 6, "the rest of the archive is still readable"

    with pytest.raises(CorruptRecordError) as info:
        log.read(2)
    assert info.value.seq == 2
    assert log.read(3).payload["content"] == "m3", "undamaged records still read"


def test_corruption_is_caught_when_streaming_too(archive, paths):
    seed(archive, 4)
    data, _ = paths
    lines = data.read_bytes().split(b"\n")
    lines[1] = lines[1].replace(b'"m1"', b'"ZZ"')
    data.write_bytes(b"\n".join(lines))

    with pytest.raises(CorruptRecordError):
        list(archive.open(SESSION))


def test_a_reordered_record_is_corruption(archive, paths):
    """seq is inside the hashed body, so a record cannot be silently moved."""
    seed(archive, 4)
    data, _ = paths
    lines = data.read_bytes().rstrip(b"\n").split(b"\n")
    lines[1], lines[2] = lines[2], lines[1]
    data.write_bytes(b"\n".join(lines) + b"\n")

    assert archive.open(SESSION).verify().corrupt == (1, 2)


def test_rebuild_index_is_explicit_and_idempotent(archive):
    log = seed(archive, 10)
    assert log.rebuild_index() == 10
    assert log.rebuild_index() == 10
    assert log.read(9).payload["content"] == "m9"


def test_an_unknown_format_is_refused(archive, tmp_path):
    seed(archive, 2)
    manifest = tmp_path / "archive" / "sessions" / SESSION / "manifest.json"
    manifest.write_text('{"format": "something-else", "version": 1, "session_id": "x"}')
    with pytest.raises(ArchiveFormatError):
        archive.open(SESSION)


def test_a_future_format_version_is_refused(archive, tmp_path):
    seed(archive, 2)
    manifest = tmp_path / "archive" / "sessions" / SESSION / "manifest.json"
    import json

    payload = json.loads(manifest.read_text())
    payload["version"] = 99
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ArchiveFormatError, match="v99"):
        archive.open(SESSION)


def test_repeated_reopen_is_stable(archive):
    seed(archive, 5)
    for _ in range(3):
        log = archive.open(SESSION)
        assert log.count == 5
        assert log.recovery.clean
        assert log.verify().ok
