"""Memory behaviour.

A 500k-token conversation must not need 500k tokens of RAM. These tests pin
that with tracemalloc rather than trusting that the code looks streaming.

Fixtures are kept small enough to keep CI fast; the property being measured is
that peak memory stays flat as the archive grows, which does not need a large
archive to demonstrate.
"""

import tracemalloc

import pytest

from open_context.archive import Archive

pytestmark = pytest.mark.integration

SESSION = "ses_" + "c" * 24
RECORD_COUNT = 400
PAYLOAD = "x" * 2_000


@pytest.fixture
def archive(tmp_path):
    store = Archive(tmp_path / "archive")
    log = store.create(SESSION)
    log.extend((f"msg_{i}", {"content": f"{i}:{PAYLOAD}"}) for i in range(RECORD_COUNT))
    return store


def peak_bytes(action):
    tracemalloc.start()
    try:
        action()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_iteration_does_not_hold_the_archive_in_memory(archive, tmp_path):
    """Streaming the whole archive must cost far less than its size on disk."""
    on_disk = archive.size_bytes(SESSION)
    assert on_disk > 700_000, "fixture should be substantially larger than one record"

    def consume():
        total = 0
        for record in archive.open(SESSION):
            total += len(record.payload["content"])
        return total

    peak = peak_bytes(consume)
    assert peak < on_disk / 4, (
        f"peak {peak} bytes while streaming a {on_disk} byte archive; "
        "iteration is materialising the history"
    )


def test_peak_memory_does_not_grow_with_archive_size(tmp_path):
    """The property that matters: doubling the archive must not double the cost."""
    store = Archive(tmp_path / "scaling")

    def build_and_stream(session_id, count):
        log = store.create(session_id)
        log.extend((f"msg_{i}", {"content": f"{i}:{PAYLOAD}"}) for i in range(count))
        return peak_bytes(lambda: [record.seq for record in store.open(session_id)])

    small = build_and_stream("ses_" + "1" * 24, 100)
    large = build_and_stream("ses_" + "2" * 24, 800)
    assert large < small * 3, (
        f"peak went from {small} to {large} bytes for an eightfold larger archive"
    )


def test_reading_one_record_costs_one_record(archive):
    on_disk = archive.size_bytes(SESSION)
    log = archive.open(SESSION)
    peak = peak_bytes(lambda: log.read(RECORD_COUNT - 1))
    assert peak < on_disk / 20, "a single read should not touch the rest of the file"


def test_range_reads_cost_the_range_not_the_archive(archive):
    on_disk = archive.size_bytes(SESSION)
    log = archive.open(SESSION)
    peak = peak_bytes(lambda: [r.seq for r in log.range(10, 20)])
    assert peak < on_disk / 10


def test_verification_streams(archive):
    on_disk = archive.size_bytes(SESSION)
    log = archive.open(SESSION)
    peak = peak_bytes(log.verify)
    assert peak < on_disk / 4, "verify must not read the archive into memory"


def test_opening_a_clean_archive_does_not_scan_it(archive):
    """Recovery is bounded to the tail. A clean archive reads nothing."""
    on_disk = archive.size_bytes(SESSION)
    peak = peak_bytes(lambda: archive.open(SESSION))
    assert peak < on_disk / 20
    assert archive.open(SESSION).recovery.clean


def test_appending_does_not_load_existing_history(archive):
    on_disk = archive.size_bytes(SESSION)
    log = archive.open(SESSION)
    peak = peak_bytes(lambda: log.append("msg_new", {"content": "small"}))
    assert peak < on_disk / 20
