"""Import memory behaviour.

A 500k-token conversation must import without needing 500k tokens of RAM. These
tests pin that with tracemalloc rather than trusting that the code looks like it
streams.

The thresholds are ratios against the source size, not absolute numbers, because
the fixed cost of an import is dominated by file buffers whose size is a property
of the interpreter rather than of this code: an import holds the source reader,
the archive data file, and the index open at once, and ``io.DEFAULT_BUFFER_SIZE``
went from 8 KiB to 128 KiB in Python 3.14. The fixture is sized so that floor
stays small next to the source in either case. The property being asserted is
that peak memory does not track the size of the conversation.
"""

import json
import tracemalloc

import pytest

from open_context.archive import Archive
from open_context.importers import import_path, stream_events

pytestmark = pytest.mark.integration

SESSION = "ses_" + "9" * 24
RECORD_COUNT = 2_000
PAYLOAD = "x" * 2_000


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "large.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(RECORD_COUNT):
            handle.write(
                json.dumps(
                    {
                        "id": f"m{i}",
                        "role": "user" if i % 2 else "assistant",
                        "content": f"{i}:{PAYLOAD}",
                        "timestamp": "2024-03-01T10:00:00Z",
                    }
                )
                + "\n"
            )
    return path


def peak_bytes(action):
    tracemalloc.start()
    try:
        action()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_importing_does_not_load_the_source_into_memory(tmp_path, source):
    on_disk = source.stat().st_size
    assert on_disk > 4_000_000, "fixture must dwarf both one record and the buffer floor"

    archive = Archive(tmp_path / "archive")
    peak = peak_bytes(lambda: import_path(archive, SESSION, source))

    assert archive.open(SESSION).count == RECORD_COUNT
    assert peak < on_disk / 4, (
        f"peak {peak} bytes importing a {on_disk} byte source; the import is buffering history"
    )


def test_peak_memory_does_not_grow_with_source_size(tmp_path):
    """The property that matters: an eightfold larger source must not cost eightfold."""

    def build(name, count):
        path = tmp_path / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for i in range(count):
                handle.write(json.dumps({"id": f"m{i}", "role": "user", "content": PAYLOAD}) + "\n")
        return path

    def import_into(name, marker, count):
        archive = Archive(tmp_path / name)
        session = "ses_" + marker * 24
        return peak_bytes(lambda: import_path(archive, session, build(name, count)))

    small = import_into("small", "1", 100)
    large = import_into("large", "2", 800)
    assert large < small * 3, (
        f"peak went from {small} to {large} bytes for an eightfold larger source"
    )


def test_reading_events_back_streams(tmp_path, source):
    on_disk = source.stat().st_size
    archive = Archive(tmp_path / "archive")
    import_path(archive, SESSION, source)
    log = archive.open(SESSION)

    def consume():
        return sum(len(event.text or "") for event in stream_events(log))

    peak = peak_bytes(consume)
    assert peak < on_disk / 4, "reading events back must not materialise the conversation"


def test_the_report_does_not_grow_with_the_number_of_problems(tmp_path):
    """Samples are capped, so a broken file cannot produce a report as big as itself."""
    path = tmp_path / "broken.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(500):
            handle.write(f"{{not json {i}\n")

    archive = Archive(tmp_path / "archive")
    report = import_path(archive, SESSION, path, on_malformed="skip")

    assert report.malformed_records == 500, "the count is exact"
    assert len(report.malformed_sample) == 20, "the sample is capped"
    assert report.truncated_samples, "and says so, rather than reading as the whole story"
