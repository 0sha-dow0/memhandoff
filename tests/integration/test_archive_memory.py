"""Memory behaviour.

A 500k-token conversation must not need 500k tokens of RAM. These tests pin that
with tracemalloc rather than trusting that the code looks streaming.

**They measure growth, not absolute size.** Touching a file at all costs a read
buffer, sized by ``io.DEFAULT_BUFFER_SIZE`` — 8 KiB through Python 3.13 and
128 KiB from 3.14. That is a property of the interpreter, not of the archive.

An earlier version of these tests compared peak memory against a fraction of the
fixture size: `peak < on_disk / 20` over an 858 KB fixture, which allowed 43 KB.
That quietly encoded the assumption that a file buffer was small next to the
fixture. It was true at 8 KiB and false at 128 KiB, so four tests began failing
on Python 3.14 without anything about the archive having changed — the tests
were measuring the interpreter, not the code.

So each test now runs the same operation against a one-record archive and
against a much larger one, and asserts the difference is small. Every fixed cost
cancels by construction, including ones nobody has thought of, and what is left
is the archive's own contribution.

This is *tighter* than the ratio it replaces, not looser. Reading one record may
now cost a few kilobytes more than the same read against a single-record
archive; before, it was free to cost a twentieth of the entire file.
"""

import tracemalloc
from typing import NamedTuple

import pytest

from open_context.archive import Archive

pytestmark = pytest.mark.integration

TINY_SESSION = "ses_" + "c" * 24
LARGE_SESSION = "ses_" + "d" * 24

TINY_COUNT = 1
LARGE_COUNT = 400
PAYLOAD = "x" * 2_000

SCALING_ALLOWANCE = 64 * 1024
"""How much more the larger archive may cost than the one-record archive.

Sized against what it has to separate. Repeated measurement of these operations
varies by around 10 KB from allocator and collector timing, so the allowance
sits well clear of that. An operation that actually scaled would hold some
meaningful share of a 858 KB fixture, more than ten times this. There is no
plausible regression that lands in the gap.
"""


class Fixture(NamedTuple):
    archive: Archive
    session: str
    size: int
    """Bytes of the data file on disk."""


def build(root, session: str, count: int) -> Fixture:
    archive = Archive(root)
    log = archive.create(session)
    log.extend((f"msg_{i}", {"content": f"{i}:{PAYLOAD}"}) for i in range(count))
    return Fixture(archive, session, archive.size_bytes(session))


@pytest.fixture
def tiny(tmp_path):
    return build(tmp_path / "tiny", TINY_SESSION, TINY_COUNT)


@pytest.fixture
def large(tmp_path):
    return build(tmp_path / "large", LARGE_SESSION, LARGE_COUNT)


def peak_bytes(action):
    tracemalloc.start()
    try:
        action()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def assert_does_not_scale(on_tiny, on_large, *, doing: str):
    """The same operation must cost the same on a much larger archive."""
    small = peak_bytes(on_tiny)
    big = peak_bytes(on_large)
    assert big - small <= SCALING_ALLOWANCE, (
        f"{doing} cost {small} bytes on a {TINY_COUNT}-record archive and {big} on a "
        f"{LARGE_COUNT}-record one, a difference of {big - small}; the operation is "
        "holding history rather than streaming it"
    )


# ----------------------------------------------------------------------


def test_the_fixtures_differ_enough_for_the_comparison_to_mean_anything(tiny, large):
    """A guard against every test below silently passing over identical archives."""
    assert large.size > 700_000
    assert large.size > tiny.size * 100


def test_peak_memory_does_not_grow_with_archive_size(tiny, large):
    """The property that matters, and the one all the others are instances of."""
    assert_does_not_scale(
        lambda: [record.seq for record in tiny.archive.open(tiny.session)],
        lambda: [record.seq for record in large.archive.open(large.session)],
        doing="streaming an archive",
    )


def test_iteration_does_not_hold_the_archive_in_memory(tiny, large):
    """Reading every record's content must not accumulate it."""

    def consume(fixture):
        def run():
            return sum(
                len(record.payload["content"]) for record in fixture.archive.open(fixture.session)
            )

        return run

    assert_does_not_scale(consume(tiny), consume(large), doing="iterating and reading content")


def test_reading_one_record_costs_one_record(tiny, large):
    tiny_log = tiny.archive.open(tiny.session)
    large_log = large.archive.open(large.session)
    assert_does_not_scale(
        lambda: tiny_log.read(TINY_COUNT - 1),
        lambda: large_log.read(LARGE_COUNT - 1),
        doing="reading the last record",
    )


def test_range_reads_cost_the_range_not_the_archive(tiny, large):
    tiny_log = tiny.archive.open(tiny.session)
    large_log = large.archive.open(large.session)
    assert_does_not_scale(
        lambda: [r.seq for r in tiny_log.range(0, 1)],
        lambda: [r.seq for r in large_log.range(10, 11)],
        doing="reading a one-record range",
    )


def test_tail_reads_cost_the_tail_not_the_archive(tiny, large):
    tiny_log = tiny.archive.open(tiny.session)
    large_log = large.archive.open(large.session)
    assert_does_not_scale(
        lambda: [r.seq for r in tiny_log.tail(1)],
        lambda: [r.seq for r in large_log.tail(1)],
        doing="reading the last record via tail",
    )


def test_verification_streams(tiny, large):
    tiny_log = tiny.archive.open(tiny.session)
    large_log = large.archive.open(large.session)
    assert_does_not_scale(tiny_log.verify, large_log.verify, doing="verifying an archive")


def test_opening_a_clean_archive_does_not_scan_it(tiny, large):
    """Recovery is bounded to the tail. A clean archive reads nothing."""
    assert_does_not_scale(
        lambda: tiny.archive.open(tiny.session),
        lambda: large.archive.open(large.session),
        doing="opening a clean archive",
    )
    assert large.archive.open(large.session).recovery.clean


def test_appending_does_not_load_existing_history(tiny, large):
    tiny_log = tiny.archive.open(tiny.session)
    large_log = large.archive.open(large.session)
    assert_does_not_scale(
        lambda: tiny_log.append("msg_new", {"content": "small"}),
        lambda: large_log.append("msg_new", {"content": "small"}),
        doing="appending one record",
    )


def test_the_comparison_would_notice_an_operation_that_scaled(tiny, large):
    """The assertion is not vacuous.

    Reading a whole archive into a list is exactly the regression these tests
    exist to catch, so it must fail the same check that the streaming paths
    pass.
    """
    with pytest.raises(AssertionError, match="holding history rather than streaming"):
        assert_does_not_scale(
            lambda: list(tiny.archive.open(tiny.session)),
            lambda: list(large.archive.open(large.session)),
            doing="materialising every record",
        )
