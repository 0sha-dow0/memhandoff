"""Updating state from new events without rereading the session.

The property that carries this module is that a watermark is *earned*: it moves
only when an extraction over that range actually succeeded. One moved on a
failure would skip a region of the conversation permanently, and nothing
downstream could tell.
"""

import json

import pytest

from open_context.archive import Archive
from open_context.extraction import StructuredExtractor
from open_context.importers import EventType, RawEvent, import_events
from open_context.llm.fakes import FakeProvider
from open_context.state import Watermark, extract_incremental, pending

pytestmark = pytest.mark.integration

SESSION = "ses_" + "a" * 24


def log_with(tmp_path, count, archive=None):
    archive = archive or Archive(tmp_path / "archive")
    log = archive.create(SESSION) if not archive.exists(SESSION) else archive.open(SESSION)
    events = [
        RawEvent(
            type=EventType.USER_MESSAGE,
            provider="t",
            source_id=f"m{i}",
            source_type="user",
            text=f"message {i} about the cache",
        )
        for i in range(log.count, log.count + count)
    ]
    import_events(log, iter(events), provider="t", source_key=f"k{log.count}")
    return log, archive


def reply(contents, source="m0"):
    return json.dumps(
        {"items": [{"type": "goal", "content": c, "source_ids": [source]} for c in contents]}
    )


# ----------------------------------------------------------------------
# The watermark


def test_a_fresh_watermark_starts_at_zero():
    assert Watermark(SESSION).next_seq == 0


def test_a_negative_watermark_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        Watermark(SESSION, next_seq=-1)


def test_a_watermark_never_moves_backward():
    """Re-extracting an earlier range is allowed; rewinding the record is not.

    A rewound watermark would make the events after it be extracted twice and
    reconciled against themselves.
    """
    mark = Watermark(SESSION, next_seq=10)
    assert mark.advanced_to(3).next_seq == 10


def test_advancing_points_at_the_first_unread_position():
    assert Watermark(SESSION).advanced_to(4).next_seq == 5


# ----------------------------------------------------------------------
# Reading only what is new


def test_nothing_new_calls_no_model(tmp_path):
    log, _ = log_with(tmp_path, 3)
    provider = FakeProvider(reply=reply(["a"]), context_window=100_000)
    extractor = StructuredExtractor(provider)

    update = extract_incremental(extractor, log, Watermark(SESSION, next_seq=log.count))

    assert not update.had_new_events
    assert update.result is None
    assert provider.requests == []


def test_an_empty_update_is_not_an_empty_result(tmp_path):
    """The two mean different things and must not look alike.

    ``None`` says nothing was read. An empty result would say a range was read
    and yielded no state.
    """
    log, _ = log_with(tmp_path, 2)
    extractor = StructuredExtractor(FakeProvider(reply=reply(["a"]), context_window=100_000))
    update = extract_incremental(extractor, log, Watermark(SESSION, next_seq=2))
    assert update.result is None


def test_only_the_new_events_reach_the_model(tmp_path):
    """The point of the phase: not resending the whole conversation."""
    log, archive = log_with(tmp_path, 3)
    log, _ = log_with(tmp_path, 2, archive=archive)

    provider = FakeProvider(reply=reply(["a"], source="m3"), context_window=100_000)
    extract_incremental(StructuredExtractor(provider), log, Watermark(SESSION, next_seq=3))

    sent = provider.requests[0].messages[-1].content
    assert "[m3]" in sent and "[m4]" in sent
    assert "[m0]" not in sent, "already-read events were resent"


def test_the_watermark_advances_past_what_was_read(tmp_path):
    log, _ = log_with(tmp_path, 4)
    extractor = StructuredExtractor(FakeProvider(reply=reply(["a"]), context_window=100_000))
    update = extract_incremental(extractor, log, Watermark(SESSION))
    assert update.watermark.next_seq == 4


def test_a_second_pass_reads_only_what_arrived_since(tmp_path):
    log, archive = log_with(tmp_path, 2)
    extractor = StructuredExtractor(FakeProvider(reply=reply(["a"]), context_window=100_000))
    first = extract_incremental(extractor, log, Watermark(SESSION))

    log, _ = log_with(tmp_path, 3, archive=archive)
    provider = FakeProvider(reply=reply(["b"], source="m2"), context_window=100_000)
    second = extract_incremental(StructuredExtractor(provider), log, first.watermark)

    assert second.events_read == 3
    assert second.watermark.next_seq == 5
    assert "[m0]" not in provider.requests[0].messages[-1].content


# ----------------------------------------------------------------------
# Earning the watermark


def test_a_malformed_extraction_does_not_advance_the_watermark(tmp_path):
    """Otherwise that range is skipped forever, with nothing to say so."""
    log, _ = log_with(tmp_path, 3)
    extractor = StructuredExtractor(FakeProvider(reply="not json at all", context_window=100_000))
    update = extract_incremental(extractor, log, Watermark(SESSION))

    assert update.watermark.next_seq == 0
    assert update.result is not None
    assert update.result.report.malformed_response


def test_the_same_range_is_retried_after_a_failure(tmp_path):
    log, _ = log_with(tmp_path, 3)
    failed = extract_incremental(
        StructuredExtractor(FakeProvider(reply="garbage", context_window=100_000)),
        log,
        Watermark(SESSION),
    )
    provider = FakeProvider(reply=reply(["a"]), context_window=100_000)
    retried = extract_incremental(StructuredExtractor(provider), log, failed.watermark)

    assert "[m0]" in provider.requests[0].messages[-1].content
    assert retried.watermark.next_seq == 3


def test_an_extraction_that_found_nothing_still_advances(tmp_path):
    """Reading a range and finding no state is success, not failure."""
    log, _ = log_with(tmp_path, 2)
    extractor = StructuredExtractor(
        FakeProvider(reply=json.dumps({"items": []}), context_window=100_000)
    )
    update = extract_incremental(extractor, log, Watermark(SESSION))
    assert update.watermark.next_seq == 2


def test_a_watermark_from_another_session_is_refused(tmp_path):
    log, _ = log_with(tmp_path, 1)
    extractor = StructuredExtractor(FakeProvider(reply=reply(["a"]), context_window=100_000))
    with pytest.raises(ValueError, match="watermark is for"):
        extract_incremental(extractor, log, Watermark("ses_" + "b" * 24))


def test_pending_counts_what_has_not_been_read(tmp_path):
    log, _ = log_with(tmp_path, 5)
    assert pending(log, Watermark(SESSION)) == 5
    assert pending(log, Watermark(SESSION, next_seq=2)) == 3
    assert pending(log, Watermark(SESSION, next_seq=99)) == 0


# ----------------------------------------------------------------------
# Old state + new events


def test_existing_state_is_shown_to_the_model(tmp_path):
    """Without it, incremental extraction cannot express a reversal.

    A model given only the new messages has never seen the decision being
    overturned, so it cannot name it, and the supersession claim it would need
    to make is unavailable. The first real incremental run produced exactly
    that: a correct new decision and no link to what it replaced.
    """
    from open_context.models.state import Decision

    log, _ = log_with(tmp_path, 2)
    provider = FakeProvider(reply=reply(["a"]), context_window=100_000)
    known = [Decision(session_id=SESSION, content="use PostgreSQL", rationale="matches stack")]

    extract_incremental(StructuredExtractor(provider), log, Watermark(SESSION), known_state=known)

    sent = provider.requests[0].messages[-1].content
    assert "use PostgreSQL" in sent
    assert "CURRENT STATE" in sent


def test_state_content_is_reproduced_verbatim(tmp_path):
    """Supersession is matched literally, so a paraphrase would resolve to nothing."""
    from open_context.models.state import Goal

    log, _ = log_with(tmp_path, 1)
    provider = FakeProvider(reply=reply(["a"]), context_window=100_000)
    exact = "Ship the CSV export before the audit log, per the agreed order."
    extract_incremental(
        StructuredExtractor(provider),
        log,
        Watermark(SESSION),
        known_state=[Goal(session_id=SESSION, content=exact)],
    )
    assert exact in provider.requests[0].messages[-1].content


def test_no_state_section_when_there_is_no_state(tmp_path):
    """A first pass should not carry a header describing an empty list."""
    log, _ = log_with(tmp_path, 1)
    provider = FakeProvider(reply=reply(["a"]), context_window=100_000)
    extract_incremental(StructuredExtractor(provider), log, Watermark(SESSION))
    assert "CURRENT STATE" not in provider.requests[0].messages[-1].content
