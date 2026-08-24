"""Extracting a session too large to send in one request.

Written against a real failure: the first live extraction offered 19,178 tokens
of conversation to a model with an 8,000-token-per-minute ceiling and was refused
with HTTP 413. One-shot extraction works on the sessions a test suite builds and
on nothing a person actually has.
"""

import json

from open_context.archive import Archive
from open_context.extraction.extractor import StructuredExtractor
from open_context.extraction.windowed import (
    DEFAULT_WINDOW_TOKENS,
    MAX_ATTEMPTS,
    extract_windowed,
    plan_windows,
)
from open_context.importers.events import EventType, RawEvent
from open_context.llm.errors import InvalidRequestError, RateLimitError
from open_context.llm.fakes import FailingProvider, FakeProvider

SESSION = "ses_" + "ab" * 12


def archive_with(tmp_path, turns, *, size=400):
    archive = Archive(tmp_path / "archive")
    with archive.open_or_create(SESSION) as log:
        log.extend(
            [
                (
                    f"m{n:05d}",
                    RawEvent(
                        type=EventType.USER_MESSAGE,
                        provider="test",
                        source_id=f"m{n:05d}",
                        text=f"turn {n} " + "x" * size,
                    ).to_payload(),
                )
                for n in range(turns)
            ]
        )
    return archive


def reply(*indices):
    return json.dumps(
        {
            "items": [
                {
                    "type": "fact",
                    "content": f"turn {n} established a value",
                    "confidence": 0.9,
                    "source_ids": [f"m{n:05d}"],
                }
                for n in indices
            ]
        }
    )


def never_sleep(_seconds):
    return None


# ----------------------------------------------------------------------
# Planning


def test_a_long_session_is_cut_into_sendable_windows(tmp_path):
    archive = archive_with(tmp_path, 200)
    with archive.open(SESSION) as log:
        windows = plan_windows(log)
    assert len(windows) > 1
    assert windows[0][0] == 0
    assert windows[-1][1] == 200


def test_windows_cover_the_session_exactly_and_do_not_overlap(tmp_path):
    """A gap loses conversation silently; an overlap pays for it twice."""
    archive = archive_with(tmp_path, 150)
    with archive.open(SESSION) as log:
        windows = plan_windows(log)
    covered: list[int] = []
    for start, stop in windows:
        covered.extend(range(start, stop))
    assert covered == list(range(150))


def test_a_short_session_is_one_window(tmp_path):
    archive = archive_with(tmp_path, 3, size=10)
    with archive.open(SESSION) as log:
        assert plan_windows(log) == [(0, 3)]


def test_a_record_larger_than_a_window_gets_one_of_its_own(tmp_path):
    """Splitting it would be worse than sending something too big, which at
    least fails loudly. Half a tool result is a different tool result."""
    archive = archive_with(tmp_path, 1, size=DEFAULT_WINDOW_TOKENS * 8)
    with archive.open(SESSION) as log:
        assert plan_windows(log) == [(0, 1)]


def test_an_empty_session_has_no_windows(tmp_path):
    archive = Archive(tmp_path / "archive")
    with archive.open_or_create(SESSION) as log:
        assert plan_windows(log) == []


# ----------------------------------------------------------------------
# Extracting


def test_state_is_gathered_across_every_window(tmp_path):
    archive = archive_with(tmp_path, 60)
    with archive.open(SESSION) as log:
        windows = plan_windows(log)
        provider = FakeProvider(responses=[reply(w[0]) for w in windows])
        result = extract_windowed(
            StructuredExtractor(provider), log, session_id=SESSION, sleep=never_sleep
        )
    assert result.windows == len(windows)
    assert len(result.items) == len(windows)
    assert result.complete


def test_the_same_claim_from_two_windows_is_kept_once(tmp_path):
    """A constraint restated three times is one constraint."""
    archive = archive_with(tmp_path, 60)
    with archive.open(SESSION) as log:
        windows = plan_windows(log)
        provider = FakeProvider(responses=[reply(w[0]) for w in windows] * 2)
        first = extract_windowed(
            StructuredExtractor(provider), log, session_id=SESSION, sleep=never_sleep
        )
        again = extract_windowed(
            StructuredExtractor(FakeProvider(responses=[reply(w[0]) for w in windows])),
            log,
            session_id=SESSION,
            known_state=first.items,
            sleep=never_sleep,
        )
    assert again.items == [], "nothing new was learned the second time"


def test_provenance_survives_the_unwrapping(tmp_path):
    """Extraction knows which records an item came from, and a package that
    dropped that could not check its own claims. Phase 6 put archive ids on the
    wrapper rather than on `sources`, so anything unwrapping loses the trail
    unless it carries this alongside."""
    archive = archive_with(tmp_path, 20)
    with archive.open(SESSION) as log:
        result = extract_windowed(
            StructuredExtractor(FakeProvider(responses=[reply(0)])),
            log,
            session_id=SESSION,
            sleep=never_sleep,
        )
    assert result.items
    assert result.provenance[result.items[0].id] == [0]


# ----------------------------------------------------------------------
# Failure


def test_a_rate_limited_window_is_waited_out(tmp_path):
    """The provider says how long. The LLM layer carries `retry_after` and never
    acts on it — retrying is a policy — so the caller applies one. Without this,
    a first live run had 32 of 35 windows refused while being told the exact
    number of seconds that would have fixed each."""
    waits: list[float] = []
    archive = archive_with(tmp_path, 10)
    with archive.open(SESSION) as log:
        extract_windowed(
            StructuredExtractor(FailingProvider(RateLimitError("slow", retry_after=2.0))),
            log,
            session_id=SESSION,
            sleep=waits.append,
        )
    assert waits == [2.0] * (MAX_ATTEMPTS - 1)


def test_waiting_is_bounded(tmp_path):
    """A provider asking for ten minutes is telling you to come back later, and
    a tool that silently obeyed would look like it had hung."""
    waits: list[float] = []
    archive = archive_with(tmp_path, 10)
    with archive.open(SESSION) as log:
        result = extract_windowed(
            StructuredExtractor(FailingProvider(RateLimitError("later", retry_after=600.0))),
            log,
            session_id=SESSION,
            sleep=waits.append,
        )
    assert waits == []
    assert not result.complete


def test_an_error_that_is_not_transient_is_not_retried(tmp_path):
    """Asking again would spend a request to learn the same thing."""
    calls: list[float] = []
    archive = archive_with(tmp_path, 10)
    with archive.open(SESSION) as log:
        extract_windowed(
            StructuredExtractor(FailingProvider(InvalidRequestError("malformed"))),
            log,
            session_id=SESSION,
            sleep=calls.append,
        )
    assert calls == []


def test_one_failed_window_does_not_lose_the_others(tmp_path):
    """Losing four windows of state because the fifth was throttled would make a
    long session strictly harder to extract than a short one."""

    class Flaky:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def model_info(self):
            return self.inner.model_info()

        def generate(self, request):
            self.calls += 1
            if self.calls == 2:
                raise InvalidRequestError("this one window is bad")
            return self.inner.generate(request)

        def structured_output(self, request, schema):
            return self.inner.structured_output(request, schema)

    archive = archive_with(tmp_path, 60)
    with archive.open(SESSION) as log:
        windows = plan_windows(log)
        provider = Flaky(FakeProvider(responses=[reply(w[0]) for w in windows]))
        result = extract_windowed(
            StructuredExtractor(provider), log, session_id=SESSION, sleep=never_sleep
        )
    assert result.failed_windows == 1
    assert result.items, "the windows that worked still produced state"
    assert not result.complete


def test_a_partial_extraction_says_it_is_partial(tmp_path):
    """The state is real, and the absence of some of it is not visible in the
    state itself."""
    archive = archive_with(tmp_path, 60)
    with archive.open(SESSION) as log:
        result = extract_windowed(
            StructuredExtractor(FailingProvider(InvalidRequestError("no"))),
            log,
            session_id=SESSION,
            sleep=never_sleep,
        )
    assert not result.complete
    assert any("extraction failed" in w for w in result.warnings)
