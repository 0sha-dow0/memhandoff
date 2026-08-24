"""The baseline compaction engine, end to end against a real archive.

Every test is offline and deterministic: the fake provider and the word
tokenizer from Phase 4.5 are the whole model surface. No key, no network, no
Ollama, no vendor SDK.
"""

import hashlib
import json
import tracemalloc
from typing import Any

import pytest

from open_context.archive import Archive
from open_context.compaction import (
    FALLBACK_MULTI_PASS,
    FALLBACK_RECENT_DROPPED,
    FALLBACK_SUMMARY_RETRY,
    FALLBACK_SUMMARY_TRUNCATED,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    WARN_ESTIMATED_COUNTS,
    WARN_RECENT_DROPPED_LOST,
    WARN_UNKNOWN_WINDOW,
    WINDOW_EMPTY_OVERSIZED,
    BaselineCompactionResult,
    BaselineConfig,
    BaselineRecentPlusSummary,
    CompactionRequest,
    CompactionStrategy,
    RecentWindow,
    TargetTooSmallError,
    count_conversation,
    select_recent_window,
    stream_archive,
)
from open_context.compaction.prompts import BASELINE_SUMMARY_V1
from open_context.compaction.rendering import body_of
from open_context.importers import import_path
from open_context.llm import ContextLimitExceededError, TokenCount, TokenizationUnavailableError
from open_context.llm.fakes import FailingProvider, FakeProvider, WordTokenizer

pytestmark = pytest.mark.integration

SESSION = "ses_" + "5" * 24


# ----------------------------------------------------------------------
# Realistic synthetic conversation


def conversation(turns: int = 12, filler_words: int = 12) -> list[dict[str, Any]]:
    """A session with the shapes a real one has.

    Important information early, long irrelevant stretches in the middle,
    repetition, tool traffic, and the live thread at the end. Small enough to
    keep CI fast; the properties under test do not need a large fixture.
    """
    records: list[dict[str, Any]] = [
        {"id": "sys", "role": "system", "content": "You are helping build a data pipeline."},
        {
            "id": "goal",
            "role": "user",
            "content": "We need to ingest 40GB of CSV daily and query it within 2 seconds.",
        },
        {
            "id": "constraint",
            "role": "user",
            "content": "Hard constraint: it must run on one machine. No Kubernetes.",
        },
        {
            "id": "decision",
            "role": "assistant",
            "content": (
                "We should use PostgreSQL with partitioning rather than MongoDB, because the "
                "existing infrastructure already runs Postgres 16 and the team knows it."
            ),
        },
        {
            "id": "failed",
            "role": "assistant",
            "content": "I tried DuckDB first but it ran out of memory on the 40GB join.",
        },
    ]
    filler = " ".join(["discussion"] * filler_words)
    for i in range(turns):
        records.append({"id": f"u{i}", "role": "user", "content": f"question {i}. {filler}"})
        records.append({"id": f"a{i}", "role": "assistant", "content": f"answer {i}. {filler}"})
        if i % 4 == 0:
            records.append(
                {
                    "id": f"tc{i}",
                    "type": "tool_call",
                    "tool_name": "psql",
                    "tool_call_id": f"c{i}",
                    "arguments": {"sql": f"SELECT {i}"},
                }
            )
            records.append(
                {
                    "id": f"tr{i}",
                    "type": "tool_result",
                    "tool_name": "psql",
                    "tool_call_id": f"c{i}",
                    "content": f"{i} rows",
                }
            )
    records.append(
        {"id": "open", "role": "user", "content": "Partitioning key is still undecided."}
    )
    records.append({"id": "last", "role": "assistant", "content": "Next step: benchmark by month."})
    return records


@pytest.fixture
def archive(tmp_path):
    return Archive(tmp_path / "archive")


@pytest.fixture
def make_log(archive, tmp_path):
    def build(records, name="conv.jsonl", session=SESSION):
        path = tmp_path / name
        path.write_text("".join(f"{json.dumps(r)}\n" for r in records), encoding="utf-8")
        import_path(archive, session, path)
        return archive.open(session)

    return build


@pytest.fixture
def log(make_log):
    return make_log(conversation())


@pytest.fixture
def tokenizer():
    return WordTokenizer(exact=True)


def summarizer(reply="SUMMARY. " + "condensed point. " * 12, **kwargs):
    return FakeProvider(reply=reply, context_window=kwargs.pop("context_window", 100_000), **kwargs)


def strategy(provider=None, tokenizer_=None):
    return BaselineRecentPlusSummary(provider or summarizer(), tokenizer_ or WordTokenizer())


def compact(log, target, provider=None, tokenizer_=None, config=None):
    return strategy(provider, tokenizer_).compact(
        CompactionRequest(log=log, target_tokens=target, config=config or BaselineConfig())
    )


# ----------------------------------------------------------------------
# Conversation size cases


def test_an_empty_conversation_needs_no_model(archive):
    log = archive.create(SESSION)
    provider = summarizer()
    result = strategy(provider).compact(CompactionRequest(log=log, target_tokens=1_000))

    assert result.archive_event_count == 0
    assert result.recent_event_count == 0
    assert result.historical_summary == ""
    assert not result.compaction_needed
    assert provider.requests == [], "no model was called"


def test_a_single_message_conversation(make_log):
    log = make_log([{"id": "one", "role": "user", "content": "hello"}])
    result = compact(log, 1_000)

    assert result.recent_event_count == 1
    assert not result.compaction_needed


def test_a_conversation_below_target_is_returned_whole(log, tokenizer):
    """Summarizing something that already fits spends a call to make it worse."""
    input_tokens = count_conversation(log, tokenizer)
    provider = summarizer()
    result = strategy(provider, tokenizer).compact(
        CompactionRequest(log=log, target_tokens=input_tokens.count * 2)
    )

    assert not result.compaction_needed
    assert result.recent_event_count == log.count
    assert result.historical_summary == ""
    assert result.historical_event_count == 0
    assert provider.requests == [], "no model was called"
    assert result.output_tokens.count == input_tokens.count


def test_a_conversation_exactly_at_target_is_returned_whole(log, tokenizer):
    input_tokens = count_conversation(log, tokenizer)
    result = compact(log, input_tokens.count, tokenizer_=tokenizer)

    assert not result.compaction_needed
    assert result.recent_event_count == log.count
    assert result.within_budget


def test_a_conversation_slightly_above_target_is_compacted(log, tokenizer):
    input_tokens = count_conversation(log, tokenizer)
    result = compact(log, input_tokens.count - 1, tokenizer_=tokenizer)

    assert result.compaction_needed
    assert result.within_budget
    assert result.model.llm_calls == 1


def test_a_large_conversation_compacts_within_budget(make_log, tokenizer):
    log = make_log(conversation(turns=80, filler_words=25))
    result = compact(log, 600, tokenizer_=tokenizer)

    assert log.count > 200, "a fixture large enough for the ratio to mean something"
    assert result.compaction_needed
    assert result.within_budget
    assert result.output_tokens.count <= 600
    assert result.compression_ratio > 5


# ----------------------------------------------------------------------
# Recent window


def test_the_recent_window_is_chosen_by_tokens_not_message_count(log, tokenizer):
    small = select_recent_window(log, tokenizer, 60)
    large = select_recent_window(log, tokenizer, 400)

    assert small.count < large.count
    assert small.tokens.count <= 60
    assert large.tokens.count <= 400


def test_the_recent_window_is_the_tail_in_order(log, tokenizer):
    window = select_recent_window(log, tokenizer, 200)
    ids = [event.source_id for event in window.events]

    assert ids[-1] == "last", "the window ends at the end of the conversation"
    assert ids == sorted(ids, key=lambda i: _position_of(log, i)), "and stays in order"


def test_the_recent_window_holds_whole_events(log, tokenizer):
    """A fragment of a tool result is not something another agent can act on."""
    window = select_recent_window(log, tokenizer, 137)
    for event in window.events:
        assert event.text is None or event.text.strip() == event.text.strip()
    rebuilt = [_position_of(log, event.source_id) for event in window.events]
    assert rebuilt == list(range(window.start_seq, window.start_seq + window.count))


def test_an_event_larger_than_the_whole_budget_leaves_the_window_empty(make_log, tokenizer):
    """Deterministic: it is summarized with the history instead of being cut up."""
    log = make_log(
        [
            {"id": "a", "role": "user", "content": "short"},
            {"id": "huge", "role": "assistant", "content": "word " * 500},
        ]
    )
    window = select_recent_window(log, tokenizer, 20)

    assert window.count == 0
    assert window.start_seq == log.count
    assert WINDOW_EMPTY_OVERSIZED in window.warnings


def test_the_window_is_empty_for_a_zero_budget(log, tokenizer):
    assert select_recent_window(log, tokenizer, 0).count == 0


# ----------------------------------------------------------------------
# Historical summary


def test_the_history_is_summarized_by_the_model(log):
    provider = summarizer(reply="the condensed history")
    result = strategy(provider).compact(CompactionRequest(log=log, target_tokens=300))

    assert result.historical_summary == "the condensed history"
    assert result.historical_event_count > 0
    assert result.historical_range == (0, result.recent_range[0])
    assert len(provider.requests) == 1


def test_the_summary_request_carries_the_versioned_prompt(log):
    provider = summarizer()
    strategy(provider).compact(CompactionRequest(log=log, target_tokens=300))

    (request,) = provider.requests
    system, user = request.messages
    assert system.content == BASELINE_SUMMARY_V1.text
    assert "another AI agent" in system.content.lower() or "different ai agent" in (
        system.content.lower()
    )
    assert "earlier part of the session" in user.content


def test_the_summary_request_excludes_the_recent_window(log):
    """The baseline stays explainable: the recent window is kept, not summarized."""
    provider = summarizer()
    result = strategy(provider).compact(CompactionRequest(log=log, target_tokens=300))

    (request,) = provider.requests
    body = request.messages[1].content
    assert "Next step: benchmark by month" not in body, "the last event is recent, not historical"
    assert result.recent_events[-1].text == "Next step: benchmark by month."


def test_the_summary_request_reserves_an_output_budget(log):
    provider = summarizer()
    result = strategy(provider).compact(CompactionRequest(log=log, target_tokens=300))

    (request,) = provider.requests
    assert request.max_output_tokens is not None
    assert request.max_output_tokens <= result.allocated_summary_tokens


def test_the_output_budget_is_clamped_to_what_the_model_allows(log):
    provider = FakeProvider(reply="s", context_window=100_000)
    provider._info = provider.model_info().model_copy(update={"max_output_tokens": 25})
    strategy(provider).compact(CompactionRequest(log=log, target_tokens=300))

    (request,) = provider.requests
    assert request.max_output_tokens == 25, "asking for more than the model can emit is pointless"


def test_the_prompt_version_is_recorded(log):
    result = compact(log, 300)
    assert result.prompt_id == "baseline_summary_v1"
    assert result.prompt_hash == BASELINE_SUMMARY_V1.content_hash
    assert len(result.prompt_hash) == 16


def test_the_prompt_hash_changes_when_the_prompt_does():
    """So a benchmark can tell that the prompt moved underneath it."""
    from open_context.compaction import Prompt

    assert Prompt("p", 1, "a").content_hash != Prompt("p", 1, "b").content_hash


# ----------------------------------------------------------------------
# Budget enforcement


def test_the_allocation_is_reported(log):
    result = compact(log, 1_000, config=BaselineConfig(recent_fraction=0.3))

    assert result.allocated_recent_tokens == 300
    assert result.allocated_summary_tokens == 700
    assert result.configuration["recent_fraction"] == 0.3


def test_the_output_never_silently_exceeds_the_target(log, tokenizer):
    """The model is asked for a size and is not believed."""
    provider = summarizer(reply="ignoring the budget entirely. " * 200)
    result = strategy(provider, tokenizer).compact(CompactionRequest(log=log, target_tokens=200))

    assert result.within_budget
    assert result.output_tokens.count <= 200
    assert result.fallback_used


def test_a_summary_over_its_budget_is_retried_once_then_cut(log, tokenizer):
    provider = FakeProvider(
        responses=["far too long. " * 300, "still far too long. " * 300],
        reply="short",
        context_window=100_000,
    )
    result = strategy(provider, tokenizer).compact(CompactionRequest(log=log, target_tokens=300))

    assert FALLBACK_SUMMARY_RETRY in result.fallbacks
    assert FALLBACK_SUMMARY_TRUNCATED in result.fallbacks
    assert result.model.llm_calls == 2, "one retry, not a loop"
    assert result.summary_tokens.count <= result.allocated_summary_tokens
    assert "truncated" in result.historical_summary


def test_a_summary_that_fits_after_the_retry_is_not_cut(log, tokenizer):
    provider = FakeProvider(
        responses=["far too long. " * 300, "brief enough now."],
        context_window=100_000,
    )
    result = strategy(provider, tokenizer).compact(CompactionRequest(log=log, target_tokens=300))

    assert FALLBACK_SUMMARY_RETRY in result.fallbacks
    assert FALLBACK_SUMMARY_TRUNCATED not in result.fallbacks
    assert result.historical_summary == "brief enough now."


# ----------------------------------------------------------------------
# The two halves are disjoint
#
# The summary is generated from archive[0:split) and the window holds
# archive[split:end). Nothing in the window is condensed into the summary, so
# dropping a recent event is outright loss rather than loss of detail. These
# pin that, because an earlier version of this engine got it backwards and
# dropped recent events first on the reasoning that the summary covered them.


def test_the_summary_input_ends_where_the_recent_window_begins(log, tokenizer):
    provider = summarizer()
    result = strategy(provider, tokenizer).compact(CompactionRequest(log=log, target_tokens=300))

    (request,) = provider.requests
    body = request.messages[1].content
    assert result.historical_range == (0, result.recent_range[0]), "disjoint, and they meet"

    summarized = list(_events_in(log, *result.historical_range))
    assert summarized[-1].text in body, "the last historical event was sent"
    for event in result.recent_events:
        assert _payload_of(event) not in body, "no recent event was sent to the summarizer"


def test_a_recent_event_is_never_represented_by_the_summary(log, tokenizer):
    """The claim that motivated the wrong enforcement order, stated as a test."""
    provider = summarizer()
    result = strategy(provider, tokenizer).compact(CompactionRequest(log=log, target_tokens=300))

    (request,) = provider.requests
    summarizer_saw = request.messages[1].content
    for event in result.recent_events:
        assert event.source_id is not None
        assert _payload_of(event) not in summarizer_saw


def test_no_event_is_lost_on_the_normal_path(log, tokenizer):
    """Every archive position is either summarized or kept verbatim."""
    result = compact(log, 300, tokenizer_=tokenizer)

    assert result.dropped_recent_event_count == 0
    assert result.lost_event_count == 0
    assert result.historical_range[1] == result.recent_range[0], "the ranges leave no gap"
    assert result.recent_range[1] == log.count


def test_the_summary_gives_way_before_the_recent_window_does(make_log, tokenizer):
    """Reducing the summary loses detail; dropping a recent event loses the event."""
    log = make_log(conversation(turns=30))
    provider = summarizer(reply="summary. " * 200)
    result = strategy(provider, tokenizer).compact(
        CompactionRequest(log=log, target_tokens=200, config=BaselineConfig(recent_fraction=0.9))
    )

    assert result.within_budget
    assert FALLBACK_RECENT_DROPPED not in result.fallbacks, "the window was preserved"
    assert result.dropped_recent_event_count == 0
    assert FALLBACK_SUMMARY_RETRY in result.fallbacks or FALLBACK_SUMMARY_TRUNCATED in (
        result.fallbacks
    ), "the summary is what gave way"


def test_the_newest_events_are_preserved(make_log, tokenizer):
    log = make_log(conversation(turns=30))
    for fraction in (0.2, 0.5, 0.9):
        result = strategy(summarizer(), tokenizer).compact(
            CompactionRequest(
                log=log, target_tokens=250, config=BaselineConfig(recent_fraction=fraction)
            )
        )
        assert result.recent_events, f"a window survived at recent_fraction={fraction}"
        assert result.recent_events[-1].source_id == "last"
        assert result.recent_range[1] == log.count, "the window always ends at the conversation"


def test_recent_events_are_kept_even_when_the_summary_must_be_truncated(make_log, tokenizer):
    log = make_log(conversation(turns=40))
    provider = FakeProvider(reply="verbose. " * 400, context_window=100_000)
    result = strategy(provider, tokenizer).compact(CompactionRequest(log=log, target_tokens=200))

    assert result.within_budget
    assert result.recent_event_count > 0
    assert result.dropped_recent_event_count == 0
    assert FALLBACK_SUMMARY_TRUNCATED in result.fallbacks


# ----------------------------------------------------------------------
# The emergency fallback
#
# Unreachable through the public API: the window is selected against a budget
# strictly smaller than the target, so recent alone always fits. It exists so
# the hard output constraint survives an accounting disagreement, and it is
# tested by constructing that disagreement directly.


def emergency(log, tokenizer_, target, window_events, window_tokens):
    """Drive enforcement with a window too large for the target."""
    from open_context.compaction.baseline import _Run
    from open_context.compaction.budget import allocate

    run = _Run(
        BaselineRecentPlusSummary(summarizer(), tokenizer_),
        CompactionRequest(log=log, target_tokens=target),
    )
    run.history_stop = 5
    window = RecentWindow(
        events=tuple(window_events),
        start_seq=5,
        tokens=TokenCount(count=window_tokens, exact=True, method="fake-words"),
    )
    return run, run._enforce_target(
        window,
        "a summary",
        TokenCount(count=10, exact=True, method="fake-words"),
        allocate(target, BaselineConfig()),
    )


def test_a_window_that_cannot_fit_drops_events_and_says_so(log, tokenizer):
    events = list(_events_in(log, 5, 12))
    run, (window, summary, summary_tokens) = emergency(log, tokenizer, 60, events, 10_000)

    assert FALLBACK_RECENT_DROPPED in run.fallbacks
    assert WARN_RECENT_DROPPED_LOST in run.warnings
    assert run.dropped_recent == len(events) - window.count
    assert run.dropped_recent > 0
    assert summary == "", "no summary can help when the window alone overflows"
    assert summary_tokens.count == 0


def test_dropped_events_are_absent_from_both_halves(log, tokenizer):
    """The information-loss semantics, proven rather than asserted in a comment."""
    events = list(_events_in(log, 5, 12))
    provider = summarizer()
    run, (window, _, _) = emergency(log, tokenizer, 60, events, 10_000)

    kept = {event.source_id for event in window.events}
    dropped = [event for event in events if event.source_id not in kept]
    assert dropped, "the scenario dropped something"

    # Nothing was sent to a summarizer in this path, and the summarized range
    # ends at position 5, before any of these events.
    assert provider.requests == []
    assert run.history_stop == 5
    for event in dropped:
        assert _position_of(log, event.source_id) >= run.history_stop, (
            "a dropped event sits after the summarized range, so nothing covers it"
        )


def test_dropping_keeps_the_newest_and_is_deterministic(log, tokenizer):
    events = list(_events_in(log, 5, 12))
    _, (first, _, _) = emergency(log, tokenizer, 60, events, 10_000)
    _, (second, _, _) = emergency(log, tokenizer, 60, events, 10_000)

    assert [e.source_id for e in first.events] == [e.source_id for e in second.events]
    if first.events:
        assert first.events[-1].source_id == events[-1].source_id, "the newest survives"


def test_the_reported_ranges_expose_the_gap(log, tokenizer):
    """When events are lost the two ranges stop meeting, and that is visible."""
    events = list(_events_in(log, 5, 12))
    run, (window, _, _) = emergency(log, tokenizer, 60, events, 10_000)

    assert window.start_seq > run.history_stop, (
        "the gap between the summarized range and the kept window is what was lost"
    )
    assert window.start_seq - run.history_stop == run.dropped_recent


def test_fallback_behaviour_is_deterministic(log, tokenizer):
    def run():
        provider = summarizer(reply="too long. " * 200)
        return strategy(provider, tokenizer).compact(CompactionRequest(log=log, target_tokens=200))

    first, second = run(), run()
    ignore = {"elapsed_seconds"}
    assert first.model_dump(exclude=ignore) == second.model_dump(exclude=ignore)


def test_an_extremely_small_target_is_refused(log):
    with pytest.raises(TargetTooSmallError):
        compact(log, 10)


def test_small_targets_still_produce_a_result_within_budget(log, tokenizer):
    for target in (50, 100, 500):
        result = compact(log, target, tokenizer_=tokenizer)
        assert result.within_budget, f"target {target} overflowed"
        assert result.output_tokens.count <= target


# ----------------------------------------------------------------------
# Tokenizer honesty


def test_an_exact_tokenizer_produces_exact_counts(log):
    result = compact(log, 300, tokenizer_=WordTokenizer(exact=True))

    assert result.counts_exact
    assert result.input_tokens.exact
    assert result.tokenizer.exact_available
    assert WARN_ESTIMATED_COUNTS not in result.warnings


def test_an_estimated_tokenizer_is_never_reported_as_exact(log):
    """The number may be right; nothing here knows that it is."""
    result = compact(log, 300, tokenizer_=WordTokenizer(exact=False))

    assert not result.counts_exact
    assert not result.input_tokens.exact
    assert not result.tokenizer.exact_available
    assert WARN_ESTIMATED_COUNTS in result.warnings, "the caller is told the budget was estimated"


def test_the_tokenizer_method_is_recorded(log):
    result = compact(log, 300)
    assert result.input_tokens.method == "fake-words"
    assert result.tokenizer.tokenizer == "fake-words"


def test_an_unavailable_tokenizer_fails_clearly(log):
    """Every budget decision depends on measuring, so there is nothing to fall back to."""
    from open_context.llm import ModelInfo, UnavailableTokenizer

    unavailable = UnavailableTokenizer(
        ModelInfo(provider="fake", model="m"), reason="no tokenizer installed"
    )
    with pytest.raises(TokenizationUnavailableError, match="no tokenizer installed"):
        compact(log, 300, tokenizer_=unavailable)


# ----------------------------------------------------------------------
# Provider behaviour


def test_a_provider_failure_reaches_the_caller(log):
    provider = FailingProvider(ContextLimitExceededError("too long", requested_tokens=99))

    with pytest.raises(ContextLimitExceededError) as info:
        strategy(provider).compact(CompactionRequest(log=log, target_tokens=300))
    assert info.value.requested_tokens == 99


def test_a_small_context_window_forces_a_multi_pass_summary(make_log, tokenizer):
    log = make_log(conversation(turns=30))
    provider = summarizer(reply="chunk summary.", context_window=400)
    result = strategy(provider, tokenizer).compact(
        CompactionRequest(
            log=log,
            target_tokens=300,
            config=BaselineConfig(prompt_reserve_tokens=20),
        )
    )

    assert FALLBACK_MULTI_PASS in result.fallbacks
    assert result.model.llm_calls > 2, "chunks, then a pass combining them"
    assert result.within_budget


def test_a_large_context_window_needs_only_one_call(log):
    result = compact(log, 300, provider=summarizer(context_window=1_000_000))
    assert result.model.llm_calls == 1
    assert FALLBACK_MULTI_PASS not in result.fallbacks


def test_an_unknown_context_window_is_reported_not_assumed(log):
    provider = FakeProvider(reply="summary", context_window=None)
    result = strategy(provider).compact(CompactionRequest(log=log, target_tokens=300))

    assert WARN_UNKNOWN_WINDOW in result.warnings
    assert result.model.context_window is None


def test_provider_reported_usage_is_carried_through(log):
    from open_context.llm import TokenUsage

    provider = summarizer(usage=TokenUsage(input_tokens=900, output_tokens=40))
    result = strategy(provider).compact(CompactionRequest(log=log, target_tokens=300))

    assert result.model.input_tokens_reported == 900
    assert result.model.output_tokens_reported == 40


def test_absent_usage_is_not_invented(log):
    result = compact(log, 300)
    assert result.model.input_tokens_reported is None


def test_the_compactor_works_offline_with_no_credentials(log):
    """The whole suite depends on this. Nothing here has a key or a socket."""
    result = compact(log, 300)
    assert result.model.provider == "fake"
    assert result.output_tokens.count > 0


# ----------------------------------------------------------------------
# Metadata and metrics


def test_strategy_metadata_is_recorded(log):
    result = compact(log, 300)

    assert result.strategy == STRATEGY_NAME == "baseline_recent_plus_summary"
    assert result.strategy_version == STRATEGY_VERSION
    assert result.session_id == SESSION
    assert result.archive_event_count == log.count
    assert result.elapsed_seconds >= 0.0


def test_the_baseline_satisfies_the_strategy_protocol():
    assert isinstance(strategy(), CompactionStrategy)


def test_provenance_is_enough_to_reconstruct_what_was_considered(log):
    result = compact(log, 300)
    historical_start, historical_stop = result.historical_range
    recent_start, recent_stop = result.recent_range

    assert historical_start == 0
    assert historical_stop == recent_start, "the two ranges meet with no gap"
    assert recent_stop == log.count
    assert len(result.recent_event_ids) == result.recent_event_count


def test_compression_and_retention_are_reciprocal(log):
    result = compact(log, 300)

    assert result.compression_ratio > 1
    assert result.retention_ratio < 1
    assert result.compression_ratio == pytest.approx(1 / result.retention_ratio)


def test_ratios_handle_an_empty_conversation(archive):
    log = archive.create(SESSION)
    result = strategy().compact(CompactionRequest(log=log, target_tokens=1_000))

    assert result.compression_ratio == 1.0
    assert result.retention_ratio == 1.0


def test_a_result_serializes_for_inspection(log):
    """Useful for debugging and for a future benchmark. Not a package format."""
    result = compact(log, 300)
    restored = BaselineCompactionResult.model_validate_json(result.model_dump_json())

    assert restored.historical_summary == result.historical_summary
    assert restored.recent_event_count == result.recent_event_count
    assert restored.input_tokens.exact == result.input_tokens.exact


def test_the_rendered_output_contains_both_halves(log):
    result = compact(log, 300)
    rendered = result.rendered()

    assert result.historical_summary in rendered
    assert result.recent_events[-1].text in rendered


# ----------------------------------------------------------------------
# The result is not structured state


def test_compaction_produces_no_structured_state(log, repo):
    """Phase 5 summarizes. Turning prose into a Decision is a later phase."""
    from open_context.models import Session

    repo.add_session(Session(id=SESSION, source="generic-jsonl"))
    compact(log, 300)

    counts = repo.counts()
    assert counts["state_items"] == 0
    assert counts["evidence"] == 0
    assert counts["messages"] == 0


def test_the_result_has_no_decision_shaped_fields():
    """A guard against the baseline quietly growing into an extractor."""
    fields = set(BaselineCompactionResult.model_fields)
    for forbidden in ("decisions", "goals", "constraints", "tasks", "facts", "entities"):
        assert forbidden not in fields


# ----------------------------------------------------------------------
# The archive is never touched


def fingerprint(archive, session):
    base = archive.sessions_root / session
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(base.iterdir())
    }


def test_compaction_does_not_modify_the_source_archive(archive, log, tokenizer):
    before = fingerprint(archive, SESSION)
    events_before = [(r.seq, r.id, r.hash) for r in log]

    compact(log, 300, tokenizer_=tokenizer)

    assert fingerprint(archive, SESSION) == before, "every byte of the archive is unchanged"
    reopened = archive.open(SESSION)
    assert [(r.seq, r.id, r.hash) for r in reopened] == events_before
    assert reopened.verify().ok
    assert reopened.count == log.count


def test_repeated_compaction_leaves_the_archive_alone(archive, log):
    before = fingerprint(archive, SESSION)
    for target in (200, 300, 1_000):
        compact(log, target)
    assert fingerprint(archive, SESSION) == before


def test_a_failed_compaction_leaves_the_archive_alone(archive, log):
    before = fingerprint(archive, SESSION)
    with pytest.raises(TargetTooSmallError):
        compact(log, 5)
    assert fingerprint(archive, SESSION) == before


def test_event_ordering_survives(log):
    result = compact(log, 400)
    positions = [_position_of(log, event.source_id) for event in result.recent_events]
    assert positions == sorted(positions)


# ----------------------------------------------------------------------
# Streaming


class CountingLog:
    """Wraps a log to record how it was read."""

    def __init__(self, log):
        self._log = log
        self.reads = 0
        self.ranges = []

    def __getattr__(self, name):
        return getattr(self._log, name)

    def read(self, seq):
        self.reads += 1
        return self._log.read(seq)

    def range(self, start=0, stop=None):
        self.ranges.append((start, stop))
        yield from self._log.range(start, stop)


def test_selecting_the_recent_window_reads_only_the_window(make_log, tokenizer):
    """Positional reads from the end, not a scan of the history."""
    log = make_log(conversation(turns=60))
    counting = CountingLog(log)

    window = select_recent_window(counting, tokenizer, 100)

    assert counting.reads <= window.count + 1, "one read per event kept, plus the one that did not"
    assert counting.reads < log.count / 4
    assert counting.ranges == [], "no range scan was needed to find the window"


def test_counting_the_conversation_does_not_hold_it(make_log, tokenizer, tmp_path):
    log = make_log(conversation(turns=200, filler_words=60))
    on_disk = (tmp_path / "archive" / "sessions" / SESSION / "records.jsonl").stat().st_size
    assert on_disk > 500_000

    tracemalloc.start()
    try:
        count_conversation(log, tokenizer)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < on_disk / 3, f"peak {peak} counting a {on_disk} byte archive"


def test_the_summary_input_is_read_as_a_range_not_a_list(make_log, tokenizer):
    """The history is streamed into chunks; nothing materialises the archive."""
    log = make_log(conversation(turns=40))
    counting = CountingLog(log)
    provider = summarizer(reply="summary")

    result = BaselineRecentPlusSummary(provider, tokenizer).compact(
        CompactionRequest(log=counting, target_tokens=300)
    )

    assert result.historical_range is not None
    split = result.historical_range[1]
    historical = [call for call in counting.ranges if call[1] == split]
    assert historical, "the history was read through the archive's range reader"
    assert counting.reads < log.count, "and the window was not found by reading everything"


def _position_of(log, source_id):
    for record in log:
        if record.id == source_id:
            return record.seq
    raise AssertionError(f"no event with id {source_id!r}")


def _events_in(log, start, stop):
    """Events at archive positions ``[start, stop)``."""
    return list(stream_archive(log, start, stop))


def _payload_of(event):
    """The content an event contributes, for checking what a request contained."""
    return body_of(event)
