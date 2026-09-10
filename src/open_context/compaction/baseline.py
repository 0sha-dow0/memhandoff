"""The baseline: summarize the old part, keep the recent part verbatim.

```
                  full conversation
                          |
              +-----------+-----------+
              |                       |
        older history            recent window
              |                       |
         LLM summary            kept verbatim
              |                       |
              +-----------+-----------+
                          |
                    measure, enforce
                          |
                  baseline context
```

**Phase 5 is a baseline intended for measurement. It is not the final MemHandoff
compaction algorithm.** It is roughly what a competent developer would build in
an afternoon, which is exactly the point: it is the thing later work has to beat,
and its failures are the evidence that later work is needed. Where it loses a
constraint stated once in message 40, that loss is a finding.

**Nothing here extracts structure.** A conversation containing "we chose
Postgres because the infra already runs it" produces a summary that says so, and
never a ``Decision`` record. Turning prose into typed state is a later phase, and
doing it here would make the baseline something other than a summarization
baseline.

**Every number is measured, never assumed.** The model is asked for a summary of
a given size and is not believed: the result is tokenized, compared to its
budget, and cut deterministically if it overshot. The same applies to the
combined output against the target.

**The archive is read, never written.** Compaction opens the log for reading
only, and the result is a separate object. This module contains no call that
appends, truncates, or rebuilds anything.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from open_context.compaction.budget import BaselineConfig, BudgetAllocation, allocate
from open_context.compaction.literals import (
    extract_literals,
    missing_from,
    render_ledger,
)
from open_context.compaction.prompts import (
    BASELINE_SUMMARY_V1,
    CHUNK_HEADER,
    COMBINE_HEADER,
    SINGLE_PASS_HEADER,
    Prompt,
)
from open_context.compaction.rendering import render_event, render_events
from open_context.compaction.result import (
    BaselineCompactionResult,
    ModelUsageInfo,
    TokenCountInfo,
    TokenizerInfo,
)
from open_context.compaction.strategy import CompactionRequest
from open_context.compaction.window import (
    RecentWindow,
    count_conversation,
    select_recent_window,
    stream_archive,
)
from open_context.importers import RawEvent
from open_context.llm import (
    ChatMessage,
    GenerationRequest,
    LLMProvider,
    TokenCount,
    Tokenizer,
)

STRATEGY_NAME = "baseline_recent_plus_summary"
STRATEGY_VERSION = 1

# Warnings and fallback names. Constants so tests and benchmarks can match on
# them without depending on prose.
WARN_ESTIMATED_COUNTS = (
    "token counts are estimated, not exact; the budget was enforced against an "
    "estimate and the real size may differ"
)
WARN_UNKNOWN_WINDOW = (
    "the model's context window is unknown, so the history was sent in one request "
    "without a size check"
)
WARN_NO_HISTORY = "there was no history to summarize; the whole conversation fits the recent window"
WARN_LITERALS_DROPPED = (
    "exact values from the summarized range were dropped: the summary already filled its "
    "budget, so there was no room to carry them"
)
WARN_RECENT_DROPPED_LOST = (
    "recent events were dropped to fit the target and are absent from the output "
    "entirely; the historical summary covers only what came before the recent "
    "window, so a dropped event is not represented anywhere"
)
FALLBACK_SUMMARY_RETRY = "summary_retried_smaller"
FALLBACK_SUMMARY_TRUNCATED = "summary_truncated"
FALLBACK_RECENT_DROPPED = "recent_events_dropped"
FALLBACK_MULTI_PASS = "multi_pass_summary"
FALLBACK_CHUNK_SUMMARIES_TRUNCATED = "chunk_summaries_truncated"


class BaselineRecentPlusSummary:
    """Historical summary plus verbatim recent context."""

    name = STRATEGY_NAME
    version = STRATEGY_VERSION

    def __init__(
        self,
        provider: LLMProvider,
        tokenizer: Tokenizer,
        *,
        prompt: Prompt = BASELINE_SUMMARY_V1,
    ) -> None:
        self.provider = provider
        self.tokenizer = tokenizer
        self.prompt = prompt

    # ------------------------------------------------------------------

    def compact(self, request: CompactionRequest) -> BaselineCompactionResult:
        started = time.perf_counter()
        run = _Run(self, request)
        result = run.execute()
        return result.model_copy(update={"elapsed_seconds": time.perf_counter() - started})


class _Run:
    """One compaction. Holds the mutable bookkeeping so the strategy stays reentrant."""

    def __init__(self, strategy: BaselineRecentPlusSummary, request: CompactionRequest) -> None:
        self.strategy = strategy
        self.request = request
        self.log = request.log
        self.config: BaselineConfig = request.config
        self.tokenizer: Tokenizer = strategy.tokenizer
        self.provider: LLMProvider = strategy.provider
        self.warnings: list[str] = []
        self.fallbacks: list[str] = []
        self.llm_calls = 0
        self.reported_input = 0
        self.reported_output = 0
        self.saw_usage = False
        self.history_stop = 0
        """Where the summarized range ends. Fixed once the window is chosen.

        Kept separately from the window's ``start_seq`` because the window can
        shrink afterwards and the summarized range cannot: the summary was built
        from ``archive[0:history_stop)`` and no later trimming changes that.
        """

        self.dropped_recent = 0

    def _note_fallback(self, name: str) -> None:
        """Record a fallback category once, however many times it was taken."""
        if name not in self.fallbacks:
            self.fallbacks.append(name)

    # ------------------------------------------------------------------

    def execute(self) -> BaselineCompactionResult:
        # Allocation first: a target too small to serve fails before any work.
        allocation = allocate(self.request.target_tokens, self.config)
        input_tokens = count_conversation(self.log, self.tokenizer)

        if input_tokens.count <= self.request.target_tokens:
            return self._passthrough(allocation, input_tokens)

        window = select_recent_window(self.log, self.tokenizer, allocation.recent_tokens)
        self.warnings.extend(window.warnings)

        # The summarized range is fixed here and never moves again. The window
        # may shrink later; the summary it was disjoint from does not grow to
        # cover what shrinking removed.
        self.history_stop = window.start_seq
        summary, summary_tokens = self._summarize(window.start_seq, allocation.summary_tokens)
        window, summary, summary_tokens = self._enforce_target(
            window, summary, summary_tokens, allocation
        )

        if not input_tokens.exact:
            self.warnings.append(WARN_ESTIMATED_COUNTS)

        return self._build(
            allocation=allocation,
            input_tokens=input_tokens,
            window=window,
            summary=summary,
            summary_tokens=summary_tokens,
            compaction_needed=True,
        )

    # ------------------------------------------------------------------
    # The conversation already fits

    def _passthrough(
        self, allocation: BudgetAllocation, input_tokens: TokenCount
    ) -> BaselineCompactionResult:
        """Return the conversation unchanged, without calling the model.

        Summarizing something that already fits would spend a model call to make
        the result worse. This is also the control case a benchmark needs: full
        original context, produced by the same code path.
        """
        events = tuple(stream_archive(self.log))
        window = RecentWindow(
            events=events,
            start_seq=0,
            tokens=input_tokens,
        )
        if not input_tokens.exact:
            self.warnings.append(WARN_ESTIMATED_COUNTS)
        return self._build(
            allocation=allocation,
            input_tokens=input_tokens,
            window=window,
            summary="",
            summary_tokens=TokenCount.total([]),
            compaction_needed=False,
        )

    # ------------------------------------------------------------------
    # Historical summary

    def _summarize(self, split: int, budget: int) -> tuple[str, TokenCount]:
        """Summarize archive positions ``[0, split)`` into at most ``budget`` tokens."""
        if split <= 0:
            self.warnings.append(WARN_NO_HISTORY)
            return "", TokenCount.total([])

        chunks = list(self._chunk_history(split, budget))
        if not chunks:
            self.warnings.append(WARN_NO_HISTORY)
            return "", TokenCount.total([])

        if len(chunks) == 1:
            text = self._request_summary(SINGLE_PASS_HEADER, chunks[0], budget)
        else:
            self._note_fallback(FALLBACK_MULTI_PASS)
            partials = [self._request_summary(CHUNK_HEADER, chunk, budget) for chunk in chunks]
            combined = "\n\n".join(partials)
            combined = self._fit_for_request(combined, budget)
            text = self._request_summary(COMBINE_HEADER, combined, budget)

        text = self._with_literals(text, split, budget)
        return self._fit_summary(text, budget)

    def _with_literals(self, summary: str, split: int, budget: int) -> str:
        """Append the exact values the summary let go of.

        Deterministic and after the fact: the model is never asked to keep these,
        because asking is what already failed. The ledger names only values the
        summary no longer states, so a summary that kept a number pays nothing
        for it, and it is capped so exact values cannot crowd out the prose that
        explains them.
        """
        if not self.config.preserve_literals or not self.config.literal_budget_tokens:
            return summary

        history = "\n".join(render_event(e) for e in stream_archive(self.log, 0, split))
        missing = missing_from(extract_literals(history), summary, self._recent_text(split))
        if not missing:
            return summary

        ledger = render_ledger(missing)
        allowed = min(
            self.config.literal_budget_tokens,
            max(0, budget - self.tokenizer.count_text(summary).count),
        )
        if allowed <= 0:
            self.warnings.append(WARN_LITERALS_DROPPED)
            return summary
        ledger = self._truncate(ledger, allowed)
        return f"{summary}\n\n{ledger}" if summary else ledger

    def _recent_text(self, split: int) -> str:
        """The verbatim tail, which needs no ledger entry for what it already says."""
        return "\n".join(render_event(e) for e in stream_archive(self.log, split, self.log.count))

    def _chunk_history(self, split: int, output_budget: int) -> Iterator[str]:
        """Rendered history in pieces that fit the model's context window.

        Streams the archive and holds one chunk. When the window is unknown the
        whole history becomes one chunk, which is the only honest option: a size
        check needs a size.
        """
        limit = self._request_limit(output_budget)
        buffer: list[str] = []
        used = 0

        for event in stream_archive(self.log, 0, split):
            rendered = render_event(event)
            cost = self.tokenizer.count_text(rendered).count
            if limit is not None and buffer and used + cost > limit:
                yield "\n".join(buffer)
                buffer, used = [], 0
            buffer.append(rendered)
            used += cost

        if buffer:
            yield "\n".join(buffer)

    def _request_limit(self, output_budget: int) -> int | None:
        """Tokens of material one request may carry, or None when unknowable."""
        window = self.provider.model_info().context_window
        if window is None:
            self.warnings.append(WARN_UNKNOWN_WINDOW)
            return None
        available = (
            int(window * self.config.context_fill_fraction)
            - self.config.prompt_reserve_tokens
            - output_budget
        )
        # A window too small to hold the overheads still gets one token of room,
        # so chunking degenerates to one event per request rather than looping.
        return max(1, available)

    def _request_summary(self, header: str, body: str, budget: int) -> str:
        """One model call. Everything provider-specific stays behind LLMProvider."""
        info = self.provider.model_info()
        requested = budget
        if info.max_output_tokens is not None:
            requested = min(requested, info.max_output_tokens)

        request = GenerationRequest.of(
            ChatMessage.system(self.strategy.prompt.text),
            ChatMessage.user(f"{header}\n\n{body}"),
            max_output_tokens=max(1, requested),
        )
        result = self.provider.generate(request)
        self.llm_calls += 1
        if result.usage is not None:
            self.saw_usage = True
            self.reported_input += result.usage.input_tokens or 0
            self.reported_output += result.usage.output_tokens or 0
        return result.text.strip()

    def _fit_for_request(self, text: str, output_budget: int) -> str:
        """Cut intermediate material down to what one request can carry."""
        limit = self._request_limit(output_budget)
        if limit is None or self.tokenizer.count_text(text).count <= limit:
            return text
        self._note_fallback(FALLBACK_CHUNK_SUMMARIES_TRUNCATED)
        return self._truncate(text, limit)

    def _fit_summary(self, text: str, budget: int) -> tuple[str, TokenCount]:
        """Hold the summary to its budget: retry once, then cut.

        The model is asked for a size and not believed. One retry, because a
        model that overshot once usually undershoots when asked for less, and an
        iterative loop would spend calls chasing a number a deterministic cut
        reaches immediately.
        """
        counted = self.tokenizer.count_text(text)
        if counted.count <= budget:
            return text, counted

        self._note_fallback(FALLBACK_SUMMARY_RETRY)
        smaller = max(1, int(budget * self.config.summary_retry_fraction))
        retried = self._request_summary(SINGLE_PASS_HEADER, text, smaller)
        counted = self.tokenizer.count_text(retried)
        if counted.count <= budget:
            return retried, counted

        self._note_fallback(FALLBACK_SUMMARY_TRUNCATED)
        cut = self._truncate(retried, budget)
        return cut, self.tokenizer.count_text(cut)

    # ------------------------------------------------------------------
    # Final budget

    def _enforce_target(
        self,
        window: RecentWindow,
        summary: str,
        summary_tokens: TokenCount,
        allocation: BudgetAllocation,
    ) -> tuple[RecentWindow, str, TokenCount]:
        """Hold summary plus recent window to the target.

        **The recent window is preserved; the summary gives way.**

        The two halves cover disjoint ranges. The summary is generated from
        ``archive[0:split)`` and the window holds ``archive[split:end)``, so an
        event dropped from the window is not condensed into the summary — it is
        not represented anywhere. Dropping one is outright information loss, not
        a loss of detail, which makes it the last thing to try rather than the
        first.

        So the summary is reduced instead: regenerated once at the smaller size,
        then cut deterministically if that was not enough. ``_fit_summary``
        guarantees its result fits the budget it is given, so passing it the room
        left by the recent window is enough to bring the total inside the target.

        Normally none of this runs. Allocation already guarantees
        ``recent_budget + summary_budget <= target`` and both halves are held to
        their budgets, so an overflow here means the accounting disagreed with
        itself somewhere.
        """
        target = allocation.target_tokens
        recent_tokens = window.tokens
        if summary_tokens.count + recent_tokens.count <= target:
            return window, summary, summary_tokens

        room = target - recent_tokens.count
        if room > 0:
            summary, summary_tokens = self._fit_summary(summary, room)
            if summary_tokens.count + recent_tokens.count <= target:
                return window, summary, summary_tokens

        # Emergency only: the recent window alone does not fit the target, so no
        # summary of any size can help and the window itself has to give. Not
        # reachable through the normal path, because the window was selected
        # against a budget strictly smaller than the target; it exists so the
        # hard output constraint holds even if that stops being true.
        self._note_fallback(FALLBACK_SUMMARY_TRUNCATED)
        summary = ""
        summary_tokens = self.tokenizer.count_text(summary)
        return self._drop_recent(window, target), summary, summary_tokens

    def _drop_recent(self, window: RecentWindow, target: int) -> RecentWindow:
        """Drop recent events, oldest first, until the window fits.

        Every dropped event is gone from the output: the summary covers only
        what came before the window, so nothing else carries it. The newest
        events survive longest, on the assumption that a continuing agent needs
        the live thread more than the turn before it — an assumption, not a fact
        the summary makes safe.
        """
        events = list(window.events)
        costs = [self.tokenizer.count_messages([_message(event)]) for event in events]

        while events and sum(cost.count for cost in costs) > target:
            events.pop(0)
            costs.pop(0)

        dropped = len(window.events) - len(events)
        if dropped:
            self._note_fallback(FALLBACK_RECENT_DROPPED)
            self.warnings.append(WARN_RECENT_DROPPED_LOST)
            self.dropped_recent += dropped

        return RecentWindow(
            events=tuple(events),
            start_seq=window.start_seq + dropped,
            tokens=TokenCount.total(costs),
            warnings=(),
        )

    def _truncate(self, text: str, budget: int) -> str:
        """Cut text to a token budget at a word boundary, deterministically.

        Binary search over word counts, which terminates in at most log2(words)
        tokenizer calls. Word boundaries rather than characters so the result
        does not end mid-token, and a marker so a reader can see it was cut. The
        marker is inside the budget, not added to it.
        """
        if budget <= 0:
            return ""
        if self.tokenizer.count_text(text).count <= budget:
            return text

        marker = self.config.truncation_marker
        if self.tokenizer.count_text(marker).count >= budget:
            return ""

        words = text.split()
        low, high, best = 0, len(words), ""
        while low <= high:
            middle = (low + high) // 2
            candidate = " ".join(words[:middle]) + marker
            if self.tokenizer.count_text(candidate).count <= budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    # ------------------------------------------------------------------

    def _build(
        self,
        *,
        allocation: BudgetAllocation,
        input_tokens: TokenCount,
        window: RecentWindow,
        summary: str,
        summary_tokens: TokenCount,
        compaction_needed: bool,
    ) -> BaselineCompactionResult:
        model_info = self.provider.model_info()
        tokenizer_info = self.tokenizer.model_info()
        output_tokens = summary_tokens + window.tokens
        # The summarized range, not the window's current start. Those differ
        # exactly when recent events were dropped, and the gap between the two
        # ranges is what was lost.
        split = self.history_stop

        return BaselineCompactionResult(
            session_id=self.request.session_id,
            strategy=self.strategy.name,
            strategy_version=self.strategy.version,
            prompt_id=self.strategy.prompt.identifier,
            prompt_hash=self.strategy.prompt.content_hash,
            configuration={
                "recent_fraction": self.config.recent_fraction,
                "minimum_target_tokens": self.config.minimum_target_tokens,
                "prompt_reserve_tokens": self.config.prompt_reserve_tokens,
                "context_fill_fraction": self.config.context_fill_fraction,
                "summary_retry_fraction": self.config.summary_retry_fraction,
            },
            historical_summary=summary,
            recent_events=window.events,
            target_tokens=allocation.target_tokens,
            allocated_recent_tokens=allocation.recent_tokens,
            allocated_summary_tokens=allocation.summary_tokens,
            input_tokens=_count_info(input_tokens),
            output_tokens=_count_info(output_tokens),
            recent_tokens=_count_info(window.tokens),
            summary_tokens=_count_info(summary_tokens),
            archive_event_count=self.log.count,
            historical_event_count=split if compaction_needed else 0,
            recent_event_count=window.count,
            historical_range=(0, split) if compaction_needed and split > 0 else None,
            recent_range=(
                (window.start_seq, window.start_seq + window.count) if window.count else None
            ),
            dropped_recent_event_count=self.dropped_recent,
            recent_event_ids=window.event_ids,
            compaction_needed=compaction_needed,
            fallbacks=tuple(self.fallbacks),
            warnings=tuple(self.warnings),
            tokenizer=TokenizerInfo(
                provider=tokenizer_info.provider,
                model=tokenizer_info.model,
                tokenizer=tokenizer_info.tokenizer,
                exact_available=tokenizer_info.counts_exactly,
            ),
            model=ModelUsageInfo(
                provider=model_info.provider,
                model=model_info.model,
                context_window=model_info.context_window,
                llm_calls=self.llm_calls,
                input_tokens_reported=self.reported_input if self.saw_usage else None,
                output_tokens_reported=self.reported_output if self.saw_usage else None,
            ),
        )


def _count_info(count: TokenCount) -> TokenCountInfo:
    return TokenCountInfo(count=count.count, exact=count.exact, method=count.method)


def _message(event: RawEvent) -> ChatMessage:
    from open_context.compaction.rendering import to_message

    return to_message(event)


__all__ = [
    "FALLBACK_CHUNK_SUMMARIES_TRUNCATED",
    "FALLBACK_MULTI_PASS",
    "FALLBACK_RECENT_DROPPED",
    "FALLBACK_SUMMARY_RETRY",
    "FALLBACK_SUMMARY_TRUNCATED",
    "STRATEGY_NAME",
    "STRATEGY_VERSION",
    "WARN_ESTIMATED_COUNTS",
    "WARN_NO_HISTORY",
    "WARN_UNKNOWN_WINDOW",
    "BaselineRecentPlusSummary",
    "render_events",
]
