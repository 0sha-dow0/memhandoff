"""Choosing the recent window.

Walks backward from the end of the archive, taking whole events while they fit
the recent budget. Each step is a positional read, which the archive answers in
constant time from its offset index, so selecting the window costs one read per
event kept and nothing per event skipped. The whole history is never loaded to
find out where the window starts.

**Selection is by token budget, not by message count.** "The last twenty
messages" is twenty tokens or twenty thousand depending on what those messages
were, which makes the output size unpredictable and the baseline unmeasurable.

**Whole events only.** A half-rendered tool result is not something another
agent can act on, and splitting one would make the window's contents depend on
tokenizer internals. The cost is that the window rarely fills its budget exactly.

**An event larger than the whole recent budget is left out, deterministically.**
It stays in the historical portion and is summarized there, so nothing is lost;
the recent window is simply empty, and a warning says so. The alternative —
truncating it — would put a fragment of a message in the place reserved for
verbatim material, which is the one thing the recent window exists to avoid.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from open_context.archive import SessionLog
from open_context.compaction.rendering import to_message
from open_context.importers import RawEvent
from open_context.llm import TokenCount, Tokenizer

WINDOW_EMPTY_OVERSIZED = (
    "the most recent event alone exceeds the recent budget; the recent window is "
    "empty and that event was summarized with the history instead"
)


@dataclass(frozen=True)
class RecentWindow:
    """The tail of the conversation that survives verbatim."""

    events: tuple[RawEvent, ...]
    start_seq: int
    """First archive position in the window. Equals the log's length when empty."""

    tokens: TokenCount
    warnings: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Source ids where the provider gave one, for future evaluation."""
        return tuple(event.source_id or "" for event in self.events)


def read_event(log: SessionLog, seq: int) -> RawEvent:
    """One event by position, without touching the rest of the archive."""
    return RawEvent.from_payload(log.read(seq).payload)


def select_recent_window(log: SessionLog, tokenizer: Tokenizer, budget_tokens: int) -> RecentWindow:
    """The largest suffix of whole events fitting in ``budget_tokens``."""
    total = log.count
    if total == 0 or budget_tokens <= 0:
        return RecentWindow(events=(), start_seq=total, tokens=TokenCount.total([]))

    chosen: list[RawEvent] = []
    costs: list[TokenCount] = []
    used = 0
    seq = total - 1
    warnings: list[str] = []

    while seq >= 0:
        event = read_event(log, seq)
        cost = tokenizer.count_messages([to_message(event)])
        if used + cost.count > budget_tokens:
            if not chosen:
                warnings.append(WINDOW_EMPTY_OVERSIZED)
            break
        used += cost.count
        chosen.append(event)
        costs.append(cost)
        seq -= 1

    chosen.reverse()
    costs.reverse()
    return RecentWindow(
        events=tuple(chosen),
        start_seq=seq + 1,
        tokens=TokenCount.total(costs),
        warnings=tuple(warnings),
    )


def count_conversation(log: SessionLog, tokenizer: Tokenizer) -> TokenCount:
    """Tokens in the whole archive, counted by streaming it.

    Folds one event at a time and holds none, so the count costs a pass over the
    file rather than a copy of it.

    Counted per event and summed. That matches how the tokenizers in this
    repository behave exactly; a tokenizer that applies a conversation-level
    template prefix would count a whole sequence slightly differently, which is
    recorded as an open question in docs/baseline-compaction.md.
    """
    return TokenCount.total(
        tokenizer.count_messages([to_message(event)]) for event in stream_archive(log)
    )


def stream_archive(log: SessionLog, start: int = 0, stop: int | None = None) -> Iterator[RawEvent]:
    """Events in order, one at a time.

    A generator over the archive's own range reader. Nothing here builds a list
    of the conversation, and nothing that consumes it should either.
    """
    for record in log.range(start, stop):
        yield RawEvent.from_payload(record.payload)


__all__ = [
    "WINDOW_EMPTY_OVERSIZED",
    "RecentWindow",
    "count_conversation",
    "read_event",
    "select_recent_window",
    "stream_archive",
]
