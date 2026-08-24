"""Updating state from new events without rereading the session.

```
watermark -> new events only -> extract -> reconcile -> updated state
```

**The point is not speed, it is cost.** Re-extracting a long session on every
update means sending the whole conversation to a model again, which is the
expense the whole project exists to avoid. A run that reprocessed everything
would be paying compaction's bill to do compaction's job.

**A watermark is a claim about what has been read, and it has to be earned.**
It advances only when an extraction over that range actually succeeded. A
watermark moved on a failed extraction would silently skip a region of the
conversation forever, and nothing downstream could tell: the state would simply
be missing whatever those events said, with no record that they were never read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from open_context.archive import SessionLog
from open_context.extraction.extractor import StructuredExtractor
from open_context.extraction.result import ExtractionResult
from open_context.models.state import StateItem


@dataclass(frozen=True)
class Watermark:
    """How far into a session extraction has read.

    ``next_seq`` is the first position **not** yet extracted, so a fresh session
    starts at zero and an empty update is the honest ``next_seq == count`` case
    rather than an error.
    """

    session_id: str
    next_seq: int = 0

    def __post_init__(self) -> None:
        if self.next_seq < 0:
            raise ValueError(f"a watermark cannot be negative, got {self.next_seq}")

    def advanced_to(self, last_seq: int) -> Watermark:
        """The watermark after a successful extraction ending at ``last_seq``.

        **Never moves backward.** A caller re-extracting an earlier range is
        allowed — it is how a bad extraction is redone — but it must not rewind
        the record of what has been read, or the events after it would be
        extracted twice and reconciled against themselves.
        """
        return Watermark(
            session_id=self.session_id,
            next_seq=max(self.next_seq, last_seq + 1),
        )


@dataclass(frozen=True)
class IncrementalUpdate:
    """One incremental pass, and whether it read anything."""

    watermark: Watermark
    result: ExtractionResult | None
    """``None`` when there was nothing new. Not an empty result, which would be
    indistinguishable from a range that was read and yielded no state."""

    events_read: int = 0

    @property
    def had_new_events(self) -> bool:
        return self.result is not None


def pending(log: SessionLog, watermark: Watermark) -> int:
    """How many events have arrived since the watermark."""
    return max(0, log.count - watermark.next_seq)


def extract_incremental(
    extractor: StructuredExtractor,
    log: SessionLog,
    watermark: Watermark,
    known_state: Sequence[StateItem] = (),
) -> IncrementalUpdate:
    """Extract only what has arrived since the watermark.

    Returns the new watermark alongside the result. **The watermark advances
    only on success**: an extraction whose reply was malformed leaves it where
    it was, so the same range is read again next time rather than being lost.

    ``known_state`` is what has already been recorded, and passing it is what
    makes a reversal expressible. A model shown only the new messages has never
    seen the decision being overturned and cannot name it, so its supersession
    claim resolves to nothing — which is precisely what the first real
    incremental run produced before this argument existed.
    """
    if watermark.session_id != log.session_id:
        raise ValueError(f"watermark is for {watermark.session_id} but the log is {log.session_id}")

    outstanding = pending(log, watermark)
    if outstanding == 0:
        return IncrementalUpdate(watermark=watermark, result=None, events_read=0)

    result = extractor.extract_range(log, start=watermark.next_seq, known_state=known_state)
    if result.report.malformed_response:
        return IncrementalUpdate(watermark=watermark, result=result, events_read=outstanding)

    return IncrementalUpdate(
        watermark=watermark.advanced_to(result.last_seq),
        result=result,
        events_read=outstanding,
    )


__all__ = [
    "IncrementalUpdate",
    "Watermark",
    "extract_incremental",
    "pending",
]
