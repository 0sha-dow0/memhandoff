"""Extracting a session too large to send in one request.

**A real session does not fit.** The first live run against a real transcript
failed on exactly this: 19,178 tokens of conversation offered to a model with an
8,000-token-per-minute ceiling, refused with HTTP 413. One-shot extraction works
on the synthetic sessions a test suite builds and on nothing a person actually
has.

So the session is read in windows, each small enough to send, with the state
found so far carried into the next.

**Carrying state forward is what makes windows safe.** A window that saw only
its own messages has never seen the decision being overturned in a later one,
and would report a supersession that resolves to nothing — which is precisely
what Phase 7 measured before ``known_state`` existed. Each window is therefore
given what the ones before it concluded, and the cost of that is why windows are
sized generously rather than made as small as possible.

**Windows are cut on record boundaries, never inside one.** Half a tool result
is not a smaller tool result; it is a different one, and a model asked to read it
would extract a fact the conversation never contained.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from open_context.archive.log import SessionLog
from open_context.extraction.extractor import StructuredExtractor
from open_context.extraction.result import ExtractedItem
from open_context.llm.errors import LLMError, RateLimitError
from open_context.models.state import StateItem

DEFAULT_WINDOW_TOKENS = 2_500
"""Conversation tokens offered to the model per request.

Chosen against the tightest limit actually met — Groq's 8,000 tokens a minute —
with room for the prompt, the carried state, and the reply. Generous rather than
maximal: a window that just fits today fails on the session that carries one
long tool result.
"""

MAX_ATTEMPTS = 3
"""Tries per window before giving up on it.

Bounded because a rate limit that does not clear is a reason to stop, not to
keep asking. Three is enough to ride out the per-minute window that causes
almost all of these and short enough that a genuinely closed door is noticed.
"""

MAX_WAIT_SECONDS = 30.0
"""The longest this will sit still for one window.

A provider asking for eleven seconds is worth waiting out; one asking for ten
minutes is telling you to come back later, and a tool that silently obeyed would
look like it had hung.
"""

CHARACTERS_PER_TOKEN = 4
"""A crude local estimate, and deliberately crude.

Sizing a window is not a place to pay for a real tokenizer: the number only has
to be low enough to stay under a limit, and being wrong in the safe direction
costs one extra request.
"""


@dataclass
class WindowedResult:
    """State from the whole session, and what it cost to get."""

    items: list[StateItem] = field(default_factory=list)
    windows: int = 0
    records_read: int = 0
    llm_calls: int = 0
    rejected: int = 0
    warnings: list[str] = field(default_factory=list)
    failed_windows: int = 0
    provenance: dict[str, list[int]] = field(default_factory=dict)
    """State item id to the archive positions it was drawn from.

    Kept because extraction knows this and a package that discards it has to
    admit it cannot check its own claims. Phase 6 deliberately put archive ids on
    the ``ExtractedItem`` wrapper rather than on ``sources`` — the two id schemes
    have never been reconciled — so anything unwrapping the item loses the trail
    unless it carries this alongside.
    """

    waited_seconds: float = 0.0
    """Time spent honouring a provider's own retry-after.

    Reported so a slow run is explicable. A tool that sat quietly for two minutes
    would be indistinguishable from one that had hung.
    """

    @property
    def complete(self) -> bool:
        """Whether every window was read.

        A partial extraction is usable and must not be mistaken for a whole one:
        the state is real, and the absence of some of it is not visible in the
        state itself.
        """
        return self.failed_windows == 0


def plan_windows(
    log: SessionLog, *, window_tokens: int = DEFAULT_WINDOW_TOKENS
) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` ranges covering the session.

    Streams the log to measure it, so planning a large session costs one pass and
    no memory. A single record larger than a window gets a window of its own —
    splitting it would be worse than sending something too big, which at least
    fails loudly.
    """
    budget = max(window_tokens, 1) * CHARACTERS_PER_TOKEN
    windows: list[tuple[int, int]] = []

    start: int | None = None
    last = -1
    size = 0

    for record in log:
        length = len(str(record.payload))
        if start is None:
            start, size = record.seq, 0
        elif size and size + length > budget:
            windows.append((start, last + 1))
            start, size = record.seq, 0
        size += length
        last = record.seq

    if start is not None:
        windows.append((start, last + 1))
    return windows


def extract_windowed(
    extractor: StructuredExtractor,
    log: SessionLog,
    *,
    session_id: str | None = None,
    known_state: Sequence[StateItem] = (),
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    max_windows: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> WindowedResult:
    """Read a whole session in pieces small enough to send.

    **A failing window does not fail the session.** Rate limits are the common
    case here and they are transient; losing four windows of state because the
    fifth was throttled would make a long session strictly harder to extract than
    a short one. Failures are counted and named, and ``complete`` says whether
    any occurred.

    **Rate limits are waited out, because the provider says how long.** The LLM
    layer carries ``retry_after`` and deliberately never acts on it — retrying is
    a policy, and the boundary refuses to choose one on a caller's behalf. This
    is a caller, and windowed extraction is where the policy belongs: without it
    a first live run had **32 of 35 windows** refused while the provider was
    naming the exact number of seconds that would have fixed each one.

    ``sleep`` is injected so tests can prove the waiting happens without doing
    any.
    """
    result = WindowedResult()
    windows = plan_windows(log, window_tokens=window_tokens)
    if max_windows is not None:
        windows = windows[:max_windows]

    carried: list[StateItem] = list(known_state)
    seen: set[tuple[str, str]] = {
        (item.type.value, " ".join(item.content.lower().split())) for item in carried
    }

    for start, stop in windows:
        result.windows += 1
        extracted = None
        last_error: LLMError | None = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                extracted = extractor.extract_range(
                    log, start=start, stop=stop, session_id=session_id, known_state=carried
                )
                break
            except RateLimitError as exc:
                last_error = exc
                wait = exc.retry_after if exc.retry_after is not None else 5.0
                if attempt == MAX_ATTEMPTS - 1 or wait > MAX_WAIT_SECONDS:
                    break
                result.waited_seconds += wait
                sleep(wait)
            except LLMError as exc:
                # Not transient. Asking again would spend a request to learn the
                # same thing.
                last_error = exc
                break

        if extracted is None:
            result.failed_windows += 1
            kind = type(last_error).__name__ if last_error else "unknown"
            result.warnings.append(f"window {start}-{stop} failed ({kind}): {last_error}")
            continue

        result.llm_calls += extracted.report.llm_calls
        result.rejected += len(extracted.report.rejected)
        result.records_read += stop - start

        for entry in extracted.items:
            _add(entry, carried, seen, result.provenance)

    result.items = carried[len(known_state) :] if known_state else carried
    if result.failed_windows and result.failed_windows == len(windows):
        # Every window failed, so nothing was extracted at all. Saying "3 of 3
        # windows could not be read" buries that; a caller wants the headline.
        result.warnings.append(
            f"extraction failed: none of the {len(windows)} windows could be read"
        )
    elif result.failed_windows:
        result.warnings.append(
            f"{result.failed_windows} of {len(windows)} windows could not be read; "
            f"the state below is real but incomplete"
        )
    return result


def _add(
    entry: ExtractedItem,
    carried: list[StateItem],
    seen: set[tuple[str, str]],
    provenance: dict[str, list[int]],
) -> None:
    """Keep an item unless the same claim is already held.

    Windows overlap in meaning even when they do not overlap in records — a
    constraint restated three times is one constraint — and a receiving agent
    shown it three times learns nothing the second or third.
    """
    key = (entry.item.type.value, " ".join(entry.item.content.lower().split()))
    if key in seen:
        return
    seen.add(key)
    carried.append(entry.item)
    if entry.archive_seqs:
        provenance[entry.item.id] = list(entry.archive_seqs)


__all__ = [
    "CHARACTERS_PER_TOKEN",
    "DEFAULT_WINDOW_TOKENS",
    "MAX_ATTEMPTS",
    "MAX_WAIT_SECONDS",
    "WindowedResult",
    "extract_windowed",
    "plan_windows",
]
