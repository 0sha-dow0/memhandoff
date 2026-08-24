"""Claude Code sessions.

**Written against the real format, not a guess at it.** Every claim below was
measured on 6 real transcripts totalling 8,890 conversation records from
``~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl``, because the phase this
belongs to says in terms: do not assume integration APIs, inspect the actual
mechanism. Three things that inspection found would each have been got wrong by
assumption.

**A session file is a DAG, not a list.** Records carry ``uuid`` and
``parentUuid``, and in **5 of the 6** files measured some parent had more than
one child — up to three. Those forks are rewinds, edits, and retries: the
abandoned attempts stay in the file. Reading it as a list therefore yields a
conversation containing work that was taken back, presented as though it
happened. Resolving that needs the whole file and so is not done here; see
``active_thread``.

**A message's content is a list of typed blocks**, not a string — ``text``,
``thinking``, ``tool_use``, ``tool_result`` — except when it is a string, which
it was for 225 of 3,323 user records. Both shapes are real and both are handled.

**Thinking is not the answer.** ``thinking`` blocks are the model's deliberation,
and this project has already been burned once by treating deliberation as a
conclusion — `strip_inline_reasoning` exists because a compactor stored a
truncated ``<think>`` block as a summary. They are skipped by default and
counted, so the count of what was skipped is available rather than the content
being silently present or silently gone.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TextIO

from open_context.importers.events import EventType, RawEvent
from open_context.importers.generic import parse_timestamp
from open_context.importers.report import BLANK_LINE, TEXT_SAMPLE_LENGTH, MalformedRecord

PROVIDER = "claude-code"

CONVERSATION_PARENT = "conversation_parent"
"""Metadata key holding the nearest conversation ancestor of an event.

Distinct from ``RawEvent.parent_id``, which stays exactly what the provider
wrote. The two differ whenever an attachment or system record sits between two
turns, which on real transcripts is most of the time.
"""

CONVERSATION_TYPES = frozenset({"user", "assistant"})
"""Record types that carry conversation.

Everything else in a session file is session bookkeeping — titles, permission
modes, file-history snapshots, queue operations — and is skipped rather than
imported. It describes the tool, not the work.
"""

KNOWN_TYPES = frozenset(
    {
        "user",
        "assistant",
        "system",
        "attachment",
        "summary",
        "file-history-snapshot",
        "file-history-delta",
        "mode",
        "permission-mode",
        "ai-title",
        "custom-title",
        "agent-color",
        "agent-name",
        "last-prompt",
        "queue-operation",
        "bridge-session",
    }
)
"""Types seen in real transcripts. Used only to recognise the format.

Not a whitelist: an unrecognised type is skipped as bookkeeping, not rejected.
The format gains record types between releases, and an importer that failed on
one it had not seen would break on an upgrade that changed nothing it reads.
"""

_BLOCK_EVENTS = {
    "tool_use": EventType.TOOL_CALL,
    "tool_result": EventType.TOOL_RESULT,
}

_ROLE_EVENTS = {
    "user": EventType.USER_MESSAGE,
    "assistant": EventType.ASSISTANT_MESSAGE,
}


class ClaudeCodeImporter:
    """Reads a Claude Code session transcript.

    ``include_thinking`` is off by default. Turning it on carries the model's
    deliberation into the archive, which is occasionally what someone wants and
    is never what a summary should be built from.
    """

    def __init__(self, *, include_thinking: bool = False, keep_raw: bool = True) -> None:
        self.include_thinking = include_thinking
        self.keep_raw = keep_raw
        """Whether each event carries the provider record it came from.

        On by default, because ``RawEvent.raw`` is what the archive stores and
        dropping it would quietly change what gets written. Off for a caller
        that only needs the normalised view — which is most of the cost of
        reading a large session, since every event otherwise pins a full parsed
        record and one transcript measured 50 MB of them.
        """

    @property
    def provider(self) -> str:
        return PROVIDER

    def detect(self, sample: str) -> bool:
        """Recognise a session file from whichever records the sample contains whole.

        **Not from the first record alone.** Real files open with whatever the
        tool wrote first, and one of the six measured begins with a
        ``custom-title`` record carrying a ``sessionId`` and no ``uuid`` at all.
        A check anchored on the first line rejected that file outright.

        The signature is a record whose ``type`` this format uses *and* which
        carries ``sessionId``: generic JSON Lines that happen to have a ``type``
        must not be claimed here.
        """
        for line in sample.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # The sample is a fixed number of bytes, so its last line is
                # usually cut mid-record — and a single record can be large. A
                # 25 MB transcript measured here has a 1.36 MB line, so a first
                # record longer than the sample is ordinary, not damage. Reading
                # that as "not this format" handed every such file to the
                # generic reader, which accepts anything shaped like JSON.
                continue
            if not isinstance(record, dict):
                return False
            kind = record.get("type")
            if not isinstance(kind, str) or kind not in KNOWN_TYPES:
                return False
            if "sessionId" in record:
                return True
        return False

    def read(self, stream: TextIO) -> Iterator[RawEvent | MalformedRecord]:
        """Yield events in file order, streaming.

        **One record may produce several events.** A single assistant turn can
        hold text and three tool calls, and those are four things that happened;
        collapsing them into one event would lose the tool calls, which Phase
        8.5 measured to be where answers hide.

        File order is *not* conversation order — see the module docstring. This
        yields what the file contains, faithfully and in order, and leaves
        branch selection to ``active_thread``, which needs the whole file and so
        cannot run inside a streaming read.
        """
        # The parent chain runs *through* records this importer does not import.
        # Measured on real files: 804 `attachment` and 62 `system` records sit
        # between conversation turns in one session alone. Indexing only what is
        # imported breaks the chain at the first attachment, which is why the
        # nearest conversation ancestor is resolved here, while every record is
        # still in view. Parents always precede children in an append-only file,
        # so this costs one dictionary and no buffering.
        parent_of: dict[str, str | None] = {}
        conversational: set[str] = set()

        for index, line in enumerate(stream):
            location = f"line {index + 1}"
            stripped = line.strip()
            if not stripped:
                yield MalformedRecord(index, location, "blank line", reason=BLANK_LINE)
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield MalformedRecord(
                    index,
                    location,
                    f"invalid JSON: {exc}",
                    stripped[:TEXT_SAMPLE_LENGTH],
                    reason="invalid_json",
                )
                continue
            if not isinstance(record, dict):
                yield MalformedRecord(
                    index,
                    location,
                    f"record is not a JSON object, got {type(record).__name__}",
                    stripped[:TEXT_SAMPLE_LENGTH],
                    reason="not_an_object",
                )
                continue
            uuid = record.get("uuid")
            if isinstance(uuid, str) and uuid:
                parent_of[uuid] = _text_or_none(record.get("parentUuid"))

            if record.get("type") not in CONVERSATION_TYPES:
                continue
            if record.get("isMeta"):
                # Injected by the tool rather than said by anyone.
                continue

            ancestor = _nearest(parent_of, conversational, _text_or_none(record.get("parentUuid")))
            if isinstance(uuid, str) and uuid:
                conversational.add(uuid)
            yield from self._events(record, index, ancestor)

    # ------------------------------------------------------------------

    def _events(
        self, record: dict[str, Any], position: int, ancestor: str | None = None
    ) -> Iterator[RawEvent | MalformedRecord]:
        location = f"line {position + 1}"
        message = record.get("message")
        if not isinstance(message, dict):
            yield MalformedRecord(
                position,
                location,
                f"{record.get('type')} record has no message object",
                reason="no_message",
            )
            return

        role = str(message.get("role") or record.get("type") or "")
        default_type = _ROLE_EVENTS.get(role, EventType.OTHER)
        content = message.get("content")

        if isinstance(content, str):
            yield self._event(record, default_type, text=content, ancestor=ancestor)
            return

        if not isinstance(content, list):
            yield MalformedRecord(
                position,
                location,
                f"message content is {type(content).__name__}, expected a string or a list",
                reason="bad_content",
            )
            return

        emitted = False
        skipped_thinking = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")

            if kind == "thinking":
                if not self.include_thinking:
                    skipped_thinking += 1
                    continue
                yield self._event(
                    record, default_type, text=str(block.get("thinking") or ""), ancestor=ancestor
                )
                emitted = True

            elif kind == "text":
                text = str(block.get("text") or "")
                if text.strip():
                    yield self._event(record, default_type, text=text, ancestor=ancestor)
                    emitted = True

            elif kind in _BLOCK_EVENTS:
                yield self._event(
                    record,
                    _BLOCK_EVENTS[kind],
                    text=_block_text(block),
                    tool_name=block.get("name"),
                    tool_call_id=_tool_id(block),
                    ancestor=ancestor,
                )
                emitted = True

        if not emitted and skipped_thinking:
            # A turn that was only thinking. Recorded as an empty turn rather
            # than dropped: something happened here, and a gap in the thread
            # would make the parent chain point at nothing.
            yield self._event(
                record,
                default_type,
                text="",
                ancestor=ancestor,
                metadata={"skipped_thinking_blocks": skipped_thinking},
            )

    def _event(
        self,
        record: dict[str, Any],
        event_type: EventType,
        *,
        text: str,
        tool_name: object = None,
        tool_call_id: object = None,
        ancestor: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RawEvent:
        extra: dict[str, Any] = dict(metadata or {})
        if ancestor is not None:
            # The previous *conversation* turn, with attachments and system
            # records between them skipped. `parent_id` stays verbatim, because
            # the field is documented as the provider's own reference and a
            # derived value there would be a quiet lie.
            extra[CONVERSATION_PARENT] = ancestor
        for field in ("cwd", "gitBranch", "version", "isSidechain", "isCompactSummary"):
            if field in record:
                extra[field] = record[field]

        return RawEvent(
            type=event_type,
            provider=PROVIDER,
            raw=record if self.keep_raw else None,
            source_id=_text_or_none(record.get("uuid")),
            source_type=_text_or_none(record.get("type")),
            timestamp=parse_timestamp(record.get("timestamp")),
            text=text,
            tool_name=_text_or_none(tool_name),
            tool_call_id=_text_or_none(tool_call_id),
            parent_id=_text_or_none(record.get("parentUuid")),
            metadata=extra,
        )


def _text_or_none(value: Any) -> str | None:  # noqa: ANN401  source values are unconstrained
    return value if isinstance(value, str) and value else None


def _nearest(
    parent_of: dict[str, str | None], conversational: set[str], start: str | None
) -> str | None:
    """Walk up until a conversation record is found, or the chain runs out."""
    seen: set[str] = set()
    cursor = start
    while cursor is not None and cursor not in seen:
        if cursor in conversational:
            return cursor
        seen.add(cursor)
        cursor = parent_of.get(cursor)
    return None


def _tool_id(block: dict[str, Any]) -> Any:  # noqa: ANN401  source values
    """The correlation id, under whichever name this block kind uses."""
    return block.get("id") or block.get("tool_use_id")


def _block_text(block: dict[str, Any]) -> str:
    """Readable text for a tool call or result.

    A tool result's content is itself sometimes a list of blocks, so this
    flattens one level. Serialising the whole structure to JSON instead would
    fill the archive with braces a model then has to parse back out.
    """
    for key in ("content", "text", "input"):
        value = block.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [
                str(item.get("text") or "")
                for item in value
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if any(parts):
                return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
    return ""


# ----------------------------------------------------------------------
# Branch selection, which needs the whole file


def active_thread(events: Sequence[RawEvent]) -> list[RawEvent]:
    """The live conversation, with abandoned attempts removed and history kept.

    Two different structures in a session file look alike and must not be
    treated alike. Both were measured on real transcripts.

    **A fork is a rewind.** When a turn is edited or retried, the replacement
    becomes a second child of the same parent and the original stays on disk.
    Only one side of a fork actually happened, so the abandoned side is dropped.

    **A second root is a continuation, not a rival.** When Claude Code compacts a
    session it starts a *new* tree rather than extending the old one. Measured on
    two long transcripts, the trees are strictly sequential and split exactly at
    the ``isCompactSummary`` record: 2,407 records then 388, and 3,294 then 927.
    They are consecutive segments of one conversation, and **both are kept**.

    Conflating the two is not a subtle error. An earlier version of this function
    followed only the chain ending at the last record, which discarded every
    pre-compaction turn — 78% and 86% of those two sessions, and precisely the
    history this project exists to carry across a handoff. One of those files
    contains no forks at all, so nothing about it was ambiguous; the algorithm
    was simply wrong, and only real data showed it.

    Within each segment the live branch is the one ending at that segment's last
    record, since an append-only file writes the surviving branch last.

    **This needs the whole file**, which is why it is separate from ``read``: the
    last record of a segment has to be known before its first can be identified.
    """
    ordered = list(events)
    if not ordered:
        return []

    parent_of: dict[str, str | None] = {}
    for event in ordered:
        if event.source_id is not None and event.source_id not in parent_of:
            # ``conversation_parent``, not ``parent_id``: the raw reference
            # usually points at an attachment, and following it leaves the
            # conversation after one step.
            parent_of[event.source_id] = event.metadata.get(CONVERSATION_PARENT)

    def root_of(source_id: str) -> str:
        seen: set[str] = set()
        cursor = source_id
        while True:
            parent = parent_of.get(cursor)
            if parent is None or parent in seen or parent not in parent_of:
                return cursor
            seen.add(cursor)
            cursor = parent

    # The last event of each segment, in file order.
    leaf_of_segment: dict[str, str] = {}
    for event in ordered:
        if event.source_id is not None:
            leaf_of_segment[root_of(event.source_id)] = event.source_id

    live: set[str] = set()
    for leaf in leaf_of_segment.values():
        cursor: str | None = leaf
        while cursor is not None and cursor not in live:
            live.add(cursor)
            cursor = parent_of.get(cursor)

    return [event for event in ordered if event.source_id is None or event.source_id in live]


def segments(events: Sequence[RawEvent]) -> list[list[RawEvent]]:
    """The conversation split wherever a turn has no conversation ancestor.

    **That is a weaker claim than "split at compaction boundaries", and
    deliberately so.** Compaction does produce such a break — measured on two
    long transcripts, the split falls exactly at the ``isCompactSummary``
    record. But it is not the only cause: a session where a slash command was
    injected showed the same break with no compaction anywhere in the file. So
    this reports the structure it can actually see, and leaves interpreting it
    to a caller who can check ``isCompactSummary`` for themselves.

    Segments are consecutive and in order; concatenating them gives back the
    live conversation.
    """
    live = active_thread(events)
    out: list[list[RawEvent]] = []
    for event in live:
        if not out or event.metadata.get(CONVERSATION_PARENT) is None:
            out.append([])
        out[-1].append(event)
    return out


def abandoned(events: Sequence[RawEvent]) -> list[RawEvent]:
    """What ``active_thread`` left out: the abandoned side of every rewind.

    Offered because "your import dropped 40 messages" deserves an answer better
    than a count.
    """
    kept = {id(event) for event in active_thread(events)}
    return [event for event in events if id(event) not in kept]


def sessions_root() -> Any:  # noqa: ANN401  a Path, imported lazily
    """Where Claude Code keeps transcripts on this machine.

    A convenience for a person at a terminal, not something anything depends on.
    The location is an observed fact about one tool's current layout and may
    change without notice, so nothing here treats it as an interface.
    """
    from pathlib import Path

    return Path.home() / ".claude" / "projects"


def find_sessions(root: Any = None) -> list[Any]:  # noqa: ANN401  Paths
    """Session files on this machine, newest first."""
    from pathlib import Path

    base = Path(root) if root is not None else sessions_root()
    if not base.is_dir():
        return []
    files: Iterable[Path] = base.glob("*/*.jsonl")
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


__all__ = [
    "CONVERSATION_TYPES",
    "KNOWN_TYPES",
    "PROVIDER",
    "ClaudeCodeImporter",
    "abandoned",
    "active_thread",
    "find_sessions",
    "segments",
    "sessions_root",
]
