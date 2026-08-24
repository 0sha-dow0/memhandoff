"""Turning archived events into something a model can read and a tokenizer can count.

The archive stores ``RawEvent``, which is provider-neutral but not a chat
message. The LLM layer speaks ``ChatMessage`` and knows nothing about events.
This module is the join, and it is the only place in compaction that knows both
shapes.

**Rendering is not interpretation.** An event becomes text that says what
happened, labelled with what kind of thing it was. Nothing here decides what an
event means, ranks it, or drops it for being uninteresting. A tool result that
looks like noise is rendered exactly like one that turns out to matter, because
telling them apart is the research problem, not a rendering detail.

**Events with no text still render.** A tool call carries its arguments in
``raw`` rather than in ``text``, so falling back to the raw record is how the
call survives into the summary input at all. Dropping it would silently remove
work the agent did.
"""

from __future__ import annotations

import json

from open_context.importers import EventType, RawEvent
from open_context.llm import ChatMessage, ChatRole

ROLES: dict[EventType, ChatRole] = {
    EventType.USER_MESSAGE: ChatRole.USER,
    EventType.SYSTEM_MESSAGE: ChatRole.SYSTEM,
    EventType.DEVELOPER_MESSAGE: ChatRole.DEVELOPER,
    EventType.ASSISTANT_MESSAGE: ChatRole.ASSISTANT,
    # A tool call is something the assistant did; its result comes back as tool.
    EventType.TOOL_CALL: ChatRole.ASSISTANT,
    EventType.TOOL_RESULT: ChatRole.TOOL,
}
"""Event kind to wire role.

``OTHER`` is deliberately absent: an event this build has no name for has no
defensible role either, and ``label_of`` keeps the provider's own word for it in
the rendered text so nothing is lost by defaulting the role to ``user``.
"""

FALLBACK_ROLE = ChatRole.USER


def label_of(event: RawEvent) -> str:
    """A short human- and model-readable tag for an event.

    Includes the tool name and the provider's own type name where present, since
    for a tool call that is most of what the event says about itself.
    """
    parts = [event.type.value]
    if event.tool_name:
        parts.append(f"name={event.tool_name}")
    elif event.type is EventType.OTHER and event.source_type:
        parts.append(f"type={event.source_type}")
    return " ".join(parts)


def body_of(event: RawEvent) -> str:
    """The event's content, falling back to its raw record.

    ``text`` when the adapter found text. Otherwise the provider record, which
    is where a tool call keeps its arguments and where anything this build does
    not model still lives.
    """
    if event.text is not None:
        return event.text
    if event.raw is None:
        return ""
    if isinstance(event.raw, str):
        return event.raw
    return json.dumps(event.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def render_event(event: RawEvent) -> str:
    """One event as a single labelled block."""
    return f"[{label_of(event)}] {body_of(event)}"


def to_message(event: RawEvent) -> ChatMessage:
    """One event as a chat message, for counting and for prompting."""
    return ChatMessage(role=ROLES.get(event.type, FALLBACK_ROLE), content=render_event(event))


def render_events(events: list[RawEvent] | tuple[RawEvent, ...]) -> str:
    """A run of events as one block of text, in order."""
    return "\n".join(render_event(event) for event in events)


__all__ = [
    "FALLBACK_ROLE",
    "ROLES",
    "body_of",
    "label_of",
    "render_event",
    "render_events",
    "to_message",
]
