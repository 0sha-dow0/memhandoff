"""The normalized event.

The smallest provider-neutral description of *what happened* in a conversation.
It deliberately does not describe what the conversation *means*: there is no
importance, no retention class, no decision, no goal. Those are interpretations
and belong to Phase 6.

**The normalized event is an interpretation of the transport format, not a
replacement for the source record.** Every event carries ``raw``, the provider
record as parsed, with unknown fields preserved. If an adapter meets a field
this model does not understand, the field survives in ``raw`` and can be read
back later by code that does understand it. The normalized fields are a
convenience so that a consumer does not have to re-parse a provider format it
has never heard of; they are not the authoritative copy.

That duplication is deliberate and it costs disk. Storing only ``raw`` would
push provider parsing into every reader, and storing only the normalized fields
would silently discard everything the model has no field for.

**``raw`` is the parsed record, not the source bytes.** It is what the adapter's
parser produced, so it survives a round trip through JSON with its values
intact, but not its original representation: key order, whitespace, the exact
spelling of a number, and a repeated key that the parser collapsed are all gone
by the time it arrives here. Byte-level preservation of the source would be a
different feature with a different cost, and this model does not claim it.

**No ordering field.** Position comes from the archive, which assigns ``seq`` on
append. An importer yields events in source order and the writer appends them in
the order received, so archive order is source order. Putting a position on the
event as well would create a second ordering that could disagree with the first.

**Timestamps are never invented.** A source that gives no usable timestamp
produces ``timestamp=None`` rather than the time of import. Import time is a
fact about the import, not about the conversation, and writing it into the
history would make an inspection weeks later unable to tell the two apart.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from open_context.models.base import Timestamp


class EventType(StrEnum):
    """What kind of thing happened.

    The message roles mirror ``models.enums.Role`` so that a later phase turning
    events into ``Message`` records does not have to invent a mapping. ``OTHER``
    exists so that a provider event this model has no name for is still stored
    with its original name in ``source_type``, rather than being dropped for
    being unrecognised.
    """

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
    DEVELOPER_MESSAGE = "developer_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OTHER = "other"


class RawEvent(BaseModel):
    """One event from a conversation source, normalized but not interpreted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: EventType = Field(description="What happened, in provider-neutral terms.")
    provider: str = Field(
        min_length=1,
        description="Which adapter produced this event, such as 'generic-jsonl'.",
    )
    raw: Any = Field(
        default=None,
        description="The provider record as parsed, with unknown fields preserved.",
    )
    source_id: str | None = Field(
        default=None,
        description="The provider's identifier for this event, verbatim. None if it gave none.",
    )
    source_type: str | None = Field(
        default=None,
        description="The provider's own name for this event kind, verbatim. None if it gave none.",
    )
    timestamp: Timestamp | None = Field(
        default=None,
        description="When the provider says it happened. None when absent or unusable.",
    )
    text: str | None = Field(
        default=None,
        description="Human-readable content, when the source carries it as text.",
    )
    tool_name: str | None = None
    tool_call_id: str | None = Field(
        default=None,
        description="The provider's correlation id between a tool call and its result.",
    )
    parent_id: str | None = Field(
        default=None, description="The provider's own parent reference, verbatim and unvalidated."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Adapter-supplied extras. Not provider content."
    )

    def to_payload(self) -> dict[str, Any]:
        """The archive payload for this event.

        Plain JSON types only, so the archive can hash it canonically without
        knowing anything about this model.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Any) -> Self:  # noqa: ANN401  archive payloads are untyped
        """Rebuild an event from an archive payload."""
        if not isinstance(payload, dict):
            raise ValueError(f"event payload must be a JSON object, got {type(payload).__name__}")
        return cls.model_validate(payload)


__all__ = ["EventType", "RawEvent"]
