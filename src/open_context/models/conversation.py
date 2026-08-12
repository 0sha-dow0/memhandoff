"""Raw conversation records.

These are the source of truth. Everything else in the data model is derived from
them and points back at them. Nothing here is ever rewritten by compaction.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import Field, field_validator, model_validator

from open_context.models import ids
from open_context.models.base import Record, Timestamp, utc_now
from open_context.models.enums import Role, TrustLevel

MessageId = Annotated[str, Field(min_length=3)]
SessionId = Annotated[str, Field(min_length=3)]


class Session(Record):
    """A conversation or agent run."""

    id: SessionId = Field(default_factory=lambda: ids.new_id(ids.SESSION))
    title: str | None = None
    source: str | None = Field(
        default=None,
        description="Where this session was imported from, such as 'chatgpt' or 'generic-json'.",
    )
    created_at: Timestamp = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return ids.validate_id(value, expected=ids.SESSION)


class Message(Record):
    """A single turn in a session.

    ``seq`` exists because timestamps are not a reliable ordering key. Provider
    exports routinely give many messages the same second, and some give no
    per-message timestamp at all. Ordering history correctly is the one thing
    this project cannot get wrong, so order is stored explicitly rather than
    inferred.
    """

    id: MessageId = Field(default_factory=lambda: ids.new_id(ids.MESSAGE))
    session_id: SessionId
    seq: int = Field(
        ge=0,
        description="Position within the session. Unique per session, ascending, gaps allowed.",
    )
    role: Role
    content: str
    timestamp: Timestamp = Field(default_factory=utc_now)
    trust: TrustLevel = TrustLevel.AGENT_GENERATED
    parent_id: MessageId | None = None
    tool_name: str | None = Field(
        default=None,
        description="Name of the tool, required for tool messages and forbidden otherwise.",
    )
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return ids.validate_id(value, expected=ids.MESSAGE)

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: str) -> str:
        return ids.validate_id(value, expected=ids.SESSION)

    @field_validator("parent_id")
    @classmethod
    def _check_parent_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ids.validate_id(value, expected=ids.MESSAGE)

    @model_validator(mode="after")
    def _check_tool_fields(self) -> Self:
        if self.role is Role.TOOL and self.tool_name is None:
            raise ValueError("a tool message must carry tool_name")
        if self.role is not Role.TOOL and self.tool_name is not None:
            raise ValueError(f"tool_name is only valid on tool messages, not {self.role.value}")
        return self

    @model_validator(mode="after")
    def _check_not_own_parent(self) -> Self:
        if self.parent_id is not None and self.parent_id == self.id:
            raise ValueError("a message cannot be its own parent")
        return self
