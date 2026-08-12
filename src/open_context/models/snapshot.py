"""Context snapshots.

A snapshot records what the active context looked like after a compaction. It is
immutable and points at its parent, so the sequence of compactions over a
session is a chain that can be walked backwards and restored from.

A snapshot holds identifiers, not content. It says which messages were active
and which state items were in play; the text stays in the archive. That keeps
snapshots cheap and stops them from becoming a second, diverging copy of the
history.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from open_context.models import ids
from open_context.models.base import Record, Timestamp, utc_now

_STATE_PREFIXES = frozenset(
    {
        ids.GOAL,
        ids.CONSTRAINT,
        ids.FACT,
        ids.DECISION,
        ids.TASK,
        ids.OPEN_QUESTION,
        ids.PREFERENCE,
        ids.ENTITY,
        ids.EVENT,
        ids.ARTIFACT,
    }
)


class TokenEstimate(Record):
    """A token count and how it was arrived at.

    The count is never presented without its method. A character heuristic and a
    provider tokeniser can differ by a wide margin, and a budget decision made on
    the heuristic while believing it exact is how a compiled context overflows.
    """

    value: int = Field(ge=0)
    method: str = Field(
        min_length=1,
        description="Identifier of the counter, such as 'chars/4' or 'anthropic:claude'.",
    )
    exact: bool = Field(
        default=False,
        description="True only when produced by the target model's own tokeniser.",
    )


class ContextSnapshot(Record):
    """An immutable record of one compaction result."""

    id: str = Field(default_factory=lambda: ids.new_id(ids.SNAPSHOT))
    session_id: str
    parent_snapshot_id: str | None = None
    created_at: Timestamp = Field(default_factory=utc_now)
    active_message_ids: list[str] = Field(default_factory=list)
    archived_message_ids: list[str] = Field(default_factory=list)
    state_item_ids: list[str] = Field(default_factory=list)
    token_estimate: TokenEstimate | None = None
    compiler_version: str | None = None
    reason: str | None = Field(
        default=None,
        description="Why this snapshot was taken, such as 'threshold:0.80' or 'manual'.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "parent_snapshot_id")
    @classmethod
    def _check_snapshot_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ids.validate_id(value, expected=ids.SNAPSHOT)

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: str) -> str:
        return ids.validate_id(value, expected=ids.SESSION)

    @field_validator("active_message_ids", "archived_message_ids")
    @classmethod
    def _check_message_ids(cls, value: list[str]) -> list[str]:
        for ref in value:
            ids.validate_id(ref, expected=ids.MESSAGE)
        if len(set(value)) != len(value):
            raise ValueError("message id lists must not contain duplicates")
        return value

    @field_validator("state_item_ids")
    @classmethod
    def _check_state_ids(cls, value: list[str]) -> list[str]:
        for ref in value:
            ids.validate_ref(ref, allowed=_STATE_PREFIXES)
        if len(set(value)) != len(value):
            raise ValueError("state_item_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _check_not_own_parent(self) -> Self:
        if self.parent_snapshot_id == self.id:
            raise ValueError("a snapshot cannot be its own parent")
        return self

    @model_validator(mode="after")
    def _check_active_and_archived_disjoint(self) -> Self:
        overlap = set(self.active_message_ids) & set(self.archived_message_ids)
        if overlap:
            sample = ", ".join(sorted(overlap)[:3])
            raise ValueError(f"a message cannot be both active and archived: {sample}")
        return self
