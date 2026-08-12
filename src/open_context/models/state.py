"""Structured state extracted from conversation.

State items are the durable semantic layer. They are not summaries: each one is
a typed record with a status, a confidence, and a list of sources it can be
checked against.

This is a discriminated union rather than one model with a ``type`` string. The
difference shows up at parse time. A decision without a rationale is not a
decision, it is a claim, and the union rejects it on construction instead of
letting it reach the context compiler and get presented to a model as settled.

There is no ``rejected_decision`` type. A rejected decision is a ``Decision``
with status ``REJECTED``, which keeps the alternatives and the rationale
attached to the thing they explain. Splitting it into a second type would mean
the reason an approach was rejected lives somewhere other than the approach.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, TypeAdapter, field_validator, model_validator

from open_context.models import ids
from open_context.models.base import Record, Timestamp, utc_now
from open_context.models.enums import RetentionClass, StateStatus, StateType, TaskStatus, TrustLevel

_SOURCE_PREFIXES = frozenset({ids.MESSAGE, ids.EVIDENCE})

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


class StateItemBase(Record):
    """Fields common to every state item."""

    id: str
    session_id: str
    content: str = Field(min_length=1)
    status: StateStatus = StateStatus.ACTIVE
    retention: RetentionClass = RetentionClass.IMPORTANT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    trust: TrustLevel = TrustLevel.AGENT_GENERATED
    created_at: Timestamp = Field(default_factory=utc_now)
    sources: list[str] = Field(
        default_factory=list,
        description="Message and evidence ids supporting this item.",
    )
    supersedes: str | None = Field(
        default=None,
        description="The state item this one replaces. The replaced item is kept, not deleted.",
    )
    related_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: str) -> str:
        return ids.validate_id(value, expected=ids.SESSION)

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, value: list[str]) -> list[str]:
        for ref in value:
            ids.validate_ref(ref, allowed=_SOURCE_PREFIXES)
        if len(set(value)) != len(value):
            raise ValueError("sources must not contain duplicates")
        return value

    @field_validator("supersedes", "id")
    @classmethod
    def _check_state_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ids.validate_ref(value, allowed=_STATE_PREFIXES)

    @field_validator("related_ids")
    @classmethod
    def _check_related(cls, value: list[str]) -> list[str]:
        for ref in value:
            ids.validate_ref(ref, allowed=_STATE_PREFIXES)
        return value

    @model_validator(mode="after")
    def _check_not_self_referential(self) -> Self:
        if self.supersedes == self.id:
            raise ValueError("a state item cannot supersede itself")
        if self.id in self.related_ids:
            raise ValueError("a state item cannot be related to itself")
        return self

    @model_validator(mode="after")
    def _check_critical_has_sources(self) -> Self:
        """Critical state must be traceable.

        An unsourced critical item is exactly the failure the runtime exists to
        prevent: a claim that cannot be checked, presented to a model as fact.
        """
        if self.retention is RetentionClass.CRITICAL and not self.sources:
            raise ValueError("a critical state item requires at least one source")
        return self


class Goal(StateItemBase):
    """Something the session is trying to achieve."""

    type: Literal[StateType.GOAL] = StateType.GOAL
    id: str = Field(default_factory=lambda: ids.new_id(ids.GOAL))


class Constraint(StateItemBase):
    """A restriction the work must respect."""

    type: Literal[StateType.CONSTRAINT] = StateType.CONSTRAINT
    id: str = Field(default_factory=lambda: ids.new_id(ids.CONSTRAINT))
    hard: bool = Field(
        default=True,
        description="A hard constraint may not be traded off. A soft one is a preference.",
    )


class Fact(StateItemBase):
    """A statement about the world or the project."""

    type: Literal[StateType.FACT] = StateType.FACT
    id: str = Field(default_factory=lambda: ids.new_id(ids.FACT))
    subject: str | None = Field(
        default=None,
        description="What the fact is about, used to detect a later fact that contradicts it.",
    )


class Decision(StateItemBase):
    """A choice that was made, with the reasoning that produced it.

    ``rationale`` is required. A decision recorded without its reason cannot be
    revisited later, which is the single question people most often go back to a
    long session to answer.
    """

    type: Literal[StateType.DECISION] = StateType.DECISION
    id: str = Field(default_factory=lambda: ids.new_id(ids.DECISION))
    rationale: str = Field(min_length=1)
    alternatives: list[str] = Field(
        default_factory=list,
        description="Options considered and not taken.",
    )


class Task(StateItemBase):
    """A unit of work."""

    type: Literal[StateType.TASK] = StateType.TASK
    id: str = Field(default_factory=lambda: ids.new_id(ids.TASK))
    task_status: TaskStatus = TaskStatus.OPEN
    blocked_by: list[str] = Field(default_factory=list)

    @field_validator("blocked_by")
    @classmethod
    def _check_blocked_by(cls, value: list[str]) -> list[str]:
        for ref in value:
            ids.validate_ref(ref, allowed=_STATE_PREFIXES)
        return value

    @model_validator(mode="after")
    def _check_blocked_consistency(self) -> Self:
        if self.task_status is TaskStatus.BLOCKED and not self.blocked_by:
            raise ValueError("a blocked task must record what blocks it")
        return self


class OpenQuestion(StateItemBase):
    """Something unresolved."""

    type: Literal[StateType.OPEN_QUESTION] = StateType.OPEN_QUESTION
    id: str = Field(default_factory=lambda: ids.new_id(ids.OPEN_QUESTION))


class Preference(StateItemBase):
    """A stated preference about how work should be done."""

    type: Literal[StateType.PREFERENCE] = StateType.PREFERENCE
    id: str = Field(default_factory=lambda: ids.new_id(ids.PREFERENCE))


class Entity(StateItemBase):
    """A named thing the session refers to repeatedly."""

    type: Literal[StateType.ENTITY] = StateType.ENTITY
    id: str = Field(default_factory=lambda: ids.new_id(ids.ENTITY))
    name: str = Field(min_length=1)
    kind: str | None = None


class Event(StateItemBase):
    """Something that happened, at a point in time."""

    type: Literal[StateType.EVENT] = StateType.EVENT
    id: str = Field(default_factory=lambda: ids.new_id(ids.EVENT))
    occurred_at: Timestamp | None = None


class Artifact(StateItemBase):
    """A file, document, or output the session produced or referred to."""

    type: Literal[StateType.ARTIFACT] = StateType.ARTIFACT
    id: str = Field(default_factory=lambda: ids.new_id(ids.ARTIFACT))
    path: str | None = None
    media_type: str | None = None


StateItem = Annotated[
    Goal
    | Constraint
    | Fact
    | Decision
    | Task
    | OpenQuestion
    | Preference
    | Entity
    | Event
    | Artifact,
    Field(discriminator="type"),
]
"""Any state item, resolved by its ``type`` field."""

StateItemAdapter: TypeAdapter[StateItem] = TypeAdapter(StateItem)
"""Parses a state item from JSON or a dict without knowing its type in advance."""
