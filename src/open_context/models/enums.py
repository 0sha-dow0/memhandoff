"""Enumerations shared across the data model.

All are ``StrEnum`` so they serialise as plain strings and survive a round trip
through JSON or SQLite without a custom codec.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Author of a message."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SourceType(StrEnum):
    """Where a piece of evidence came from."""

    MESSAGE = "message"
    TOOL_RESULT = "tool_result"
    ARTIFACT = "artifact"
    EXTERNAL = "external"


class TrustLevel(StrEnum):
    """How much authority content carries.

    Conversation content is untrusted input. Text reaching the runtime through a
    tool result or a fetched page can contain instructions aimed at the model,
    and nothing recovered from the archive may override system or developer
    instructions in a compiled context. Trust is recorded at ingest so the
    compiler can act on it later.
    """

    SYSTEM = "system"
    USER_AUTHORED = "user_authored"
    AGENT_GENERATED = "agent_generated"
    EXTERNAL = "external"
    UNTRUSTED = "untrusted"


class RetentionClass(StrEnum):
    """How aggressively a piece of information may be compacted."""

    CRITICAL = "critical"
    IMPORTANT = "important"
    RECONSTRUCTABLE = "reconstructable"
    EPHEMERAL = "ephemeral"


class StateType(StrEnum):
    """Discriminator values for the state item union."""

    GOAL = "goal"
    CONSTRAINT = "constraint"
    FACT = "fact"
    DECISION = "decision"
    TASK = "task"
    OPEN_QUESTION = "open_question"
    PREFERENCE = "preference"
    ENTITY = "entity"
    EVENT = "event"
    ARTIFACT = "artifact"


class StateStatus(StrEnum):
    """Lifecycle of a state item.

    ``SUPERSEDED`` means a later item replaced this one and points back at it.
    The original is never edited or deleted, so the history of a changing fact
    stays reconstructable.
    """

    PROPOSED = "proposed"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class TaskStatus(StrEnum):
    """Lifecycle of a task."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    ABANDONED = "abandoned"
