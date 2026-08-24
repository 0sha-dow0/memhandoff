"""What a model is sent.

A deliberately tiny chat message: the shape every chat API accepts, and the
shape a tokenizer counts. It is not a stored record and has no identity, no
session, and no provenance.

**Why this duplicates ``models.enums.Role``.** The roles happen to coincide
today, but they answer different questions. ``models.Role`` is who authored a
turn in stored history; ``ChatRole`` is what a provider's wire format accepts.
Importing the storage enum here would make the LLM layer depend on the data
model, and the boundary this phase exists to draw is worth more than five
duplicated strings. Converting between them is the caller's job, in whichever
later phase needs it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ChatRole(StrEnum):
    """Who a message is from, in provider-neutral terms."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """One message in a prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ChatRole
    content: str
    name: str | None = Field(
        default=None,
        description="Optional speaker or tool name, where a provider accepts one.",
    )

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role=ChatRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role=ChatRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(role=ChatRole.ASSISTANT, content=content)


__all__ = ["ChatMessage", "ChatRole"]
