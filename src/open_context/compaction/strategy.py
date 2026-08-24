"""The strategy seam.

One protocol and one request type, so a later strategy can replace the baseline
without the caller changing. That is the whole abstraction: no factory, no
registry, no plugin loading, no injection container. A second strategy is a
second class implementing three members.

The request carries the archive rather than a session id and a way to find it.
A ``SessionLog`` already knows its own session, so passing both would create two
sources of truth for which conversation is being compacted and a way for them to
disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from open_context.archive import SessionLog
from open_context.compaction.budget import BaselineConfig
from open_context.compaction.result import BaselineCompactionResult


@dataclass(frozen=True)
class CompactionRequest:
    """What to compact, and how small."""

    log: SessionLog
    target_tokens: int
    config: BaselineConfig = field(default_factory=BaselineConfig)

    @property
    def session_id(self) -> str:
        return self.log.session_id


@runtime_checkable
class CompactionStrategy(Protocol):
    """Turns a conversation into a smaller representation of it."""

    @property
    def name(self) -> str:
        """Stable identifier, recorded on every result."""

    @property
    def version(self) -> int:
        """Bumped when the algorithm changes in a way that moves results."""

    def compact(self, request: CompactionRequest) -> BaselineCompactionResult:
        """Produce a representation fitting ``request.target_tokens``."""


__all__ = ["CompactionRequest", "CompactionStrategy"]
