"""Keeping state between runs, so a session is not re-extracted from scratch.

Phase 7 built watermarked incremental extraction and nothing used it: every
package re-read the whole archive and paid for the whole conversation again.
This is the missing half — somewhere to keep the watermark and the state it
earned, so the second run over a session only pays for what arrived since the
first.

**The file is the boundary of what this promises.** State and a watermark, as
JSON, beside the archive. No database, no server, no schema migration: the data
is small, the reader is one process, and the failure mode of a corrupt file is
"extract again", which is exactly what would have happened anyway.

**A watermark is only ever as good as the state stored with it.** Storing them
apart would let a crash leave a watermark saying "everything up to event 400 has
been read" beside state that only knows about the first 200 — and nothing
downstream could tell. They are written together or not at all.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from open_context.models.state import StateItem, StateItemAdapter
from open_context.state.incremental import Watermark

FORMAT = "open-context-state"
FORMAT_VERSION = 1


@dataclass
class StoredState:
    """What one session has accumulated so far."""

    session_id: str
    next_seq: int = 0
    items: list[StateItem] = field(default_factory=list)
    project_root: str = ""
    """Which project this session belonged to.

    Stored because sibling lookup is otherwise impossible: a directory of state
    files says which sessions exist and nothing about which of them are related.
    Without it, carrying state forward would mix unrelated projects — which it
    did, until a test caught it.
    """

    @property
    def watermark(self) -> Watermark:
        return Watermark(session_id=self.session_id, next_seq=self.next_seq)

    @property
    def fresh(self) -> bool:
        """Whether nothing has been read yet."""
        return self.next_seq == 0 and not self.items


def path_for(root: str | Path, session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64] or "session"
    return Path(root) / "state" / f"{safe}.json"


def load(root: str | Path, session_id: str) -> StoredState:
    """Read what is known, or start empty.

    **Damage is treated as "nothing known", never as an error.** A truncated
    file means the previous run died mid-write; the correct response is to read
    the session again, which is what an empty state produces. Refusing to run
    would turn a recoverable interruption into a permanent one.
    """
    path = path_for(root, session_id)
    if not path.is_file():
        return StoredState(session_id=session_id)

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("format") != FORMAT or document.get("format_version") != FORMAT_VERSION:
            return StoredState(session_id=session_id)
        if document.get("session_id") != session_id:
            # A file naming another session is not this session's state, however
            # it got here. Reading it would attribute one conversation's
            # decisions to another.
            return StoredState(session_id=session_id)
        items = [StateItemAdapter.validate_python(raw) for raw in document.get("items", [])]
        return StoredState(
            session_id=session_id,
            next_seq=int(document.get("next_seq", 0)),
            items=items,
            project_root=str(document.get("project_root") or ""),
        )
    except (OSError, ValueError, TypeError):
        return StoredState(session_id=session_id)


def save(root: str | Path, state: StoredState) -> Path:
    """Write state and watermark together, atomically.

    Replaced through a temporary file in the same directory, so a reader either
    sees the previous version or the new one. A partial write would leave a
    watermark ahead of the state that earned it, and nothing downstream could
    detect that.
    """
    path = path_for(root, state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "session_id": state.session_id,
        "next_seq": state.next_seq,
        "project_root": state.project_root,
        "items": [item.model_dump(mode="json") for item in state.items],
    }

    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "StoredState",
    "load",
    "path_for",
    "save",
]
