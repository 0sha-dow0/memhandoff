"""The archive root.

A directory holding one append-only log per session.

```
<root>/
    sessions/
        ses_<id>/
            manifest.json
            records.jsonl
            records.idx
```

Sessions are separate files rather than one shared log so that opening a session
never touches another session's bytes, and so a damaged session cannot cost the
others.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from open_context.archive.errors import ArchiveExistsError, ArchiveNotFoundError
from open_context.archive.log import DATA_FILE, MANIFEST_FILE, SessionLog

SESSIONS_DIR = "sessions"


class Archive:
    """A local archive directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.sessions_root = self.root / SESSIONS_DIR
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def _directory(self, session_id: str) -> Path:
        if "/" in session_id or "\\" in session_id or session_id in {"", ".", ".."}:
            raise ValueError(f"unsafe session id {session_id!r}")
        return self.sessions_root / session_id

    def exists(self, session_id: str) -> bool:
        return (self._directory(session_id) / MANIFEST_FILE).is_file()

    def create(self, session_id: str) -> SessionLog:
        """Create a new empty log. Raises if one already exists."""
        if self.exists(session_id):
            raise ArchiveExistsError(session_id)
        directory = self._directory(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        log = SessionLog(directory)
        log.data_path.touch()
        log.index.clear()
        log._write_manifest(session_id, datetime.now(UTC).isoformat())
        return log

    def open(self, session_id: str) -> SessionLog:
        """Open an existing log, recovering a torn tail if there is one."""
        if not self.exists(session_id):
            raise ArchiveNotFoundError(session_id)
        log = SessionLog(self._directory(session_id))
        log._open_existing()
        return log

    def open_or_create(self, session_id: str) -> SessionLog:
        return self.open(session_id) if self.exists(session_id) else self.create(session_id)

    def sessions(self) -> Iterator[str]:
        """Session ids present in the archive, in sorted order."""
        if not self.sessions_root.is_dir():
            return
        for entry in sorted(self.sessions_root.iterdir()):
            if (entry / MANIFEST_FILE).is_file():
                yield entry.name

    def size_bytes(self, session_id: str) -> int:
        """Bytes of raw conversation stored for a session."""
        path = self._directory(session_id) / DATA_FILE
        return path.stat().st_size if path.exists() else 0
