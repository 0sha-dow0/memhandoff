"""Where a session sits: project, session, event.

    PROJECT
     ↓
    SESSION
     ↓
    EVENT

**The logical model only.** No graph database, no second store, no new
infrastructure — the roadmap is explicit that a graph-shaped model is not a
reason to install a graph engine, and this is three levels with a parent link,
which is a dictionary.

**Project identity is a repository root, not a working directory.** Real
transcripts carry `cwd` on every record, and the obvious reading — one `cwd`, one
project — is wrong in a way that shows up immediately on real data. Of six
sessions measured here, two ran in

    /Users/…/open-context
    /Users/…/open-context/files (4)/open-context-runtime 2

which is one project and a directory inside it. Treating those as two projects
would hide a session's own history from itself, which is the exact failure this
level exists to prevent.

So a project is the nearest ancestor holding a repository marker, and the working
directory only when there is none. That is a filesystem question and is answered
by looking, never by parsing paths for something that resembles a project name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from open_context.models.enums import RetentionClass, StateType
from open_context.models.state import StateItem

MARKERS = (".git", ".hg", ".svn")
"""What makes a directory a project root.

Repository markers, because that is what a person means by "this project" and
what the agent itself already tracks — every transcript record carries a
`gitBranch` alongside its `cwd`.
"""


class Level(StrEnum):
    """A rung of the hierarchy.

    ``TASK`` from the roadmap's sketch is deliberately absent. Nothing in a real
    transcript marks where one task ends and the next begins, so a task level
    would have to be inferred — and a boundary invented here would look exactly
    like one that was observed. It goes in when something real delimits it.
    """

    PROJECT = "project"
    SESSION = "session"
    EVENT = "event"


RETENTION_BY_LEVEL: dict[Level, RetentionClass] = {
    Level.PROJECT: RetentionClass.CRITICAL,
    Level.SESSION: RetentionClass.IMPORTANT,
    Level.EVENT: RetentionClass.RECONSTRUCTABLE,
}
"""Default retention per level, which is the point of having levels.

A constraint that holds for the whole project outlives the session that stated
it; a detail of one exchange does not. Events are reconstructable because the
archive still has them — nothing here is a deletion policy, only an ordering for
what to carry when there is not room for everything.
"""

PROJECT_TYPES = frozenset({StateType.GOAL, StateType.CONSTRAINT, StateType.DECISION})
"""State kinds that usually outlive the session that produced them.

A guess about the common case, applied only where nothing better is known, and
overridable per item. "Never write to the shared NFS mount" is true of the
project; "the file I am editing is at line 40" is true of a moment.
"""


def project_root(directory: str | Path) -> Path:
    """The repository this directory belongs to, or the directory itself.

    Walks upward looking for a marker. Returns the directory unchanged when
    there is none, so a session outside any repository is its own project rather
    than being attached to whatever happens to sit above it.
    """
    path = Path(directory).expanduser()
    candidates = [path, *path.parents] if path.is_absolute() else [path]
    for candidate in candidates:
        if any((candidate / marker).exists() for marker in MARKERS):
            return candidate
    return path


def project_id(directory: str | Path) -> str:
    """A stable name for a project, derived from its root."""
    return str(project_root(directory))


@dataclass(frozen=True)
class Placement:
    """Where one state item sits, and how long it should outlive its session."""

    item: StateItem
    level: Level
    retention: RetentionClass

    @property
    def project_scoped(self) -> bool:
        return self.level is Level.PROJECT


def place(item: StateItem, *, project_types: Iterable[StateType] = PROJECT_TYPES) -> Placement:
    """Decide which level an item belongs to.

    **An item that says what it wants is believed.** A `CRITICAL` retention set
    by extraction or by a person is a judgement about *this* item; the type-based
    guess is about its kind in general, and the specific claim wins.
    """
    wanted = set(project_types)
    if item.retention is RetentionClass.CRITICAL:
        return Placement(item, Level.PROJECT, RetentionClass.CRITICAL)
    if item.retention is RetentionClass.EPHEMERAL:
        return Placement(item, Level.EVENT, RetentionClass.EPHEMERAL)
    if item.type in wanted:
        return Placement(item, Level.PROJECT, RETENTION_BY_LEVEL[Level.PROJECT])
    return Placement(item, Level.SESSION, RETENTION_BY_LEVEL[Level.SESSION])


@dataclass
class Project:
    """One project and the sessions under it."""

    root: str
    sessions: dict[str, list[StateItem]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.root).name or self.root

    def add(self, session_id: str, items: Sequence[StateItem]) -> None:
        self.sessions.setdefault(session_id, []).extend(items)

    def carried_forward(self, *, exclude: str = "") -> list[StateItem]:
        """Project-level state from every session but the one named.

        This is the whole reason the level exists: a constraint agreed in
        Monday's session is still true on Tuesday, and a session that could only
        see itself would ask about it again — or worse, decide differently.

        Duplicates are collapsed on type and content, because the same constraint
        restated in three sessions is one constraint, and a receiving agent shown
        it three times learns nothing the third time.
        """
        seen: set[tuple[str, str]] = set()
        out: list[StateItem] = []
        for session_id, items in self.sessions.items():
            if session_id == exclude:
                continue
            for item in items:
                if not place(item).project_scoped:
                    continue
                key = (item.type.value, " ".join(item.content.lower().split()))
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
        return out


def group_by_project(
    sessions: Iterable[tuple[str, str, Sequence[StateItem]]],
) -> dict[str, Project]:
    """Group ``(session_id, working_directory, items)`` into projects.

    A plain dictionary. The model is a tree three levels deep with a parent
    pointer; storing it in a graph engine would buy traversal nobody is doing
    and cost a service to run.
    """
    projects: dict[str, Project] = {}
    for session_id, directory, items in sessions:
        root = project_id(directory)
        projects.setdefault(root, Project(root=root)).add(session_id, items)
    return projects


__all__ = [
    "MARKERS",
    "PROJECT_TYPES",
    "RETENTION_BY_LEVEL",
    "Level",
    "Placement",
    "Project",
    "group_by_project",
    "place",
    "project_id",
    "project_root",
]
