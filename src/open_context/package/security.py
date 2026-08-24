"""Reading a `.ctx` package that came from somewhere else.

**A package is untrusted input.** That is the whole point of the format — it
travels between machines and between agents — and it is also what makes it the
project's real attack surface. A `.ctx` is a file a stranger can hand you, whose
contents you will paste into a model's system prompt.

**Nothing here claims to sanitize.** Prompt injection cannot be detected
reliably, and a function that removed "the dangerous parts" would be making a
promise it cannot keep — worse than none, because a caller would stop being
careful. What this does is *report*: it says what a package contains that a
reader should know about before handing it to an agent, and leaves the decision
where it belongs.

The one hard rule, from the roadmap, is kept by construction: **nothing in this
project executes anything because it appeared in imported context.** There is no
eval, no shell, no dynamic import, no path resolution off a package field. A
test asserts it stays that way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from open_context.package.format import CtxPackage

MAX_REASONABLE_ITEM = 20_000
"""Characters in one state item before it is worth remarking on.

Not a limit — a package is data and may legitimately be large. But a single
"constraint" holding 50,000 characters is not a constraint, and a reader about
to spend it on a context budget should be told.
"""


class Hazard(StrEnum):
    """What a package might carry that a reader should know about."""

    INSTRUCTION_SHAPED = "instruction_shaped"
    """Text that reads like an instruction to the receiving model.

    Reported, never removed. A legitimate session about prompt engineering will
    trip this, and a determined injection will not — the check is a courtesy to
    an attentive reader, not a control.
    """

    ABSOLUTE_PATH = "absolute_path"
    """An archive location pointing somewhere specific on some machine.

    Harmless here because nothing resolves it, and worth surfacing because a
    reader may otherwise assume it was checked.
    """

    OVERSIZED_ITEM = "oversized_item"
    UNVERIFIED_CONTENT = "unverified_content"
    """State that cannot be checked against any source."""


_INSTRUCTION_PATTERNS = (
    re.compile(r"ignore (all |any |the )?(previous|prior|above|earlier)", re.I),
    re.compile(r"disregard (all |any |the )?(previous|prior|above|earlier)", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"\bnew (system )?(instructions?|prompt)\b", re.I),
    re.compile(r"</?(system|instructions?)>", re.I),
    re.compile(r"\b(rm|del|drop)\s+(-rf?\s+|table\s+)", re.I),
    re.compile(r"\bcurl\b.{0,40}\|\s*(sh|bash)", re.I),
)
"""Shapes worth mentioning. Deliberately few.

A long list would produce noise on ordinary engineering conversations — this
project's own sessions discuss `rm -rf` and system prompts constantly — and
noise is what makes a reader stop reading warnings.
"""


@dataclass(frozen=True)
class Note:
    """One thing worth knowing before trusting a package."""

    hazard: Hazard
    where: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.hazard}] {self.where}: {self.detail}"


@dataclass
class SecurityReport:
    """What a package carries. **Not a verdict on whether it is safe.**"""

    notes: list[Note] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Nothing was remarked on.

        Not "this package is safe" — an injection this check does not recognise
        leaves the report clean, and saying otherwise would be the false promise
        this module exists to avoid.
        """
        return not self.notes

    def of(self, hazard: Hazard) -> list[Note]:
        return [note for note in self.notes if note.hazard is hazard]

    def __str__(self) -> str:
        if not self.notes:
            return "nothing remarked on (which is not a guarantee of safety)"
        return "\n".join(str(note) for note in self.notes)


def review(package: CtxPackage) -> SecurityReport:
    """Report what a package carries, for a reader deciding whether to use it."""
    report = SecurityReport()

    for item in package.state:
        where = f"{item.type.value} {item.id}"
        content = item.content or ""

        for pattern in _INSTRUCTION_PATTERNS:
            match = pattern.search(content)
            if match:
                report.notes.append(
                    Note(
                        Hazard.INSTRUCTION_SHAPED,
                        where,
                        f"contains {match.group(0)!r}, which reads as an instruction rather "
                        f"than a record of the work",
                    )
                )
                break

        if len(content) > MAX_REASONABLE_ITEM:
            report.notes.append(
                Note(
                    Hazard.OVERSIZED_ITEM,
                    where,
                    f"{len(content)} characters, which will dominate any context it is "
                    f"compiled into",
                )
            )

    traced = {reference.state_id for reference in package.evidence if reference.resolvable}
    untraced = [item for item in package.state if item.id not in traced]
    if untraced:
        report.notes.append(
            Note(
                Hazard.UNVERIFIED_CONTENT,
                "package",
                f"{len(untraced)} of {len(package.state)} state items name no source that "
                f"can be followed, so nothing they assert can be checked against a conversation",
            )
        )

    if package.archive is not None and package.archive.location.startswith("/"):
        report.notes.append(
            Note(
                Hazard.ABSOLUTE_PATH,
                "archive.location",
                f"{package.archive.location!r} names a path on the machine that built this "
                f"package. Nothing resolves it, and nothing should.",
            )
        )

    return report


__all__ = [
    "MAX_REASONABLE_ITEM",
    "Hazard",
    "Note",
    "SecurityReport",
    "review",
]
