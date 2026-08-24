"""Reading a `.ctx` package as a person.

The format is already JSON a person *can* read. This is for the case where they
should not have to: a package with sixty state items is legible in principle and
exhausting in practice, and "human-inspectable" is a property of what a reader
can actually take in.

**It shows what is missing as prominently as what is present.** A rendering that
listed forty confident state items and quietly omitted that none of them can be
traced to a source would be the most misleading possible view of that package.
"""

from __future__ import annotations

from open_context.models.enums import StateType
from open_context.models.state import StateItem
from open_context.package.format import CtxPackage

_ORDER: tuple[StateType, ...] = (
    StateType.GOAL,
    StateType.CONSTRAINT,
    StateType.FACT,
    StateType.DECISION,
    StateType.TASK,
    StateType.OPEN_QUESTION,
    StateType.ENTITY,
)
"""Reading order: what the work is for, what limits it, what is true, what was
decided, what is left. The same order the hybrid compactor renders under budget,
because a reader who has seen one should not have to relearn the other."""

_WIDTH = 78


def _wrap(text: str, indent: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=_WIDTH - indent) or [""]


def summarize(package: CtxPackage) -> str:
    """One screen: what this package is and whether it can be trusted."""
    manifest = package.manifest
    lines = [
        f"{manifest.format} v{manifest.format_version}",
        f"package  {manifest.package_id}",
        f"session  {manifest.session_id}",
    ]
    if manifest.title:
        lines.append(f"title    {manifest.title}")
    if manifest.source:
        lines.append(f"source   {manifest.source}")
    lines.append(f"created  {manifest.created_at.isoformat()}")
    if manifest.created_by:
        lines.append(f"by       {manifest.created_by}")

    lines.append("")
    lines.append(f"state    {len(package.state)} items")

    traced = sum(1 for reference in package.evidence if reference.resolvable)
    if package.state:
        lines.append(f"traced   {traced} of {len(package.state)} to an archive record")

    if package.archive is not None:
        archive = package.archive
        lines.append(
            f"archive  {archive.record_count} records, seq {archive.first_seq}"
            f"-{archive.last_seq}, {len(archive.record_hashes)} hashed"
        )
        if archive.location:
            lines.append(f"         was at {archive.location} (a hint, not resolved)")
    else:
        lines.append("archive  none — nothing here can be traced to a source")

    if package.recent is not None:
        lines.append(f"recent   {len(package.recent.messages)} verbatim messages")

    lines.append(
        f"hash     {package.content_hash[:16]}... {'intact' if package.intact() else 'MISMATCH'}"
    )
    return "\n".join(lines)


def render(package: CtxPackage, *, show_evidence: bool = True) -> str:
    """The whole package, laid out for reading."""
    by_state = {reference.state_id: reference for reference in package.evidence}
    lines = [summarize(package), ""]

    grouped: dict[StateType, list[StateItem]] = {}
    for item in package.state:
        grouped.setdefault(item.type, []).append(item)

    ordered = [t for t in _ORDER if t in grouped] + [t for t in grouped if t not in _ORDER]
    for state_type in ordered:
        items = grouped[state_type]
        lines.append(f"{state_type.value.upper()}  ({len(items)})")
        for item in items:
            for index, line in enumerate(_wrap(item.content, 4)):
                lines.append(f"    {line}" if index else f"  - {line}")

            reference = by_state.get(item.id)
            if not show_evidence or reference is None:
                continue
            if reference.resolvable:
                seqs = ", ".join(str(seq) for seq in reference.archive_seqs)
                lines.append(f"      from archive {seqs}")
                if reference.excerpt:
                    for line in _wrap(f'"{reference.excerpt}"', 8):
                        lines.append(f"        {line}")
            else:
                lines.append("      no traceable source")
        lines.append("")

    if not package.state:
        lines.append("(no state items)")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render", "summarize"]
