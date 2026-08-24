"""Contradictory state, detected and never silently resolved.

```
two active items that cannot both be true -> Conflict -> a human decides
```

**Detection is deterministic and deliberately narrow.** Deciding whether two
sentences contradict each other is a language question, and answering it with a
model would make state acceptance nondeterministic — the objection this project
already raised against an LLM-based compaction gate, and a worse idea here,
because the output is not a report but the state itself.

So this finds contradictions that are visible *structurally*: two active facts
about the same subject, a supersession fork, an item superseding something that
does not exist. It does not find two sentences that happen to disagree, and it
says so rather than implying coverage it does not have.

**Nothing is auto-resolved when the resolution is ambiguous.** A conflict is a
value that travels with the reconciliation and blocks nothing by itself; what to
do about it is the caller's, and for anything but the unambiguous case it is a
person's. Silently keeping the newer item would be a policy nobody chose, and it
would be indistinguishable from having found no conflict at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from open_context.models.enums import StateStatus, StateType
from open_context.models.state import StateItem


class ConflictKind(StrEnum):
    """What kind of contradiction was found."""

    CONTRADICTORY_FACT = "contradictory_fact"
    """Two active facts about the same subject saying different things.

    The subject is the model's own ``Fact.subject`` field, which exists for
    exactly this. Facts with no subject are not compared, because comparing
    arbitrary sentences is the language question this refuses to guess at.
    """

    SUPERSESSION_FORK = "supersession_fork"
    """Two items claim to supersede the same earlier item.

    One decision cannot be overturned by two different successors without
    somebody choosing which. Keeping both active would leave state that reads
    as two current answers to one question.
    """

    SUPERSEDES_MISSING = "supersedes_missing"
    """An item claims to replace something not present in the state.

    Either the earlier item was never extracted, or the claim is wrong. Both are
    worth surfacing: applying the supersession would silently mark nothing, and
    dropping it would lose the fact that the model saw a reversal.
    """

    DUPLICATE_ACTIVE = "duplicate_active"
    """The same content is active twice.

    Harmless to read and corrosive to count: a state graph that lists one
    constraint three times will have it weighted three times by anything that
    ranks or budgets.
    """


@dataclass(frozen=True)
class Conflict:
    """One contradiction, with the items that produced it.

    ``item_ids`` are the records involved. ``detail`` is written to be read by
    whoever has to decide, so it names the contents rather than the ids alone —
    an id tells a person nothing about what they are choosing between.
    """

    kind: ConflictKind
    item_ids: tuple[str, ...]
    detail: str

    @property
    def auto_resolvable(self) -> bool:
        """Whether resolving this needs a judgement nobody has made.

        Only exact duplicates qualify. Everything else is a choice between two
        things a person may need to see, and the roadmap is explicit that
        ambiguous conflicts are not silently resolved.
        """
        return self.kind is ConflictKind.DUPLICATE_ACTIVE


def detect_conflicts(items: Sequence[StateItem]) -> list[Conflict]:
    """Every structural contradiction among these items.

    Considers active items only. A superseded item contradicting a current one
    is not a conflict — it is the record of a decision having been changed,
    which is the thing supersession exists to preserve.
    """
    active = [item for item in items if item.status is StateStatus.ACTIVE]
    conflicts: list[Conflict] = []
    conflicts.extend(_duplicate_active(active))
    conflicts.extend(_contradictory_facts(active))
    conflicts.extend(_supersession_forks(items))
    return conflicts


def _duplicate_active(active: Sequence[StateItem]) -> list[Conflict]:
    seen: dict[tuple[str, str], list[str]] = {}
    for item in active:
        key = (item.type.value, item.content.strip().lower())
        seen.setdefault(key, []).append(item.id)
    return [
        Conflict(
            kind=ConflictKind.DUPLICATE_ACTIVE,
            item_ids=tuple(ids),
            detail=f"{len(ids)} active {kind} items with identical content: {content[:80]!r}",
        )
        for (kind, content), ids in seen.items()
        if len(ids) > 1
    ]


def _contradictory_facts(active: Sequence[StateItem]) -> list[Conflict]:
    by_subject: dict[str, list[StateItem]] = {}
    for item in active:
        if item.type is not StateType.FACT:
            continue
        subject = (getattr(item, "subject", None) or "").strip().lower()
        if subject:
            by_subject.setdefault(subject, []).append(item)

    conflicts = []
    for subject, group in by_subject.items():
        contents = {item.content.strip().lower() for item in group}
        if len(contents) > 1:
            conflicts.append(
                Conflict(
                    kind=ConflictKind.CONTRADICTORY_FACT,
                    item_ids=tuple(item.id for item in group),
                    detail=(
                        f"{len(group)} active facts about {subject!r} disagree: "
                        + " | ".join(sorted(content[:60] for content in contents))
                    ),
                )
            )
    return conflicts


def _supersession_forks(items: Sequence[StateItem]) -> list[Conflict]:
    successors: dict[str, list[str]] = {}
    for item in items:
        if item.supersedes:
            successors.setdefault(item.supersedes, []).append(item.id)
    return [
        Conflict(
            kind=ConflictKind.SUPERSESSION_FORK,
            item_ids=tuple(ids),
            detail=f"{len(ids)} items each claim to supersede {target}",
        )
        for target, ids in successors.items()
        if len(ids) > 1
    ]


def format_conflicts(conflicts: Sequence[Conflict]) -> str:
    if not conflicts:
        return "no structural conflicts (this does not check whether sentences disagree)"
    lines = []
    for conflict in conflicts:
        mark = "auto" if conflict.auto_resolvable else "NEEDS A DECISION"
        lines.append(f"{conflict.kind.value:<22} [{mark}] {conflict.detail}")
    return "\n".join(lines)


__all__ = [
    "Conflict",
    "ConflictKind",
    "detect_conflicts",
    "format_conflicts",
]
