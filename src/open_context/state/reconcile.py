"""Folding a new extraction into state that already exists.

```
OLD STATE  +  NEW EVENTS  ->  UPDATED STATE
```

**Nothing is deleted, ever.** An item that has been overturned is marked
superseded and keeps its place; the record that a decision *changed* is part of
the state, not noise around it. Reconciliation therefore only ever adds records
and flips statuses, which is what makes the append-only storage underneath a
constraint the design wanted rather than one it works around.

**Extraction cites supersession by content; this resolves it to ids.** The
extractor could not do better — ids are minted after the model has answered — so
it records the prior item's *content* and leaves the resolution here, where the
existing state is actually in hand.

**What it will not do is guess.** Where the incoming item cannot be matched to
what is already stored, or where two successors claim the same predecessor, the
result carries a conflict and the caller decides. Silently keeping the newer one
is a policy nobody chose, and it looks exactly like having found no problem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from open_context.extraction.result import ExtractedItem
from open_context.models.enums import StateStatus
from open_context.models.state import StateItem
from open_context.state.conflict import Conflict, ConflictKind, detect_conflicts


def fingerprint(item: StateItem) -> tuple[str, str]:
    """What makes two records the same item.

    Type and normalised content. Deliberately not the id, which is minted per
    extraction and would make every re-run look like new state, and deliberately
    not confidence or provenance, which can differ between two readings of the
    same sentence without the sentence having changed.

    Crude, and the reason a re-extraction that rewords an item slightly reads as
    a new one. Better matching is a language question; this is the honest
    deterministic answer to it.
    """
    return (item.type.value, " ".join(item.content.lower().split()))


@dataclass
class Reconciliation:
    """What folding an extraction into existing state would do.

    **A plan, not an effect.** Nothing here writes; a caller inspects it,
    resolves any conflict it cannot accept, and only then commits. Returning the
    plan rather than performing it is what lets a conflicting update be refused
    without having already half-applied it.
    """

    added: list[ExtractedItem] = field(default_factory=list)
    """Genuinely new items."""

    unchanged: list[ExtractedItem] = field(default_factory=list)
    """Items already present with identical content. Re-extracting a session
    should not double its state, and this is what stops it."""

    supersedes: list[tuple[str, ExtractedItem]] = field(default_factory=list)
    """``(existing item id, the item replacing it)``.

    Applying one marks the existing item superseded and stores the new item with
    ``supersedes`` set. Both survive.
    """

    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def blocked(self) -> list[Conflict]:
        """Conflicts that need a decision before this can be committed."""
        return [conflict for conflict in self.conflicts if not conflict.auto_resolvable]

    @property
    def safe(self) -> bool:
        return not self.blocked

    def describe(self) -> str:
        parts = [
            f"{len(self.added)} new",
            f"{len(self.unchanged)} unchanged",
            f"{len(self.supersedes)} superseding",
        ]
        if self.conflicts:
            parts.append(f"{len(self.conflicts)} conflicts ({len(self.blocked)} blocking)")
        return ", ".join(parts)


def reconcile(
    existing: Sequence[StateItem],
    incoming: Sequence[ExtractedItem],
) -> Reconciliation:
    """Work out what an extraction changes, without changing anything.

    ``existing`` is the state already held for the session — current items, not
    the full history, since an item that was already superseded cannot be
    superseded again and matching against it would resurrect settled questions.
    """
    known = {fingerprint(item): item for item in existing}
    plan = Reconciliation()

    # Incoming items can supersede each other within one extraction, so their
    # own fingerprints have to be visible to the resolution below. Without this,
    # a reversal found entirely inside one new range would report its own
    # predecessor as missing.
    incoming_by_fingerprint = {
        (entry.item.type.value, " ".join(entry.item.content.lower().split())): entry
        for entry in incoming
    }

    for entry in incoming:
        claim = str(entry.item.metadata.get("supersedes_content") or "").strip()
        if claim:
            # An item claiming to replace something is never a duplicate of what
            # is already stored, whatever its content matches: it is asserting a
            # change, and filing it as "unchanged" would discard the change.
            _resolve_supersession(entry, claim, known, incoming_by_fingerprint, plan)
            continue

        key = fingerprint(entry.item)
        if key in known:
            plan.unchanged.append(entry)
        else:
            plan.added.append(entry)

    plan.conflicts.extend(_forks(plan))
    plan.conflicts.extend(detect_conflicts([*existing, *(entry.item for entry in plan.added)]))
    return plan


def _resolve_supersession(
    entry: ExtractedItem,
    claim: str,
    known: dict[tuple[str, str], StateItem],
    incoming_by_fingerprint: dict[tuple[str, str], ExtractedItem],
    plan: Reconciliation,
) -> None:
    """Match a supersession claim to a stored item, or record that it did not.

    Matched on type and normalised content, the same fingerprint used for
    identity everywhere else — a claim that matched on looser terms than
    identity would let one item supersede something it is not a successor to.
    """
    normalised = " ".join(claim.lower().split())
    target_key = (entry.item.type.value, normalised)

    target = known.get(target_key)
    if target is not None:
        plan.supersedes.append((target.id, entry))
        return

    if target_key in incoming_by_fingerprint:
        # The predecessor arrived in this same extraction. It is already being
        # added, so the supersession is coherent; storing the pair is the
        # caller's to order.
        plan.added.append(entry)
        return

    plan.conflicts.append(
        Conflict(
            kind=ConflictKind.SUPERSEDES_MISSING,
            item_ids=(entry.item.id,),
            detail=(
                f"claims to supersede {claim[:70]!r}, which is not in the current state "
                "for this session"
            ),
        )
    )
    plan.added.append(entry)


def _forks(plan: Reconciliation) -> list[Conflict]:
    """Two incoming items claiming the same predecessor."""
    targets: dict[str, list[str]] = {}
    for target_id, entry in plan.supersedes:
        targets.setdefault(target_id, []).append(entry.item.id)
    return [
        Conflict(
            kind=ConflictKind.SUPERSESSION_FORK,
            item_ids=tuple(ids),
            detail=f"{len(ids)} incoming items each claim to supersede {target}",
        )
        for target, ids in targets.items()
        if len(ids) > 1
    ]


def apply_supersession(existing: StateItem, replacement: StateItem) -> tuple[StateItem, StateItem]:
    """The pair of records a committed supersession produces.

    Returns the old item marked superseded and the new one pointing at it.
    **Neither is discarded**, and the old one keeps its content, sources, and
    timestamps: a superseded decision that lost its rationale would defeat the
    reason for keeping it.
    """
    retired = existing.model_copy(update={"status": StateStatus.SUPERSEDED})
    successor = replacement.model_copy(update={"supersedes": existing.id})
    return retired, successor


__all__ = [
    "Reconciliation",
    "apply_supersession",
    "fingerprint",
    "reconcile",
]
