"""Checking an extraction against the conversation it came from.

Phase 6's exit criterion is that structured context can be *validated against
original source material*, not merely produced. An extractor with no validator
is a machine for generating confident claims.

```
ExtractionResult + SessionLog -> findings
```

**Deterministic, offline, and no model is called.** A validator that asked a
model whether the extraction was right would inherit the failure mode it exists
to detect, and would make validity nondeterministic at the margin — the same
objection the project already raised against an LLM-based compaction gate.

## What this can and cannot establish

It checks **grounding**: that every item cites records which exist, in range,
and whose text plausibly supports it. It does not check **completeness** — that
nothing important was missed — because the conversation does not carry a list of
what should have been found. Recall is what the Phase 5.5 retention scenarios
measure, and conflating the two would let an extraction that found one decision
out of ten score perfectly here.

So a clean validation means "nothing asserted is unsupported". It does not mean
"everything present was found".
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from open_context.archive import SessionLog
from open_context.compaction import render_events
from open_context.extraction.result import ExtractedItem, ExtractionResult
from open_context.importers import RawEvent
from open_context.models.enums import StateType

MIN_OVERLAP = 0.34
"""Fraction of an item's distinctive words that must appear in its sources.

A blunt instrument, and named as one. It catches an item citing a record that
plainly does not mention what the item claims; it cannot catch a subtly wrong
paraphrase, and it will flag a correct item written in words the source did not
use. Findings are therefore reported for a human to read, never used to delete
anything.
"""

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "must",
        "not",
        "of",
        "on",
        "or",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "will",
        "with",
        "would",
        "we",
        "you",
        "our",
        "your",
    ]
)


class Severity(StrEnum):
    ERROR = "error"
    """The extraction asserts something the archive cannot support."""

    WARNING = "warning"
    """Suspicious, and possibly fine. Read it."""


@dataclass(frozen=True)
class Finding:
    """One thing wrong, or possibly wrong, with an extraction."""

    severity: Severity
    code: str
    detail: str
    item_content: str = ""


def validate(result: ExtractionResult, log: SessionLog) -> list[Finding]:
    """Every finding, most severe first.

    An empty list means nothing asserted is unsupported. It does not mean the
    extraction is complete.
    """
    records = {record.id: record for record in log.range(0, None)}
    texts = {
        record_id: render_events((RawEvent.from_payload(record.payload),)).lower()
        for record_id, record in records.items()
    }

    findings: list[Finding] = []
    findings.extend(_check_session(result))
    for entry in result.items:
        findings.extend(_check_citations(entry, result, records, texts))
        findings.extend(_check_shape(entry))
    findings.extend(_check_supersession(result.items))

    findings.sort(key=lambda f: 0 if f.severity is Severity.ERROR else 1)
    return findings


def _check_session(result: ExtractionResult) -> Iterable[Finding]:
    for entry in result.items:
        if entry.item.session_id != result.session_id:
            yield Finding(
                severity=Severity.ERROR,
                code="wrong_session",
                detail=(
                    f"item is scoped to {entry.item.session_id} but the extraction is "
                    f"{result.session_id}"
                ),
                item_content=entry.item.content,
            )


def _check_citations(
    entry: ExtractedItem,
    result: ExtractionResult,
    records: Mapping[str, object],
    texts: Mapping[str, str],
) -> Iterable[Finding]:
    if not entry.archive_record_ids:
        yield Finding(
            severity=Severity.ERROR,
            code="unattributed",
            detail="item cites no archive record",
            item_content=entry.item.content,
        )
        return

    for record_id in entry.archive_record_ids:
        if record_id not in records:
            yield Finding(
                severity=Severity.ERROR,
                code="dangling_citation",
                detail=f"cites {record_id!r}, which is not in this session",
                item_content=entry.item.content,
            )

    for seq in entry.archive_seqs:
        if not result.first_seq <= seq <= result.last_seq:
            yield Finding(
                severity=Severity.ERROR,
                code="out_of_range",
                detail=(
                    f"cites position {seq}, outside the extracted range "
                    f"{result.first_seq}-{result.last_seq}"
                ),
                item_content=entry.item.content,
            )

    supporting = " ".join(texts.get(rid, "") for rid in entry.archive_record_ids)
    if supporting and _overlap(entry.item.content, supporting) < MIN_OVERLAP:
        yield Finding(
            severity=Severity.WARNING,
            code="weak_grounding",
            detail=(
                "little of the item's wording appears in the records it cites; it may be "
                "a paraphrase, or it may not be supported"
            ),
            item_content=entry.item.content,
        )


def _check_shape(entry: ExtractedItem) -> Iterable[Finding]:
    item = entry.item
    if item.type is StateType.DECISION:
        rationale = getattr(item, "rationale", "")
        if not rationale.strip():
            yield Finding(
                severity=Severity.ERROR,
                code="decision_without_rationale",
                detail="a decision with no reason cannot be revisited",
                item_content=item.content,
            )


def _check_supersession(items: Sequence[ExtractedItem]) -> Iterable[Finding]:
    """Whether the superseded/active bookkeeping is coherent.

    An extraction that marked everything superseded, or that superseded an item
    it never recorded, has misunderstood the instruction rather than found an
    unusual conversation.
    """
    contents = {entry.item.content.strip().lower() for entry in items}
    for entry in items:
        target = str(entry.item.metadata.get("supersedes_content") or "").strip().lower()
        if target and target not in contents:
            yield Finding(
                severity=Severity.WARNING,
                code="supersedes_unknown_item",
                detail="claims to supersede an item that was not extracted",
                item_content=entry.item.content,
            )

    if items and all(entry.item.status.value == "superseded" for entry in items):
        yield Finding(
            severity=Severity.ERROR,
            code="everything_superseded",
            detail="every item is marked superseded, which leaves no current state at all",
        )


def _overlap(claim: str, source: str) -> float:
    """Fraction of the claim's distinctive words present in the source."""
    words = _distinctive(claim)
    if not words:
        return 1.0
    present = sum(1 for word in words if word in source)
    return present / len(words)


def _distinctive(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9._-]*", text.lower())
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 2}


def format_findings(findings: Sequence[Finding]) -> str:
    if not findings:
        return "no findings: nothing asserted is unsupported (completeness is not checked)"
    lines = []
    for finding in findings:
        lines.append(f"{finding.severity.value.upper():<8} {finding.code}: {finding.detail}")
        if finding.item_content:
            lines.append(f"         item: {finding.item_content[:120]}")
    return "\n".join(lines)


__all__ = [
    "MIN_OVERLAP",
    "Finding",
    "Severity",
    "format_findings",
    "validate",
]
