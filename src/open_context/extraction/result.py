"""What an extraction produced, and what it could not.

```
archive range -> LLM -> parsed items -> typed state records + a report
```

**An extraction reports its own losses.** A run that silently returned three
items from a conversation containing thirty decisions would look identical to
one that correctly found three. So every item the model emitted and the parser
refused is counted and sampled, and the reasons are kept.

**Nothing here writes to storage.** Extraction produces records; deciding
whether they belong in a database, and reconciling them with what is already
there, is Phase 7's problem. Keeping the two apart means an extraction can be
run, inspected, and thrown away without touching state.

## Provenance points at the archive, not at SQLite

`StateItemBase.sources` accepts `msg_` and `ev_` ids, which are SQLite's naming.
The archive names its records differently — a provider's own id, or a positional
`pos-` or `synth-` id the importer minted. Nothing has ever established a
correspondence between the two; phases.md records that as an open question, and
this is the phase that first needs an answer.

The answer taken here is to cite **what actually exists**. An extracted item
carries the archive record ids and sequence numbers it came from, on the wrapper
rather than inside the state record, and `sources` is left empty. Minting
`msg_`-shaped ids for archive records would assert a correspondence no code
maintains, and a provenance reference that resolves to nothing is worse than one
that is honestly out of band.

Turning archive coordinates into `Evidence` and `Message` rows is the job of
whichever phase first writes both stores. Extraction does not, so it does not
have to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from open_context.models.state import StateItem

MAX_SAMPLES = 10
"""How many rejected items to keep verbatim.

Enough to see the shape of what went wrong, bounded so a pathological run cannot
grow the report without limit. The count is exact regardless.
"""


@dataclass(frozen=True)
class RejectedItem:
    """One item the model produced that could not become a record."""

    reason: str
    raw: str
    """The item as the model emitted it, truncated. Kept so a failure can be
    read rather than guessed at."""


@dataclass
class ExtractionReport:
    """What happened, including what did not work.

    Counts are exact; samples are capped. A caller that needs every rejection
    should lower the temperature of its expectations rather than raise the cap:
    an extraction rejecting hundreds of items has a problem the samples already
    show.
    """

    items_emitted: int = 0
    """How many items the model claimed."""

    items_accepted: int = 0
    malformed_response: bool = False
    """The reply was not JSON of the expected shape at all."""

    rejected: list[RejectedItem] = field(default_factory=list)
    unattributed: int = 0
    """Items dropped for citing no source message.

    Counted separately because it is the failure most likely to be systematic —
    a model that ignores the provenance instruction fails every item the same
    way, and that is a prompt problem rather than a parsing one.
    """

    unknown_types: dict[str, int] = field(default_factory=dict)
    """Type strings the model invented, and how often."""

    llm_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def reject(self, reason: str, raw: str) -> None:
        if len(self.rejected) < MAX_SAMPLES:
            self.rejected.append(RejectedItem(reason=reason, raw=raw[:300]))

    @property
    def clean(self) -> bool:
        """Whether every item the model produced became a record."""
        return (
            not self.malformed_response
            and self.items_emitted == self.items_accepted
            and not self.unattributed
        )

    def describe(self) -> str:
        if self.malformed_response:
            return "extraction failed: the model's reply was not usable JSON"
        parts = [f"{self.items_accepted} of {self.items_emitted} items accepted"]
        if self.unattributed:
            parts.append(f"{self.unattributed} dropped for citing no source")
        if self.unknown_types:
            parts.append(f"unknown types {sorted(self.unknown_types)}")
        return "; ".join(parts)


@dataclass(frozen=True)
class ExtractedItem:
    """One state record, and where in the archive it came from.

    The provenance rides on the wrapper rather than inside the record because
    the record's own ``sources`` field speaks SQLite's id language and the
    archive does not speak it. Keeping them separate means neither has to lie:
    the record is a valid Phase 1 record, and the citation resolves to something
    a reader can actually open.
    """

    item: StateItem
    archive_record_ids: tuple[str, ...]
    """Record ids in the session log. Never empty — an item that cannot be
    attributed is rejected rather than stored unattributed."""

    archive_seqs: tuple[int, ...]
    """Positions of those records, so a reader can seek without a scan."""


@dataclass(frozen=True)
class ExtractionResult:
    """Typed state extracted from one range of a session.

    ``items`` are records, not proposals: each already validated against the
    Phase 1 model, each carrying the archive coordinates it came from. Whether
    they are *correct* is a separate question, and the reason Phase 6 ships a
    validator alongside the extractor.
    """

    session_id: str
    items: tuple[ExtractedItem, ...]
    report: ExtractionReport
    prompt_id: str
    prompt_hash: str
    provider: str
    model: str
    first_seq: int
    last_seq: int
    """The archive range this was extracted from, inclusive.

    Recorded so an incremental run can tell what has already been read, which is
    what Phase 7 will need and what a range-less result could not support.
    """

    def of_type(self, type_name: str) -> tuple[ExtractedItem, ...]:
        return tuple(entry for entry in self.items if entry.item.type.value == type_name)

    @property
    def records(self) -> tuple[StateItem, ...]:
        return tuple(entry.item for entry in self.items)


__all__ = [
    "MAX_SAMPLES",
    "ExtractedItem",
    "ExtractionReport",
    "ExtractionResult",
    "RejectedItem",
]
