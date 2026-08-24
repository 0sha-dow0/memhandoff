"""What an import did.

An import that silently succeeds is not good enough. A conversation history is
the one thing this project cannot quietly damage, so every departure from a
clean read is counted and returned: records that would not parse, ids the source
never gave, timestamps it gave in a form that cannot be trusted, event kinds this
build has no name for.

**Counts are exact; samples are capped.** A malformed 200MB file would otherwise
produce a report as large as the problem it describes. Every capped field is
paired with a full count, and ``truncated_samples`` says plainly that the lists
are not the whole story, so a cap can never be mistaken for completeness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

SAMPLE_LIMIT = 20
"""How many examples each sampled field keeps. Counts are never capped."""

TEXT_SAMPLE_LENGTH = 200
"""How much of an unreadable record is kept, so a report cannot hold a whole file."""

BLANK_LINE = "blank"
"""``MalformedRecord.reason`` for a line holding no record at all."""


@dataclass(frozen=True)
class MalformedRecord:
    """A source record that could not be read.

    ``location`` is human-readable and specific, such as ``line 412``, because
    the only useful thing to say about a broken record is where to find it.

    ``reason`` is a short machine-readable code so the pipeline can treat
    categories differently without matching on prose. A blank line is reported
    through the same channel as a broken one: it is not a record and carries
    nothing, but a run of them is how a zeroed file tail looks, and a count of
    zero blank lines is worth more than an unexplained gap in the positions.
    """

    position: int
    location: str
    detail: str
    text: str = ""
    reason: str = "malformed"

    def __str__(self) -> str:
        return f"{self.location}: {self.detail}"


@dataclass(frozen=True)
class ImportReport:
    """The outcome of one import."""

    session_id: str
    provider: str

    source_key: str = ""
    """The logical source identity these events were imported under.

    Reported so a caller can see which identity was used, particularly when it
    was defaulted rather than passed. Not written into the archive; only its
    fingerprint appears there, inside synthetic record ids.
    """

    completed: bool = True
    """Whether the source was read to the end.

    ``False`` on the report attached to an ``IncompleteImportError``, where the
    counts below describe a partial import rather than a whole one. An import is
    not atomic and cannot be rolled back, so this flag and ``last_seq`` are how a
    caller learns what is actually in the archive.
    """

    events_read: int = 0
    events_written: int = 0
    first_seq: int | None = None
    last_seq: int | None = None
    appended_to_existing: bool = False

    by_type: Mapping[str, int] = field(default_factory=dict)

    without_source_id: int = 0
    without_timestamp: int = 0
    unusable_timestamps: int = 0
    unmatched_tool_results: int = 0
    tool_events_without_name: int = 0
    blank_lines_skipped: int = 0

    unknown_source_types: int = 0
    """Events this build has no name for, stored as ``EventType.OTHER``."""

    unknown_source_type_sample: tuple[str, ...] = ()
    """Distinct provider names behind those events, capped."""

    duplicate_source_ids: int = 0
    """Events whose source id had already been seen in this import."""

    duplicate_source_id_sample: tuple[str, ...] = ()
    """Distinct ids that repeated, capped."""

    malformed_records: int = 0
    malformed_sample: tuple[MalformedRecord, ...] = ()

    @property
    def clean(self) -> bool:
        """Whether the source read without a single reservation.

        Deliberately strict. A missing timestamp is not corruption, but it is
        something the caller should have to look at rather than discover later.
        A partial import is never clean, whatever the counts say.
        """
        return (
            self.completed
            and self.malformed_records == 0
            and self.duplicate_source_ids == 0
            and self.without_source_id == 0
            and self.without_timestamp == 0
            and self.unusable_timestamps == 0
            and self.unknown_source_types == 0
            and self.unmatched_tool_results == 0
        )

    @property
    def truncated_samples(self) -> bool:
        """Whether any sample hit the cap and may therefore be incomplete.

        Read this before treating a sample as an exhaustive list. The counts
        beside each sample are always exact.
        """
        return any(
            len(sample) >= SAMPLE_LIMIT
            for sample in (
                self.malformed_sample,
                self.duplicate_source_id_sample,
                self.unknown_source_type_sample,
            )
        )


__all__ = [
    "BLANK_LINE",
    "SAMPLE_LIMIT",
    "TEXT_SAMPLE_LENGTH",
    "ImportReport",
    "MalformedRecord",
]
