"""One session in, one portable package out.

The workflow Phase 11 exists to make real:

    Agent A  ->  runtime  ->  project.ctx  ->  Agent B  ->  continue the work

**This is orchestration, not logic.** It reads a transcript with an importer,
extracts state with the extractor, and packages the result with the builder —
each of which already exists and none of which is reimplemented here. The
roadmap's rule for adapters is that they receive context, invoke the runtime,
and export or compile; a handoff that grew its own idea of what to keep would be
a fourth compactor nobody asked for.

**The archive is written before anything is interpreted.** Extraction can fail,
a model can be rate-limited, a package can turn out empty — and when any of that
happens the conversation is already on disk, losslessly, and the expensive part
can be retried without re-reading the source. A pipeline that interpreted first
would lose the input to a transient failure.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_context.archive import Archive
from open_context.archive.records import FORMAT as ARCHIVE_FORMAT
from open_context.archive.records import FORMAT_VERSION as ARCHIVE_FORMAT_VERSION
from open_context.archive.records import ArchiveRecord
from open_context.importers.claude_code import (
    CONVERSATION_PARENT,
    ClaudeCodeImporter,
    active_thread,
    segments,
)
from open_context.importers.events import RawEvent
from open_context.importers.pipeline import MalformedPolicy, detect_importer, import_path
from open_context.importers.report import ImportReport, MalformedRecord
from open_context.llm.errors import LLMError
from open_context.llm.provider import LLMProvider
from open_context.models.state import StateItem
from open_context.package.builder import PackageBuildReport, build
from open_context.package.format import ArchiveReference, CtxPackage, RecentContext

RECENT_TURNS = 12
MAX_TURN_CHARACTERS = 1_200
"""How much of one recent turn is carried.

A recent window exists to show *what was just happening*, which needs several
turns. Without a cap one long answer takes the whole section — measured on a real
handoff, a single assistant message spent 476 of a 500-token budget and pushed
ten other turns out. Breadth is what the section is for.

Truncation is marked, never silent, so a reader can tell a turn that was cut from
one that ended.
"""
"""Verbatim turns carried into the package.

Enough for a reader to see what was just happening, few enough that recent
context does not become a second transcript. The compiler budgets it again
anyway, so this is a ceiling rather than a promise.
"""


@dataclass
class HandoffReport:
    """What the handoff did, and what it could not do."""

    session_id: str
    events_read: int = 0
    events_kept: int = 0
    events_abandoned: int = 0
    segment_sizes: list[int] = field(default_factory=list)
    state_items: int = 0
    malformed: int = 0
    warnings: list[str] = field(default_factory=list)
    carried_from_project: int = 0
    """State brought in from other sessions in the same project.

    Reported because a receiving agent shown a constraint should be able to find
    out that it came from a different session — silently mixing two sessions'
    conclusions would make provenance a lie at exactly the level people trust it.
    """

    extracted: bool = False
    """Whether a model was asked for state, regardless of what it returned.

    Distinct from ``state_items == 0``: "nobody asked" and "asked and got
    nothing" are different facts, and only the second says anything about the
    conversation.
    """

    import_report: ImportReport | None = None
    package_report: PackageBuildReport | None = None

    @property
    def compacted(self) -> bool:
        """Whether the source session had already been compacted at least once.

        Not certain, and the wording says so: a break in the conversation chain
        is what compaction looks like, and also what an injected command looks
        like. See ``importers.claude_code.segments``.
        """
        return len(self.segment_sizes) > 1

    def summary(self) -> str:
        lines = [
            f"session {self.session_id}",
            f"  read      {self.events_read} events",
            f"  kept      {self.events_kept} ({self.events_abandoned} abandoned as rewound)",
        ]
        if len(self.segment_sizes) > 1:
            lines.append(f"  segments  {self.segment_sizes} (the session was broken and resumed)")
        lines.append(f"  state     {self.state_items} items")
        if self.malformed:
            lines.append(f"  malformed {self.malformed} records, located in the import report")
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


SAMPLE_LINES = 4
SAMPLE_LIMIT = 2_000_000
"""How much of a file is read to work out what it is.

**Whole lines, not a byte slice.** One JSON Lines record can be large — 1.36 MB
in a measured transcript — so a fixed-size prefix routinely cuts the first record
in half, and a format check that sees invalid JSON concludes the file is not its
format. That handed every such session to the generic reader, which accepts
anything shaped like JSON and would have imported it as anonymous records.

The byte ceiling is there so a pathological single-line file cannot make
detection read the whole thing.
"""


def _sample(path: Path) -> str:
    lines: list[str] = []
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            lines.append(line)
            seen += len(line)
            if len(lines) >= SAMPLE_LINES or seen >= SAMPLE_LIMIT:
                break
    return "".join(lines)


def read_session(
    path: str | Path, *, keep_raw: bool = True
) -> tuple[list[RawEvent], list[MalformedRecord]]:
    """Every event in a transcript, and every record that could not be read.

    The importer is chosen by content rather than by extension, so a session
    file and a generic export both work and neither has to be declared.

    ``keep_raw=False`` drops the verbatim provider record from each event. The
    whole file must be held to resolve branches, so this is where a large
    session actually costs memory — 50 MB of parsed records on one measured
    transcript — and a caller that archives by re-reading the file never looks
    at them.
    """
    source = Path(path)
    importer = detect_importer(_sample(source))
    if not keep_raw and isinstance(importer, ClaudeCodeImporter):
        importer = ClaudeCodeImporter(include_thinking=importer.include_thinking, keep_raw=False)

    events: list[RawEvent] = []
    malformed: list[MalformedRecord] = []
    with source.open(encoding="utf-8") as handle:
        for result in importer.read(handle):
            if isinstance(result, MalformedRecord):
                malformed.append(result)
            else:
                events.append(result)
    return events, malformed


def archive_session(
    archive_root: str | Path,
    session_id: str,
    path: str | Path,
    *,
    on_malformed: MalformedPolicy = "skip",
) -> ImportReport:
    """Put the raw conversation on disk before anything interprets it.

    Malformed records are skipped rather than fatal. The default elsewhere is to
    abort, which is right for an export a user chose to import; a live session
    file may have a torn tail simply because the agent that writes it is still
    running, and refusing the whole conversation over its last line would make
    handoff impossible exactly when it is most wanted. The records are counted
    and located in the report.
    """
    return import_path(Archive(Path(archive_root)), session_id, path, on_malformed=on_malformed)


@dataclass
class SessionSurvey:
    """What two cheap passes over a transcript establish.

    **Deliberately not the conversation.** Resolving branches needs every turn's
    identity and nothing else, so the first pass keeps ids and drops text. The
    second keeps only the handful of turns that will actually be carried.

    Holding the whole conversation instead costs memory in proportion to the
    transcript — measured at 82 MB on a 25 MB session, most of it tool output
    that nothing downstream reads.
    """

    events_read: int = 0
    events_kept: int = 0
    malformed: int = 0
    segment_sizes: list[int] = field(default_factory=list)
    tail: list[RawEvent] = field(default_factory=list)
    working_directory: str = ""
    """Where the session ran, which is what places it in a project."""
    source: str = ""
    """The importer that actually read the transcript."""


def survey_session(source: str | Path, *, tail: int = RECENT_TURNS) -> SessionSurvey:
    """Read a transcript twice, holding only what each pass needs."""
    path = Path(source)
    importer = detect_importer(_sample(path))
    claude_code = isinstance(importer, ClaudeCodeImporter)
    if isinstance(importer, ClaudeCodeImporter):
        importer = ClaudeCodeImporter(include_thinking=importer.include_thinking, keep_raw=False)

    # Pass one: identity only.
    skeleton: list[RawEvent] = []
    survey = SessionSurvey(source=importer.provider)
    with path.open(encoding="utf-8") as handle:
        for result in importer.read(handle):
            if isinstance(result, MalformedRecord):
                survey.malformed += 1
                continue
            survey.events_read += 1
            if not survey.working_directory:
                survey.working_directory = str(result.metadata.get("cwd") or "")
            skeleton.append(
                RawEvent(
                    type=result.type,
                    provider=result.provider,
                    source_id=result.source_id,
                    parent_id=result.parent_id,
                    metadata={
                        CONVERSATION_PARENT: result.metadata.get(CONVERSATION_PARENT),
                    },
                )
            )

    # Only Claude Code has the measured rewind and post-compaction tree
    # semantics implemented by ``active_thread``. Generic JSONL preserves its
    # source order; treating an absent Claude-only conversation ancestor as a
    # compaction break made every ordinary generic message look like a resumed
    # segment.
    live = active_thread(skeleton) if claude_code else list(skeleton)
    survey.events_kept = len(live)
    survey.segment_sizes = (
        [len(part) for part in segments(skeleton)]
        if claude_code
        else ([len(skeleton)] if skeleton else [])
    )
    keep = {event.source_id for event in live if event.source_id}
    del skeleton, live

    # Pass two: the last few spoken turns, and nothing else.
    spoken: list[RawEvent] = []
    with path.open(encoding="utf-8") as handle:
        for result in importer.read(handle):
            if isinstance(result, MalformedRecord):
                continue
            if result.source_id is not None and result.source_id not in keep:
                continue
            if not result.type.value.endswith("_message"):
                continue
            if not (result.text or "").strip():
                continue
            spoken.append(result)
            if len(spoken) > tail:
                spoken.pop(0)
    survey.tail = spoken
    return survey


def project_root_of(archive_root: str | Path, session_id: str) -> str:
    """The project a stored session belongs to, preserved across updates."""
    from open_context.incremental_store import load

    return load(archive_root, session_id).project_root


def _carry_forward(archive_root: str | Path, session_id: str, directory: str) -> list[StateItem]:
    """Project-level state established by other sessions in the same project.

    Reads the stored state of sibling sessions — the files V2.2 already writes —
    and returns only what `hierarchy.place` calls project-scoped. Nothing new is
    stored, and no model is called: this is selection over what is already known.
    """
    from open_context.hierarchy import Project, project_id
    from open_context.incremental_store import load, path_for

    root = Path(archive_root)
    directory_of_this = project_id(directory)

    state_dir = path_for(root, session_id).parent
    if not state_dir.is_dir():
        return []

    project = Project(root=directory_of_this)
    for stored_file in sorted(state_dir.glob("*.json")):
        stored = load(root, stored_file.stem)
        if stored.fresh or stored.session_id == session_id:
            continue
        if stored.project_root != directory_of_this:
            # A different project's conclusions are not this project's context.
            # Without the stored root there is nothing to compare, which is how
            # an unrelated constraint reached a package before this check.
            continue
        project.add(stored.session_id, stored.items)

    return project.carried_forward(exclude=session_id)


def _describe_archive(
    root: Path, session_id: str, *, wanted: set[str]
) -> tuple[ArchiveReference | None, list[ArchiveRecord]]:
    """Summarise an archive in one streaming pass, keeping only cited records.

    A package needs two things from an archive: its shape, and the few records
    its state actually points at. Reading it into a list gives both and costs
    memory in proportion to the conversation, which is precisely what the
    archive was built to avoid.
    """
    archive = Archive(root)
    if not archive.exists(session_id):
        return None, []

    first: int | None = None
    last = 0
    count = 0
    cited: list[ArchiveRecord] = []

    with archive.open(session_id) as log:
        for record in log:
            if first is None:
                first = record.seq
            last = record.seq
            count += 1
            if record.id in wanted:
                cited.append(record)

    if first is None:
        return None, []

    return (
        ArchiveReference(
            archive_format=ARCHIVE_FORMAT,
            archive_format_version=ARCHIVE_FORMAT_VERSION,
            session_id=session_id,
            first_seq=first,
            last_seq=last,
            record_count=count,
            location=str(root),
            record_hashes={record.seq: record.hash for record in cited},
        ),
        cited,
    )


def recent_context(events: Sequence[RawEvent], limit: int = RECENT_TURNS) -> RecentContext:
    """The last few turns, verbatim, with tool traffic left out.

    Tool calls and their results are the bulk of a coding session — 3,082 of
    8,890 records in the transcripts measured — and a recent window full of
    shell invocations tells a receiving agent much less than the same budget
    spent on what was said.
    """
    spoken = [
        event
        for event in events
        if event.type.value.endswith("_message") and (event.text or "").strip()
    ]
    tail = spoken[-limit:]
    return RecentContext(
        messages=tuple(
            {
                "role": event.type.value.removesuffix("_message"),
                "content": _clip(event.text or ""),
            }
            for event in tail
        ),
        reason=(
            f"the last {len(tail)} spoken turns, tool traffic excluded, each turn "
            f"clipped to {MAX_TURN_CHARACTERS} characters"
        ),
    )


def _clip(text: str) -> str:
    """Cut a long turn, and say that it was cut."""
    if len(text) <= MAX_TURN_CHARACTERS:
        return text
    return text[:MAX_TURN_CHARACTERS].rstrip() + " […truncated]"


def extract_state(
    archive_root: str | Path,
    session_id: str,
    provider: LLMProvider,
    *,
    incremental: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[StateItem], list[str], dict[str, list[int]]]:
    """Read the archived session with a model, and say where each item came from.

    Separate from ``handoff`` because it is the only step that needs a model, a
    credential, and a network. Everything else works offline, and a caller who
    cannot reach a provider still gets a package.

    **Incremental by default.** A watermark and the state it earned are kept
    beside the archive, so a second run over a growing session reads only what
    arrived since the first. Phase 7 built that machinery and nothing called it,
    so every package paid for the whole conversation again.

    Falls back to a full read whenever nothing is stored yet, which is the first
    run on any session — so the saving is on later runs, where it belongs.

    Extraction failure is reported, never fatal: the conversation is already
    archived, so the expensive step can be retried without re-reading anything.
    The watermark does not advance on failure, so the unread range is read next
    time rather than silently skipped.
    """
    from open_context.extraction.extractor import StructuredExtractor
    from open_context.extraction.windowed import extract_windowed
    from open_context.incremental_store import StoredState, load, save
    from open_context.state import extract_incremental, reconcile

    archive = Archive(Path(archive_root))
    if not archive.exists(session_id):
        return [], [f"no archived session {session_id!r} to extract from"], {}

    stored = load(archive_root, session_id) if incremental else StoredState(session_id=session_id)
    extractor = StructuredExtractor(provider)
    warnings: list[str] = []
    provenance: dict[str, list[int]] = {}

    try:
        with archive.open(session_id) as log:
            if stored.fresh:
                # Windowed, because a real session does not fit in one request:
                # the first live run offered 19,178 tokens to a model with an
                # 8,000-per-minute ceiling and was refused outright.
                windowed = extract_windowed(extractor, log, session_id=session_id, sleep=sleep)
                items = list(windowed.items)
                provenance = dict(windowed.provenance)
                read_to = len(log)
                warnings.extend(windowed.warnings)
                if not windowed.complete:
                    # The watermark must not claim a range that was not read.
                    read_to = 0
                result = None
            else:
                update = extract_incremental(
                    extractor, log, stored.watermark, known_state=stored.items
                )
                if update.result is None:
                    # Phase 7's contract: nothing new calls no model and returns
                    # None rather than an empty result, so "we did not look" and
                    # "we looked and found nothing" stay distinguishable.
                    return stored.items, ["nothing new since the last extraction"], {}

                result = update.result
                plan = reconcile(stored.items, result.items)
                items = [*stored.items, *(entry.item for entry in plan.added)]
                provenance = {
                    entry.item.id: list(entry.archive_seqs)
                    for entry in plan.added
                    if entry.archive_seqs
                }
                read_to = update.watermark.next_seq
                if plan.conflicts:
                    warnings.append(
                        f"{len(plan.conflicts)} conflicts between new and existing state were "
                        f"left unresolved; the new items are not in the package"
                    )
                    items = stored.items
    except LLMError as exc:
        return stored.items, [f"extraction failed ({type(exc).__name__}): {exc}"], {}

    if result is not None and result.report.rejected:
        warnings.append(
            f"the model returned {len(result.report.rejected)} items that were rejected; "
            f"they are not in the package. First reason: "
            f"{result.report.rejected[0].reason}"
        )

    if incremental:
        save(
            archive_root,
            StoredState(
                session_id=session_id,
                next_seq=read_to,
                items=items,
                project_root=project_root_of(archive_root, session_id),
            ),
        )
    return items, warnings, provenance


def handoff(
    *,
    source: str | Path,
    session_id: str,
    archive_root: str | Path,
    state: Sequence[Any] = (),
    provider: LLMProvider | None = None,
    carry_project_state: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    title: str | None = None,
    include_recent: bool = True,
) -> tuple[CtxPackage, HandoffReport]:
    """Read a session, archive it, and package what it means.

    ``state`` is passed in rather than extracted here. Extraction needs a model,
    a budget, and a decision about which one — none of which belongs to a
    function whose job is to move a conversation from one place to another. A
    caller with no state gets a package of provenance and recent context, which
    is a smaller claim honestly made rather than a larger one faked.
    """
    survey = survey_session(source)
    provenance: dict[str, list[int]] = {}
    report = HandoffReport(session_id=session_id, events_read=survey.events_read)
    report.malformed = survey.malformed
    report.events_kept = survey.events_kept
    report.events_abandoned = survey.events_read - survey.events_kept
    report.segment_sizes = survey.segment_sizes
    live = survey.tail

    report.import_report = archive_session(archive_root, session_id, source)

    if provider is not None and not state:
        # After archiving, so a failure here costs the model call and not the
        # conversation.
        state, extraction_warnings, extracted_provenance = extract_state(
            archive_root, session_id, provider, sleep=sleep
        )
        provenance.update(extracted_provenance)
        report.warnings.extend(extraction_warnings)
        report.extracted = True

    # Streamed, never listed. Holding a whole archive in memory to describe it is
    # the proportional cost the archive exists to avoid — measured at ~80 MB on a
    # 25 MB transcript before this read one record at a time.
    reference, cited = _describe_archive(
        Path(archive_root), session_id, wanted={s for item in state for s in item.sources}
    )

    carried: list[StateItem] = []
    if carry_project_state and survey.working_directory:
        carried = _carry_forward(archive_root, session_id, survey.working_directory)
        if carried:
            report.carried_from_project = len(carried)

    package, package_report = build(
        session_id=session_id,
        state=state,
        inherited=carried,
        title=title,
        source=survey.source,
        created_by="open_context handoff",
        archive_records=cited or None,
        archive_reference=reference,
        archive_location=str(Path(archive_root)),
        provenance=dict(provenance) if provenance else None,
        recent=list(recent_context(live).messages) if include_recent else None,
        recent_reason="the last spoken turns, tool traffic excluded",
    )
    report.state_items = package_report.state_items
    report.package_report = package_report
    # Extended, not replaced: warnings raised earlier — a failed extraction, most
    # of all — would otherwise be dropped on the floor by the last step to speak.
    report.warnings.extend(package_report.warnings)

    if not state:
        if not report.extracted:
            report.warnings.append(
                "no state was extracted, so this package carries recent context and "
                "provenance but no goals, constraints, or decisions; pass a provider "
                "to extract it"
            )
        elif any("could not be read" in w or "extraction failed" in w for w in report.warnings):
            # The model was asked and mostly refused. Saying "no state was
            # extracted" here would read as "this conversation had none", which
            # is a claim about the session rather than about the quota.
            report.warnings.append(
                "the model was asked but most windows were refused, so this package "
                "carries no state. This is usually a per-minute rate limit rather "
                "than an empty conversation — waiting a minute and running the same "
                "command again resumes from where it stopped"
            )
        else:
            report.warnings.append(
                "the model was asked and found no goals, constraints, or decisions in this session"
            )
    return package, report


__all__ = [
    "RECENT_TURNS",
    "HandoffReport",
    "archive_session",
    "extract_state",
    "handoff",
    "read_session",
    "recent_context",
]
