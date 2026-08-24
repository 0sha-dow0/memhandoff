"""Turning a conversation into typed work state.

```
session log -> rendered window -> LLM -> JSON -> validated records
```

**This is the first thing in the project that interprets.** Import asks what
happened; compaction asks what to keep; extraction asks what it *means* — that
this sentence is a decision, that one a prohibition, and that the second
overturns the first. Everything before it could be checked against the source
mechanically. This cannot, which is why it reports its own losses and ships a
validator.

**Nothing is trusted.** The model is asked for JSON and expected to get it
wrong: unknown types, absent rationales, invented fields, prose wrapped around
the object, and items citing no source at all are each handled and counted
rather than crashing or silently vanishing. An item that cannot become a valid
record does not become one.

**The archive is never modified.** Extraction reads through the archive's own
range interface and writes nothing anywhere.

**One call means the whole conversation is in memory and in the prompt.** That
is a real ceiling: a session larger than the model's context window cannot be
extracted this way, and the project's rule that a large archive must not need
proportional RAM is not met here. Chunking it needs supersession and conflict
resolution to reconcile what the chunks each say, which is Phase 7's, so the
limit is recorded rather than worked around badly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from open_context.archive import ArchiveRecord, SessionLog
from open_context.compaction import Prompt, render_events
from open_context.extraction.prompts import CURRENT_STATE_HEADER, EXTRACTION_PROMPT_V1
from open_context.extraction.result import (
    ExtractedItem,
    ExtractionReport,
    ExtractionResult,
)
from open_context.importers import RawEvent
from open_context.llm import ChatMessage, GenerationRequest, LLMProvider
from open_context.models.enums import StateStatus, StateType, TaskStatus
from open_context.models.state import (
    Constraint,
    Decision,
    Entity,
    Fact,
    Goal,
    OpenQuestion,
    StateItem,
    Task,
)

_BUILDERS: dict[StateType, Any] = {
    StateType.GOAL: Goal,
    StateType.CONSTRAINT: Constraint,
    StateType.FACT: Fact,
    StateType.DECISION: Decision,
    StateType.TASK: Task,
    StateType.OPEN_QUESTION: OpenQuestion,
    StateType.ENTITY: Entity,
}
"""The types extraction may produce.

Deliberately a subset. ``Preference``, ``Event``, and ``Artifact`` exist in the
model and are not extracted, because nothing downstream consumes them yet and a
type the prompt names is a type the model will produce whether or not it means
anything.
"""


@dataclass(frozen=True)
class ExtractionConfig:
    """Knobs that change what an extraction produces."""

    prompt: Prompt = EXTRACTION_PROMPT_V1
    max_output_tokens: int = 900
    """How much JSON the model may return.

    **Requested output counts against a per-minute token allowance**, not just
    what is actually generated, so a generous ceiling is charged whether or not
    it is used. An earlier 2,000 helped push a single adversarial request to
    6,395 tokens against a 6,000/minute limit — a wall no amount of pacing can
    move, because one request exceeded the whole minute.

    900 holds roughly twenty items with rationales, which is more than any
    scenario has produced. A truncated reply is a recorded failure rather than a
    silent partial extraction, so running out is visible if it happens.
    """
    temperature: float = 0.0
    """Zero, because extraction should be as reproducible as the model allows.

    It does not make it deterministic — nothing here can — but a benchmark that
    re-extracts the same conversation should not be measuring sampling noise.
    """


class StructuredExtractor:
    """Extracts typed state from a session log through ``LLMProvider``.

    One model call per extraction. Chunking a long session into several calls,
    and reconciling what they each produce, is Phase 7's problem: it needs the
    supersession and conflict machinery that does not exist yet, and doing it
    here would mean doing it twice.
    """

    version = 1

    def __init__(
        self,
        provider: LLMProvider,
        config: ExtractionConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or ExtractionConfig()

    def extract(
        self,
        log: SessionLog,
        *,
        session_id: str | None = None,
        known_state: Sequence[StateItem] = (),
    ) -> ExtractionResult:
        """Read a whole session and return the state it arrived at.

        The session id is the archive's unless overridden, and is what every
        produced record is scoped to. A record scoped to the wrong session would
        be rejected by storage later, far from the cause.
        """
        return self.extract_range(log, session_id=session_id, known_state=known_state)

    def extract_range(
        self,
        log: SessionLog,
        *,
        start: int = 0,
        stop: int | None = None,
        session_id: str | None = None,
        known_state: Sequence[StateItem] = (),
    ) -> ExtractionResult:
        """The same, over part of a session.

        **What makes an incremental update possible.** Re-extracting a long
        session on every change would send the whole conversation to a model
        again, which is the expense this project exists to avoid.

        The range is the archive's own, half-open at ``stop``. The result
        records the positions actually read, so a caller can tell what a
        watermark should advance to without assuming the range it asked for was
        the range it got — a session that grew during the call, or one shorter
        than expected, would make that assumption wrong.
        """
        records = tuple(log.range(start, stop))
        scoped = session_id or log.session_id
        report = ExtractionReport()

        if not records:
            report.warnings.append("the session is empty; nothing to extract")
            return self._empty(scoped, report, first_seq=0, last_seq=0)

        request = GenerationRequest.of(
            ChatMessage.system(self.config.prompt.text),
            ChatMessage.user(self._render_state(known_state) + self._render(records)),
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
        )

        started = time.perf_counter()
        answer = self.provider.generate(request)
        report.latency_seconds = time.perf_counter() - started
        report.llm_calls = 1
        if answer.usage is not None:
            report.input_tokens = answer.usage.input_tokens
            report.output_tokens = answer.usage.output_tokens

        known = {record.id: record for record in records}
        items = self._parse(answer.text, scoped, known, report)

        info = self.provider.model_info()
        return ExtractionResult(
            session_id=scoped,
            items=items,
            report=report,
            prompt_id=self.config.prompt.identifier,
            prompt_hash=self.config.prompt.content_hash,
            provider=info.provider,
            model=info.model,
            first_seq=records[0].seq,
            last_seq=records[-1].seq,
        )

    # ------------------------------------------------------------------

    def _render(self, records: Sequence[ArchiveRecord]) -> str:
        """The conversation, with every record's id visible to the model.

        **The ids are the point.** The prompt demands provenance, and a model
        cannot cite what it was never shown. Rendering without them and then
        rejecting every item for citing nothing would be blaming the model for
        the harness's omission.
        """
        lines = []
        for record in records:
            rendered = render_events((RawEvent.from_payload(record.payload),))
            lines.append(f"[{record.id}] {rendered}")
        return "\n".join(lines)

    def _render_state(self, known_state: Sequence[StateItem]) -> str:
        """Existing state, so a new range can say what it overturns.

        Content is reproduced verbatim because supersession is matched
        literally: a model shown a paraphrase would emit a paraphrase, and the
        claim would resolve to nothing.
        """
        if not known_state:
            return ""
        lines = [CURRENT_STATE_HEADER]
        for item in known_state:
            lines.append(f"- ({item.type.value}) {item.content}")
        lines.append("\nNEW MESSAGES:\n")
        return "\n".join(lines)

    def _empty(
        self, session_id: str, report: ExtractionReport, *, first_seq: int, last_seq: int
    ) -> ExtractionResult:
        info = self.provider.model_info()
        return ExtractionResult(
            session_id=session_id,
            items=(),
            report=report,
            prompt_id=self.config.prompt.identifier,
            prompt_hash=self.config.prompt.content_hash,
            provider=info.provider,
            model=info.model,
            first_seq=first_seq,
            last_seq=last_seq,
        )

    def _parse(
        self,
        text: str,
        session_id: str,
        known: Mapping[str, ArchiveRecord],
        report: ExtractionReport,
    ) -> tuple[ExtractedItem, ...]:
        payload = parse_json_object(text)
        if payload is None or not isinstance(payload.get("items"), list):
            report.malformed_response = True
            report.warnings.append("the model's reply was not a JSON object with an items list")
            return ()

        raw_items = payload["items"]
        report.items_emitted = len(raw_items)

        extracted: list[ExtractedItem] = []
        for raw in raw_items:
            entry = self._build(raw, session_id, known, report)
            if entry is not None:
                extracted.append(entry)
        report.items_accepted = len(extracted)
        return tuple(extracted)

    def _build(
        self,
        raw: object,
        session_id: str,
        known: Mapping[str, ArchiveRecord],
        report: ExtractionReport,
    ) -> ExtractedItem | None:
        """One item, or nothing and a recorded reason."""
        if not isinstance(raw, dict):
            report.reject("item is not an object", repr(raw))
            return None

        content = str(raw.get("content") or "").strip()
        if not content:
            report.reject("item has no content", json.dumps(raw)[:300])
            return None

        type_name = str(raw.get("type") or "").strip().lower()
        try:
            state_type = StateType(type_name)
        except ValueError:
            report.unknown_types[type_name or "<missing>"] = (
                report.unknown_types.get(type_name or "<missing>", 0) + 1
            )
            report.reject(f"unknown type {type_name!r}", json.dumps(raw)[:300])
            return None

        builder = _BUILDERS.get(state_type)
        if builder is None:
            report.unknown_types[type_name] = report.unknown_types.get(type_name, 0) + 1
            report.reject(f"type {type_name!r} is not extracted", json.dumps(raw)[:300])
            return None

        cited = [str(ref) for ref in raw.get("source_ids") or [] if str(ref) in known]
        if not cited:
            report.unattributed += 1
            report.reject(
                "no source id resolved to a record in this session", json.dumps(raw)[:300]
            )
            return None

        fields: dict[str, Any] = {
            "session_id": session_id,
            "content": content,
            "status": _status(raw.get("status")),
            "confidence": _unit(raw.get("confidence")),
        }

        # The model is asked which item this replaces by *content*, not by id:
        # ids are minted here, after the model has answered, so it could not cite
        # one. Resolving content to an id is supersession bookkeeping, which is
        # Phase 7's, so the claim is carried verbatim until there is something to
        # resolve it against.
        supersedes = str(raw.get("supersedes_content") or "").strip()
        if supersedes:
            fields["metadata"] = {"supersedes_content": supersedes}
        if state_type is StateType.DECISION:
            rationale = str(raw.get("rationale") or "").strip()
            if not rationale:
                report.reject("decision has no rationale", json.dumps(raw)[:300])
                return None
            fields["rationale"] = rationale
            fields["alternatives"] = [str(v) for v in raw.get("alternatives") or []]
        elif state_type is StateType.CONSTRAINT:
            fields["hard"] = bool(raw.get("hard", True))
        elif state_type is StateType.TASK:
            fields["task_status"] = _task_status(raw.get("task_status"))
        elif state_type is StateType.ENTITY:
            # `Entity` requires a name and the prompt does not ask for one, so
            # without this every extracted entity was rejected as an invalid
            # record — silently, since a rejection is a counted outcome rather
            # than an error. Falling back to the content keeps the item; a
            # separate `name` is an improvement, not a precondition.
            fields["name"] = str(raw.get("name") or "").strip() or content

        try:
            item: StateItem = builder(**fields)
        except (ValueError, TypeError) as exc:
            report.reject(f"record rejected by the model: {exc}", json.dumps(raw)[:300])
            return None

        return ExtractedItem(
            item=item,
            archive_record_ids=tuple(cited),
            archive_seqs=tuple(known[ref].seq for ref in cited),
        )


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in the reply, or nothing.

    Models wrap JSON in prose and markdown fences however firmly they are asked
    not to, so a strict ``json.loads`` of the whole reply would fail on output
    that is perfectly usable. This finds the outermost braces and parses that.

    It does not repair anything. Truncated or genuinely malformed JSON returns
    ``None`` and becomes a recorded failure, because guessing at what a
    half-written object meant is how an extraction invents state.
    """
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _status(value: object) -> StateStatus:
    try:
        return StateStatus(str(value).strip().lower())
    except ValueError:
        return StateStatus.ACTIVE


def _task_status(value: object) -> TaskStatus:
    try:
        return TaskStatus(str(value).strip().lower())
    except ValueError:
        return TaskStatus.OPEN


def _unit(value: object) -> float:
    """A confidence in range, defaulting to the model's stated uncertainty.

    Out-of-range and unparseable values become 0.5 rather than raising: a model
    writing ``"high"`` where a float was asked for has said something useful
    about the item, and discarding the item over its confidence field would lose
    more than it protects.

    ``bool`` is excluded deliberately. It is a subclass of ``int``, so ``True``
    would otherwise arrive as a confidence of 1.0 — the most confident value in
    the range, produced by a model that answered the wrong question.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return 0.5
    try:
        number = float(value)
    except ValueError:
        return 0.5
    return min(1.0, max(0.0, number))


__all__ = [
    "ExtractionConfig",
    "StructuredExtractor",
    "parse_json_object",
]
