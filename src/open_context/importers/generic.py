"""The generic JSON Lines importer.

One JSON object per line. This is the first adapter because it is the only shape
that can be read without knowing whose export it is, and because it streams: a
line is a record, so a source larger than memory imports in constant space.

**Why not a top-level JSON array.** ``json.load`` builds the entire document
before returning the first element, so a 500MB array costs 500MB of RAM plus the
parsed object graph on top. That breaks the constraint this phase is built
around. Reading an array incrementally needs a pull parser, which is a
dependency, and this project does not take one until a measurement says it must.
An array export converts with ``jq -c '.[]' export.json > export.jsonl``, which
costs one command and no architecture.

**Field names are conventions, not a schema.** The defaults below are ordinary
JSON naming, not any particular provider's transcript format. Nothing here was
inferred from a real Claude, Codex, or Gemini export, and nothing here should be
read as claiming to parse one. A source that names things differently is handled
by passing a ``FieldMap``, not by this file growing a branch per provider.

**Roles are the ones the data model already defines.** ``models.enums.Role`` and
nothing else, so this adapter cannot quietly invent a vocabulary that the rest of
the codebase then has to honour. A role outside that set is not an error and is
not dropped: the event becomes ``OTHER`` and keeps the original name in
``source_type``. Callers with a source that says ``human`` pass a ``roles``
mapping rather than editing this file.

**``role="tool"`` reads as a tool result, by convention only.** It is the
prevailing convention in JSON conversation formats, where a tool-role message
carries what a tool returned rather than a request to run one, and it is the
reading ``models.enums.Role.TOOL`` was defined against. It is a default for
sources whose format nobody has verified, not a fact about any provider.

A provider-specific adapter must classify tool events from that provider's
verified format and must not inherit this mapping by default. Providers differ:
one may put the call and its result in a single record, another may nest a call
inside an assistant message, another may use a role name this adapter has never
seen. Where a caller's source disagrees, the ``roles`` mapping overrides it
here; a provider adapter decides for itself. Neither route adds a branch to this
file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, TextIO

from open_context.importers.base import ReadResult
from open_context.importers.events import EventType, RawEvent
from open_context.importers.report import BLANK_LINE, TEXT_SAMPLE_LENGTH, MalformedRecord
from open_context.models.enums import Role

PROVIDER: Final = "generic-jsonl"

DEFAULT_ROLES: Final[Mapping[str, EventType]] = {
    Role.USER.value: EventType.USER_MESSAGE,
    Role.ASSISTANT.value: EventType.ASSISTANT_MESSAGE,
    Role.SYSTEM.value: EventType.SYSTEM_MESSAGE,
    Role.DEVELOPER.value: EventType.DEVELOPER_MESSAGE,
    # A generic convention, not a provider fact. See the module docstring: a
    # provider adapter classifies tool events from its own verified format.
    Role.TOOL.value: EventType.TOOL_RESULT,
}


@dataclass(frozen=True)
class FieldMap:
    """Which keys hold which value, in order of preference.

    The first key present on a record wins, so a source using more than one
    convention across its history still reads. An empty tuple disables a field
    entirely, which is how a caller says "this key means something else here".
    """

    id: Sequence[str] = ("id", "uuid", "event_id", "message_id")
    type: Sequence[str] = ("type", "event_type")
    role: Sequence[str] = ("role", "author", "sender")
    content: Sequence[str] = ("content", "text", "body")
    timestamp: Sequence[str] = ("timestamp", "created_at", "time")
    tool_name: Sequence[str] = ("tool_name", "tool")
    tool_call_id: Sequence[str] = ("tool_call_id", "tool_use_id", "call_id")
    parent_id: Sequence[str] = ("parent_id", "parent_uuid", "parent")


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> Any:  # noqa: ANN401  source values
    """The first present, non-null value among ``keys``."""
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _as_identifier(value: Any) -> str | None:  # noqa: ANN401  source values are unconstrained
    """Coerce a source identifier to a string, or give up.

    Strings pass through untouched. Integers are stringified, since a numeric id
    is common and unambiguous. Anything else is left alone rather than forced
    into a shape it does not have; the original stays readable in ``raw``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    return None


def parse_timestamp(value: Any) -> datetime | None:  # noqa: ANN401  source values
    """Read a source timestamp, or report that it is unusable.

    ISO 8601 with an offset, or epoch seconds. A naive ISO string is refused: it
    could be any timezone, and guessing is how an export produced in another
    timezone silently reorders history. An unusable timestamp becomes ``None``
    and is counted in the report; the original text survives in ``raw``, so
    nothing is lost and a later phase can decide what it means.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


class JsonLinesImporter:
    """Reads a conversation from one JSON object per line."""

    provider: Final = PROVIDER

    def __init__(
        self,
        fields: FieldMap | None = None,
        *,
        roles: Mapping[str, EventType] | None = None,
    ) -> None:
        self.fields = fields or FieldMap()
        self.roles = dict(DEFAULT_ROLES if roles is None else roles)

    # ------------------------------------------------------------------

    def detect(self, sample: str) -> bool:
        """Whether the sample opens with a JSON object on its own line."""
        for line in sample.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                return isinstance(json.loads(stripped), dict)
            except json.JSONDecodeError:
                # The first line may be cut mid-record by the sample boundary,
                # which looks like an object without closing. That is still a
                # JSON Lines source, so it is accepted on the opening brace.
                return stripped.startswith("{")
        return False

    def read(self, stream: TextIO) -> Iterator[ReadResult]:
        """Yield one result per line, in file order.

        Reads one line at a time and holds no history, so a source larger than
        memory imports in constant space. A truncated final line arrives here as
        invalid JSON and is reported at its own position, which is what makes a
        cut-off export locatable rather than merely broken.
        """
        for index, line in enumerate(stream):
            location = f"line {index + 1}"
            stripped = line.strip()
            if not stripped:
                yield MalformedRecord(index, location, "blank line", reason=BLANK_LINE)
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield MalformedRecord(
                    index,
                    location,
                    f"invalid JSON: {exc}",
                    stripped[:TEXT_SAMPLE_LENGTH],
                    reason="invalid_json",
                )
                continue
            if not isinstance(record, dict):
                yield MalformedRecord(
                    index,
                    location,
                    f"record is not a JSON object, got {type(record).__name__}",
                    stripped[:TEXT_SAMPLE_LENGTH],
                    reason="not_an_object",
                )
                continue
            yield self.normalize(record)

    # ------------------------------------------------------------------

    def normalize(self, record: Mapping[str, Any]) -> RawEvent:
        """Turn one source record into a normalized event.

        Never fails. Every field this adapter cannot read becomes ``None``, and
        the whole record is kept in ``raw`` regardless, so a source shape nobody
        anticipated still imports with everything recoverable.
        """
        event_type, source_type = self._classify(record)
        content = _first(record, self.fields.content)
        return RawEvent(
            type=event_type,
            provider=self.provider,
            raw=dict(record),
            source_id=_as_identifier(_first(record, self.fields.id)),
            source_type=source_type,
            timestamp=parse_timestamp(_first(record, self.fields.timestamp)),
            text=content if isinstance(content, str) else None,
            tool_name=_as_identifier(_first(record, self.fields.tool_name)),
            tool_call_id=_as_identifier(_first(record, self.fields.tool_call_id)),
            parent_id=_as_identifier(_first(record, self.fields.parent_id)),
        )

    def _classify(self, record: Mapping[str, Any]) -> tuple[EventType, str | None]:
        """Decide what kind of event this is, and what the source called it.

        An explicit event type wins over a role, because a source that names both
        is telling us the role is the author and the type is the event. Where
        neither is recognised the event is ``OTHER`` and the source's own word is
        preserved, so nothing is classified into silence.
        """
        declared = _first(record, self.fields.type)
        role = _first(record, self.fields.role)
        source_type = declared if isinstance(declared, str) else None
        if source_type is None and isinstance(role, str):
            source_type = role

        if isinstance(declared, str):
            try:
                return EventType(declared), source_type
            except ValueError:
                pass
        for candidate in (declared, role):
            if isinstance(candidate, str) and candidate in self.roles:
                return self.roles[candidate], source_type
        return EventType.OTHER, source_type


"""Importers tried in order when a caller does not name one."""


__all__ = [
    "DEFAULT_ROLES",
    "PROVIDER",
    "FieldMap",
    "JsonLinesImporter",
    "parse_timestamp",
]
