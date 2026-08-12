"""Mapping between models and rows.

Kept apart from the repository so the shape of the translation is readable on
its own, and so a schema change touches one file.

Two invariants hold throughout: an id is written exactly as the model generated
it, and a value that goes in comes back equal. Round-trip equality is asserted
in the tests for every model, because a storage layer that quietly alters a
record is worse than one that refuses to store it.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from open_context.models import (
    ContextSnapshot,
    Evidence,
    Message,
    Session,
    StateItemAdapter,
    TokenEstimate,
)
from open_context.models import ids as id_utils
from open_context.models.state import StateItem, StateItemBase
from open_context.storage.errors import SerializationError

_BASE_FIELDS = frozenset(StateItemBase.model_fields)


def dump_json(value: Any, *, field: str, record_id: str) -> str:  # noqa: ANN401
    """Serialise a JSON column, or fail with a message that names the field.

    ``metadata`` is typed ``dict[str, Any]``, so a value such as a set passes
    model validation and only fails here. The error says which field and which
    record rather than surfacing a bare TypeError from deep inside sqlite3.
    """
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise SerializationError(
            f"cannot store {field!r} of record {record_id!r} as JSON: {exc}"
        ) from exc


def load_json(raw: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(raw)
    return loaded


# --------------------------------------------------------------------------
# Session


def session_to_row(session: Session) -> tuple[Any, ...]:
    return (
        session.id,
        session.title,
        session.source,
        session.created_at.isoformat(),
        dump_json(session.metadata, field="metadata", record_id=session.id),
    )


def row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        source=row["source"],
        created_at=row["created_at"],
        metadata=load_json(row["metadata"]),
    )


# --------------------------------------------------------------------------
# Message


def message_to_row(message: Message) -> tuple[Any, ...]:
    return (
        message.id,
        message.session_id,
        message.seq,
        message.role.value,
        message.content,
        message.timestamp.isoformat(),
        message.trust.value,
        message.parent_id,
        message.tool_name,
        message.tool_call_id,
        dump_json(message.metadata, field="metadata", record_id=message.id),
    )


def row_to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        seq=row["seq"],
        role=row["role"],
        content=row["content"],
        timestamp=row["timestamp"],
        trust=row["trust"],
        parent_id=row["parent_id"],
        tool_name=row["tool_name"],
        tool_call_id=row["tool_call_id"],
        metadata=load_json(row["metadata"]),
    )


# --------------------------------------------------------------------------
# Evidence


def evidence_to_row(evidence: Evidence) -> tuple[Any, ...]:
    return (
        evidence.id,
        evidence.session_id,
        evidence.source_type.value,
        evidence.trust.value,
        evidence.content,
        evidence.content_hash,
        evidence.timestamp.isoformat(),
        dump_json(evidence.metadata, field="metadata", record_id=evidence.id),
    )


def row_to_evidence(row: sqlite3.Row, source_ids: list[str]) -> Evidence:
    return Evidence(
        id=row["id"],
        session_id=row["session_id"],
        source_ids=source_ids,
        source_type=row["source_type"],
        trust=row["trust"],
        content=row["content"],
        content_hash=row["content_hash"],
        timestamp=row["timestamp"],
        metadata=load_json(row["metadata"]),
    )


def split_evidence_source(reference: str) -> tuple[str | None, str | None]:
    """Route an evidence source to its column.

    Evidence cites a message, or an artifact or event state item. SQLite cannot
    aim one foreign key at two tables, so the reference goes into whichever
    column matches its prefix and the other stays null.
    """
    prefix = id_utils.prefix_of(reference)
    if prefix == id_utils.MESSAGE:
        return reference, None
    return None, reference


def split_state_source(reference: str) -> tuple[str | None, str | None]:
    """Route a state item's provenance reference to its column."""
    prefix = id_utils.prefix_of(reference)
    if prefix == id_utils.MESSAGE:
        return reference, None
    return None, reference


# --------------------------------------------------------------------------
# State items


def state_attributes(item: StateItem) -> dict[str, Any]:
    """Subtype-specific fields, excluding everything shared with the base.

    ``blocked_by`` is stored as relation rows rather than in the JSON column, so
    the references are foreign keys the database can actually check.
    """
    extra = set(item.__class__.model_fields) - _BASE_FIELDS - {"type", "blocked_by"}
    return {name: getattr(item, name) for name in sorted(extra)}


def state_to_row(item: StateItem) -> tuple[Any, ...]:
    return (
        item.id,
        item.session_id,
        item.type.value,
        item.content,
        item.status.value,
        item.retention.value,
        item.importance,
        item.confidence,
        item.trust.value,
        item.created_at.isoformat(),
        item.supersedes,
        dump_json(
            json.loads(
                json.dumps(state_attributes(item), default=_json_default),
            ),
            field="attributes",
            record_id=item.id,
        ),
        dump_json(item.metadata, field="metadata", record_id=item.id),
    )


def _json_default(value: Any) -> Any:  # noqa: ANN401
    """Serialise the few non-primitive types a state attribute can hold."""
    if hasattr(value, "isoformat"):
        iso: str = value.isoformat()
        return iso
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"cannot serialise {type(value).__name__} in a state attribute")


def row_to_state(
    row: sqlite3.Row,
    sources: list[str],
    related_ids: list[str],
    blocked_by: list[str],
) -> StateItem:
    payload: dict[str, Any] = {
        "id": row["id"],
        "session_id": row["session_id"],
        "type": row["type"],
        "content": row["content"],
        "status": row["status"],
        "retention": row["retention"],
        "importance": row["importance"],
        "confidence": row["confidence"],
        "trust": row["trust"],
        "created_at": row["created_at"],
        "supersedes": row["supersedes"],
        "sources": sources,
        "related_ids": related_ids,
        "metadata": load_json(row["metadata"]),
        **load_json(row["attributes"]),
    }
    if row["type"] == "task":
        payload["blocked_by"] = blocked_by
    return StateItemAdapter.validate_python(payload)


# --------------------------------------------------------------------------
# Snapshots


def snapshot_to_row(snapshot: ContextSnapshot) -> tuple[Any, ...]:
    estimate = snapshot.token_estimate
    return (
        snapshot.id,
        snapshot.session_id,
        snapshot.parent_snapshot_id,
        snapshot.created_at.isoformat(),
        None if estimate is None else estimate.value,
        None if estimate is None else estimate.method,
        None if estimate is None else int(estimate.exact),
        snapshot.compiler_version,
        snapshot.reason,
        dump_json(snapshot.metadata, field="metadata", record_id=snapshot.id),
    )


def row_to_snapshot(
    row: sqlite3.Row,
    active_message_ids: list[str],
    archived_message_ids: list[str],
    state_item_ids: list[str],
) -> ContextSnapshot:
    estimate = (
        None
        if row["token_value"] is None
        else TokenEstimate(
            value=row["token_value"],
            method=row["token_method"],
            exact=bool(row["token_exact"]),
        )
    )
    return ContextSnapshot(
        id=row["id"],
        session_id=row["session_id"],
        parent_snapshot_id=row["parent_snapshot_id"],
        created_at=row["created_at"],
        active_message_ids=active_message_ids,
        archived_message_ids=archived_message_ids,
        state_item_ids=state_item_ids,
        token_estimate=estimate,
        compiler_version=row["compiler_version"],
        reason=row["reason"],
        metadata=load_json(row["metadata"]),
    )
