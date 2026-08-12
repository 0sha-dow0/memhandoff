"""The storage repository.

Append and read. There is no ``update`` and no ``delete``.

**Why there is no update.** Records are frozen in the model, and the archive is
the source of truth. A change is a new record whose ``supersedes`` points at the
old one, so both survive and the history of a changing fact stays answerable.

**Why supersession is derived rather than written.** Marking an old item
``SUPERSEDED`` would mean editing a stored historical record, which is the exact
thing the architecture forbids. Instead the relationship is read from the edge:
an item is superseded when some other item's ``supersedes`` points at it. The
stored record keeps the status it was authored with, and ``is_superseded`` and
``current_state_items`` answer from the graph.

**Why references are checked in Python as well as by the database.** Foreign keys
are deferred to commit, so a violation surfaces as one opaque error at COMMIT
with no indication of which reference broke. Checking first produces an error
naming the record and the reference; the database constraint stays as a
backstop for anything the checks miss.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from open_context.models import ContextSnapshot, Evidence, Message, Session, StateStatus
from open_context.models.state import StateItem
from open_context.storage import rows
from open_context.storage.database import Database
from open_context.storage.errors import (
    CrossSessionReferenceError,
    DanglingReferenceError,
    DuplicateRecordError,
    RecordNotFoundError,
    SequenceConflictError,
    StorageError,
    SupersessionCycleError,
)


class Repository:
    """Append-only access to stored records."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def database(self) -> Database:
        return self._db

    # ------------------------------------------------------------------
    # Existence

    def _exists(self, table: str, record_id: str, connection: sqlite3.Connection) -> bool:
        row = connection.execute(
            f"SELECT 1 FROM {table} WHERE id = ? LIMIT 1", (record_id,)
        ).fetchone()
        return row is not None

    def _session_of(self, table: str, record_id: str, connection: sqlite3.Connection) -> str | None:
        column = "id" if table == "sessions" else "session_id"
        row = connection.execute(
            f"SELECT {column} AS session_id FROM {table} WHERE id = ? LIMIT 1",
            (record_id,),
        ).fetchone()
        return None if row is None else str(row["session_id"])

    def _require(
        self,
        table: str,
        reference: str,
        *,
        source_id: str,
        connection: sqlite3.Connection,
        session_id: str | None = None,
        pending: Mapping[str, str] | None = None,
    ) -> None:
        """Raise unless the reference resolves, in the same session.

        ``pending`` maps ids being written in this batch to their session, so a
        reference to a record that appears later in the same import is satisfied
        without weakening the session check.
        """
        pending = pending or {}
        if reference in pending:
            found = pending[reference]
        else:
            found = self._session_of(table, reference, connection) or ""
            if not found:
                raise DanglingReferenceError(source_id, reference)
        if session_id is not None and found != session_id:
            raise CrossSessionReferenceError(source_id, reference, session_id, found)

    def _require_absent(self, table: str, record_id: str, connection: sqlite3.Connection) -> None:
        if self._exists(table, record_id, connection):
            raise DuplicateRecordError(record_id)

    # ------------------------------------------------------------------
    # Sessions

    def add_session(self, session: Session) -> Session:
        with self._db.transaction() as connection:
            self._require_absent("sessions", session.id, connection)
            connection.execute(
                "INSERT INTO sessions (id, title, source, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                rows.session_to_row(session),
            )
        return session

    def get_session(self, session_id: str) -> Session:
        row = self._db.connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(session_id)
        return rows.row_to_session(row)

    def list_sessions(self) -> list[Session]:
        cursor = self._db.connection.execute("SELECT * FROM sessions ORDER BY created_at, id")
        return [rows.row_to_session(row) for row in cursor]

    # ------------------------------------------------------------------
    # Messages

    def add_message(self, message: Message) -> Message:
        self.add_messages([message])
        return message

    def add_messages(self, messages: Sequence[Message]) -> int:
        """Write messages atomically.

        A parent may appear later in the same batch, so intra-batch ids count as
        satisfying a reference. Either every message lands or none does.
        """
        if not messages:
            return 0
        pending = {message.id: message.session_id for message in messages}
        with self._db.transaction() as connection:
            for message in messages:
                self._require_absent("messages", message.id, connection)
                self._require(
                    "sessions", message.session_id, source_id=message.id, connection=connection
                )
                if message.parent_id is not None:
                    self._require(
                        "messages",
                        message.parent_id,
                        source_id=message.id,
                        connection=connection,
                        session_id=message.session_id,
                        pending=pending,
                    )
                self._check_seq_free(message, connection)
                connection.execute(
                    "INSERT INTO messages (id, session_id, seq, role, content, timestamp, "
                    "trust, parent_id, tool_name, tool_call_id, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows.message_to_row(message),
                )
        return len(messages)

    def _check_seq_free(self, message: Message, connection: sqlite3.Connection) -> None:
        taken = connection.execute(
            "SELECT 1 FROM messages WHERE session_id = ? AND seq = ? LIMIT 1",
            (message.session_id, message.seq),
        ).fetchone()
        if taken is not None:
            raise SequenceConflictError(message.session_id, message.seq)

    def get_message(self, message_id: str) -> Message:
        row = self._db.connection.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(message_id)
        return rows.row_to_message(row)

    def list_messages(self, session_id: str) -> list[Message]:
        """Messages in session order.

        Ordered by ``seq``, never by timestamp. Exports routinely give many
        messages the same second, and history order is the one thing the runtime
        cannot get wrong.
        """
        cursor = self._db.connection.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY seq", (session_id,)
        )
        return [rows.row_to_message(row) for row in cursor]

    def next_seq(self, session_id: str) -> int:
        """One past the highest used position in the session."""
        row = self._db.connection.execute(
            "SELECT MAX(seq) AS m FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return 0 if row["m"] is None else int(row["m"]) + 1

    # ------------------------------------------------------------------
    # Evidence

    def add_evidence(self, evidence: Evidence) -> Evidence:
        with self._db.transaction() as connection:
            self._require_absent("evidence", evidence.id, connection)
            self._require(
                "sessions", evidence.session_id, source_id=evidence.id, connection=connection
            )
            for reference in evidence.source_ids:
                message_id, _ = rows.split_evidence_source(reference)
                table = "messages" if message_id is not None else "state_items"
                self._require(
                    table,
                    reference,
                    source_id=evidence.id,
                    connection=connection,
                    session_id=evidence.session_id,
                )
            connection.execute(
                "INSERT INTO evidence (id, session_id, source_type, trust, content, "
                "content_hash, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows.evidence_to_row(evidence),
            )
            connection.executemany(
                "INSERT INTO evidence_sources (evidence_id, ordinal, message_id, state_item_id) "
                "VALUES (?, ?, ?, ?)",
                [
                    (evidence.id, ordinal, *rows.split_evidence_source(reference))
                    for ordinal, reference in enumerate(evidence.source_ids)
                ],
            )
        return evidence

    def get_evidence(self, evidence_id: str) -> Evidence:
        row = self._db.connection.execute(
            "SELECT * FROM evidence WHERE id = ?", (evidence_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(evidence_id)
        return rows.row_to_evidence(row, self._evidence_sources(evidence_id))

    def _evidence_sources(self, evidence_id: str) -> list[str]:
        cursor = self._db.connection.execute(
            "SELECT message_id, state_item_id FROM evidence_sources "
            "WHERE evidence_id = ? ORDER BY ordinal",
            (evidence_id,),
        )
        return [row["message_id"] or row["state_item_id"] for row in cursor]

    def list_evidence(self, session_id: str) -> list[Evidence]:
        cursor = self._db.connection.execute(
            "SELECT * FROM evidence WHERE session_id = ? ORDER BY timestamp, id", (session_id,)
        )
        return [rows.row_to_evidence(row, self._evidence_sources(row["id"])) for row in cursor]

    def find_evidence_by_hash(self, content_hash: str) -> list[Evidence]:
        """Every evidence record with this exact content. Used to detect duplicates."""
        cursor = self._db.connection.execute(
            "SELECT * FROM evidence WHERE content_hash = ? ORDER BY timestamp, id", (content_hash,)
        )
        return [rows.row_to_evidence(row, self._evidence_sources(row["id"])) for row in cursor]

    # ------------------------------------------------------------------
    # State items

    def add_state_item(self, item: StateItem) -> StateItem:
        self.add_state_items([item])
        return item

    def add_state_items(self, items: Sequence[StateItem]) -> int:
        """Write state items atomically, resolving intra-batch references."""
        if not items:
            return 0
        pending = {item.id: item.session_id for item in items}
        with self._db.transaction() as connection:
            for item in items:
                self._write_state_item(item, connection, pending)
            for item in items:
                if item.supersedes is not None:
                    self._check_supersession_acyclic(item.id, connection)
        return len(items)

    def _check_supersession_acyclic(self, state_id: str, connection: sqlite3.Connection) -> None:
        """Reject a write that would close a supersession loop.

        A batch can contain items that supersede each other, and the referential
        checks alone would accept it. Supersession says which record replaced
        which, so a loop leaves no current state to resolve to. Checked inside
        the transaction, so a violation rolls the whole batch back.
        """
        seen = {state_id}
        current = state_id
        while True:
            row = connection.execute(
                "SELECT supersedes FROM state_items WHERE id = ?", (current,)
            ).fetchone()
            if row is None or row["supersedes"] is None:
                return
            current = str(row["supersedes"])
            if current in seen:
                raise SupersessionCycleError(state_id)
            seen.add(current)

    def _write_state_item(
        self, item: StateItem, connection: sqlite3.Connection, pending: Mapping[str, str]
    ) -> None:
        self._require_absent("state_items", item.id, connection)
        self._require("sessions", item.session_id, source_id=item.id, connection=connection)

        if item.supersedes is not None:
            self._require(
                "state_items",
                item.supersedes,
                source_id=item.id,
                connection=connection,
                session_id=item.session_id,
                pending=pending,
            )
        for reference in item.related_ids:
            self._require(
                "state_items",
                reference,
                source_id=item.id,
                connection=connection,
                session_id=item.session_id,
                pending=pending,
            )
        for reference in item.sources:
            message_id, _ = rows.split_state_source(reference)
            table = "messages" if message_id is not None else "evidence"
            self._require(
                table,
                reference,
                source_id=item.id,
                connection=connection,
                session_id=item.session_id,
            )

        blocked_by: tuple[str, ...] = tuple(getattr(item, "blocked_by", ()))
        for reference in blocked_by:
            self._require(
                "state_items",
                reference,
                source_id=item.id,
                connection=connection,
                session_id=item.session_id,
                pending=pending,
            )

        connection.execute(
            "INSERT INTO state_items (id, session_id, type, content, status, retention, "
            "importance, confidence, trust, created_at, supersedes, attributes, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows.state_to_row(item),
        )
        connection.executemany(
            "INSERT INTO state_sources (state_id, ordinal, message_id, evidence_id) "
            "VALUES (?, ?, ?, ?)",
            [
                (item.id, ordinal, *rows.split_state_source(reference))
                for ordinal, reference in enumerate(item.sources)
            ],
        )
        relations: list[tuple[str, str, int, str]] = [
            (item.id, "related", ordinal, reference)
            for ordinal, reference in enumerate(item.related_ids)
        ]
        relations.extend(
            (item.id, "blocks", ordinal, reference) for ordinal, reference in enumerate(blocked_by)
        )
        connection.executemany(
            "INSERT INTO state_relations (state_id, kind, ordinal, target_id) VALUES (?, ?, ?, ?)",
            relations,
        )

    def _state_sources(self, state_id: str) -> list[str]:
        cursor = self._db.connection.execute(
            "SELECT message_id, evidence_id FROM state_sources WHERE state_id = ? ORDER BY ordinal",
            (state_id,),
        )
        return [row["message_id"] or row["evidence_id"] for row in cursor]

    def _state_relations(self, state_id: str, kind: str) -> list[str]:
        cursor = self._db.connection.execute(
            "SELECT target_id FROM state_relations "
            "WHERE state_id = ? AND kind = ? ORDER BY ordinal",
            (state_id, kind),
        )
        return [row["target_id"] for row in cursor]

    def _hydrate_state(self, row: sqlite3.Row) -> StateItem:
        state_id = row["id"]
        return rows.row_to_state(
            row,
            sources=self._state_sources(state_id),
            related_ids=self._state_relations(state_id, "related"),
            blocked_by=self._state_relations(state_id, "blocks"),
        )

    def get_state_item(self, state_id: str) -> StateItem:
        row = self._db.connection.execute(
            "SELECT * FROM state_items WHERE id = ?", (state_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(state_id)
        return self._hydrate_state(row)

    def list_state_items(
        self,
        session_id: str,
        *,
        types: Iterable[str] | None = None,
        status: StateStatus | None = None,
    ) -> list[StateItem]:
        query = "SELECT * FROM state_items WHERE session_id = ?"
        params: list[Any] = [session_id]
        if types is not None:
            wanted = list(types)
            if not wanted:
                return []
            query += f" AND type IN ({','.join('?' * len(wanted))})"
            params.extend(wanted)
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY created_at, id"
        cursor = self._db.connection.execute(query, params)
        return [self._hydrate_state(row) for row in cursor]

    # ------------------------------------------------------------------
    # Supersession, derived from edges rather than stored

    def superseded_by(self, state_id: str) -> StateItem | None:
        """The item that replaced this one, if any."""
        row = self._db.connection.execute(
            "SELECT * FROM state_items WHERE supersedes = ? ORDER BY created_at, id LIMIT 1",
            (state_id,),
        ).fetchone()
        return None if row is None else self._hydrate_state(row)

    def is_superseded(self, state_id: str) -> bool:
        """Whether something later replaced this item.

        Read from the graph, not from a stored flag. Writing a flag would mean
        editing a historical record, which the architecture forbids.
        """
        row = self._db.connection.execute(
            "SELECT 1 FROM state_items WHERE supersedes = ? LIMIT 1", (state_id,)
        ).fetchone()
        return row is not None

    def supersession_chain(self, state_id: str) -> list[StateItem]:
        """The full lineage from the given item forward to the current one.

        The first element is the item asked for and the last is whatever nothing
        else has replaced. Loops terminate rather than hang.
        """
        chain: list[StateItem] = [self.get_state_item(state_id)]
        seen = {state_id}
        while True:
            successor = self.superseded_by(chain[-1].id)
            if successor is None or successor.id in seen:
                return chain
            chain.append(successor)
            seen.add(successor.id)

    def current_state_items(self, session_id: str) -> list[StateItem]:
        """State items nothing has replaced.

        Rejected items are excluded; a rejected decision is history, not current
        state, and it stays reachable through ``list_state_items``.
        """
        cursor = self._db.connection.execute(
            "SELECT s.* FROM state_items s "
            "WHERE s.session_id = ? AND s.status <> ? "
            "AND NOT EXISTS (SELECT 1 FROM state_items t WHERE t.supersedes = s.id) "
            "ORDER BY s.created_at, s.id",
            (session_id, StateStatus.REJECTED.value),
        )
        return [self._hydrate_state(row) for row in cursor]

    # ------------------------------------------------------------------
    # Snapshots

    def add_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        with self._db.transaction() as connection:
            self._require_absent("snapshots", snapshot.id, connection)
            self._require(
                "sessions", snapshot.session_id, source_id=snapshot.id, connection=connection
            )
            if snapshot.parent_snapshot_id is not None:
                self._require(
                    "snapshots",
                    snapshot.parent_snapshot_id,
                    source_id=snapshot.id,
                    connection=connection,
                    session_id=snapshot.session_id,
                )
            for reference in (*snapshot.active_message_ids, *snapshot.archived_message_ids):
                self._require(
                    "messages",
                    reference,
                    source_id=snapshot.id,
                    connection=connection,
                    session_id=snapshot.session_id,
                )
            for reference in snapshot.state_item_ids:
                self._require(
                    "state_items",
                    reference,
                    source_id=snapshot.id,
                    connection=connection,
                    session_id=snapshot.session_id,
                )

            connection.execute(
                "INSERT INTO snapshots (id, session_id, parent_snapshot_id, created_at, "
                "token_value, token_method, token_exact, compiler_version, reason, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows.snapshot_to_row(snapshot),
            )
            membership = [
                (snapshot.id, "active", ordinal, reference)
                for ordinal, reference in enumerate(snapshot.active_message_ids)
            ]
            membership.extend(
                (snapshot.id, "archived", ordinal, reference)
                for ordinal, reference in enumerate(snapshot.archived_message_ids)
            )
            connection.executemany(
                "INSERT INTO snapshot_messages (snapshot_id, bucket, ordinal, message_id) "
                "VALUES (?, ?, ?, ?)",
                membership,
            )
            connection.executemany(
                "INSERT INTO snapshot_state_items (snapshot_id, ordinal, state_id) "
                "VALUES (?, ?, ?)",
                [
                    (snapshot.id, ordinal, reference)
                    for ordinal, reference in enumerate(snapshot.state_item_ids)
                ],
            )
        return snapshot

    def _snapshot_messages(self, snapshot_id: str, bucket: str) -> list[str]:
        cursor = self._db.connection.execute(
            "SELECT message_id FROM snapshot_messages WHERE snapshot_id = ? AND bucket = ? "
            "ORDER BY ordinal",
            (snapshot_id, bucket),
        )
        return [row["message_id"] for row in cursor]

    def _snapshot_state_ids(self, snapshot_id: str) -> list[str]:
        cursor = self._db.connection.execute(
            "SELECT state_id FROM snapshot_state_items WHERE snapshot_id = ? ORDER BY ordinal",
            (snapshot_id,),
        )
        return [row["state_id"] for row in cursor]

    def _hydrate_snapshot(self, row: sqlite3.Row) -> ContextSnapshot:
        snapshot_id = row["id"]
        return rows.row_to_snapshot(
            row,
            active_message_ids=self._snapshot_messages(snapshot_id, "active"),
            archived_message_ids=self._snapshot_messages(snapshot_id, "archived"),
            state_item_ids=self._snapshot_state_ids(snapshot_id),
        )

    def get_snapshot(self, snapshot_id: str) -> ContextSnapshot:
        row = self._db.connection.execute(
            "SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(snapshot_id)
        return self._hydrate_snapshot(row)

    def list_snapshots(self, session_id: str) -> list[ContextSnapshot]:
        cursor = self._db.connection.execute(
            "SELECT * FROM snapshots WHERE session_id = ? ORDER BY created_at, id", (session_id,)
        )
        return [self._hydrate_snapshot(row) for row in cursor]

    def latest_snapshot(self, session_id: str) -> ContextSnapshot | None:
        row = self._db.connection.execute(
            "SELECT * FROM snapshots WHERE session_id = ? ORDER BY created_at DESC, id DESC "
            "LIMIT 1",
            (session_id,),
        ).fetchone()
        return None if row is None else self._hydrate_snapshot(row)

    def snapshot_lineage(self, snapshot_id: str) -> list[ContextSnapshot]:
        """The chain of parents from this snapshot back to the first.

        Oldest first. Terminates on a cycle rather than looping.
        """
        current = self.get_snapshot(snapshot_id)
        lineage = [current]
        seen = {current.id}
        while current.parent_snapshot_id is not None:
            if current.parent_snapshot_id in seen:
                break
            current = self.get_snapshot(current.parent_snapshot_id)
            lineage.append(current)
            seen.add(current.id)
        return list(reversed(lineage))

    # ------------------------------------------------------------------
    # Counts

    def counts(self) -> dict[str, int]:
        """Row counts per table. For inspection and tests, not for reporting."""
        tables = (
            "sessions",
            "messages",
            "evidence",
            "state_items",
            "snapshots",
        )
        result: dict[str, int] = {}
        for table in tables:
            row = self._db.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            result[table] = int(row["n"])
        return result


__all__ = ["Repository", "StorageError"]
