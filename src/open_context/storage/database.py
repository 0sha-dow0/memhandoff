"""Database connection and migrations.

Every connection sets the same pragmas. ``foreign_keys`` is off by default in
SQLite and is per-connection, not per-database, so forgetting it on one
connection silently disables every referential integrity guarantee in the
schema.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from open_context.storage.errors import SchemaVersionError
from open_context.storage.schema import MIGRATIONS, SCHEMA_VERSION

MEMORY: str = ":memory:"

_META_TABLE = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version    INTEGER NOT NULL,
    applied_at TEXT NOT NULL
) STRICT;
"""


def _configure(connection: sqlite3.Connection, *, wal: bool) -> None:
    """Apply the pragmas every connection needs.

    ``foreign_keys`` is the important one. It defaults to off and is scoped to
    the connection, so a connection that skips it gets no referential integrity
    at all while appearing to work.
    """
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if wal:
        connection.execute("PRAGMA journal_mode = WAL")
    connection.row_factory = sqlite3.Row


def current_version(connection: sqlite3.Connection) -> int:
    """Schema version of an open database. Zero means empty."""
    connection.execute(_META_TABLE)
    row = connection.execute("SELECT MAX(version) AS v FROM schema_meta").fetchone()
    value = row["v"]
    return 0 if value is None else int(value)


def split_statements(script: str) -> list[str]:
    """Split a migration script into individual statements.

    ``executescript`` cannot be used for migrations: it issues an implicit
    COMMIT before running, which ends any transaction the caller opened and
    leaves a failed migration's completed statements permanently applied.
    Splitting lets each statement run inside one explicit transaction instead.

    ``sqlite3.complete_statement`` is used rather than splitting on semicolons,
    so a semicolon inside a string literal does not break the script.
    """
    statements: list[str] = []
    buffer = ""
    for character in script:
        buffer += character
        if character == ";" and sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                statements.append(statement)
            buffer = ""
    tail = buffer.strip()
    if tail:
        statements.append(tail)
    return statements


def migrate(connection: sqlite3.Connection, *, target: int = SCHEMA_VERSION) -> int:
    """Apply pending migrations in order and return the resulting version.

    Deterministic and idempotent: running it twice applies nothing the second
    time.

    Each migration is atomic. Every statement in it, plus the row recording its
    version, runs inside one explicit transaction. If any statement fails, the
    whole migration rolls back and the database is left exactly as it was before
    that migration began: no half-created tables, no version row, no partial
    schema change. Migrations already applied are unaffected, since each commits
    on its own.
    """
    version = current_version(connection)
    if version > target:
        raise SchemaVersionError(found=version, expected=target)

    for number, script in MIGRATIONS:
        if number <= version or number > target:
            continue
        connection.execute("BEGIN")
        try:
            for statement in split_statements(script):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_meta (version, applied_at) VALUES (?, ?)",
                (number, datetime.now(UTC).isoformat()),
            )
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        version = number
    return version


class Database:
    """An open SQLite database with the schema applied.

    Usable as a context manager. One instance owns one connection and is not
    safe to share across threads, which matches ``sqlite3``'s own default.
    """

    def __init__(self, path: str | Path = MEMORY, *, migrate_on_open: bool = True) -> None:
        self.path = str(path)
        is_memory = self.path == MEMORY
        if not is_memory:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None)
        _configure(self._connection, wal=not is_memory)
        if migrate_on_open:
            migrate(self._connection)
        else:
            self._require_current_version()

    def _require_current_version(self) -> None:
        version = current_version(self._connection)
        if version != SCHEMA_VERSION:
            raise SchemaVersionError(found=version, expected=SCHEMA_VERSION)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def schema_version(self) -> int:
        return current_version(self._connection)

    @contextmanager
    def transaction(self, *, defer_foreign_keys: bool = True) -> Iterator[sqlite3.Connection]:
        """Run a unit of work atomically.

        Foreign keys are deferred to commit time by default. Records reference
        each other in both directions (evidence cites a state item, a state item
        cites evidence) and lists may point forward, so no static insert order
        satisfies immediate checks. Deferring means the constraints are all
        checked at COMMIT, and any failure rolls the whole unit back.
        """
        connection = self._connection
        connection.execute("BEGIN")
        if defer_foreign_keys:
            connection.execute("PRAGMA defer_foreign_keys = ON")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
