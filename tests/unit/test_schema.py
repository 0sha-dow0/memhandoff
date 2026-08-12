"""Schema and migration determinism."""

import sqlite3

import pytest

from open_context.storage import SCHEMA_VERSION, Database, current_version, latest_version, migrate
from open_context.storage.errors import SchemaVersionError
from open_context.storage.schema import MIGRATIONS


def test_migration_versions_are_ordered_and_unique():
    versions = [version for version, _ in MIGRATIONS]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


def test_latest_version_matches_declared_version():
    assert latest_version() == SCHEMA_VERSION


def test_empty_database_is_version_zero():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    assert current_version(connection) == 0


def test_migrate_is_idempotent():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    assert migrate(connection) == SCHEMA_VERSION
    assert migrate(connection) == SCHEMA_VERSION
    rows = connection.execute("SELECT COUNT(*) AS n FROM schema_meta").fetchone()
    assert rows["n"] == len(MIGRATIONS)


def test_database_from_the_future_is_rejected():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    migrate(connection)
    connection.execute(
        "INSERT INTO schema_meta (version, applied_at) VALUES (?, ?)", (99, "2026-01-01T00:00:00Z")
    )
    with pytest.raises(SchemaVersionError, match="schema version 99"):
        migrate(connection)


def test_expected_tables_exist():
    with Database() as db:
        found = {
            row["name"]
            for row in db.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    expected = {
        "sessions",
        "messages",
        "evidence",
        "evidence_sources",
        "state_items",
        "state_sources",
        "state_relations",
        "snapshots",
        "snapshot_messages",
        "snapshot_state_items",
        "schema_meta",
    }
    assert expected <= found


def test_foreign_keys_are_enabled_on_every_connection():
    """Off by default and per-connection. Forgetting it disables every guarantee
    in the schema while everything still appears to work."""
    with Database() as db:
        assert db.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_tables_are_strict():
    """STRICT rejects a non-numeric string in an INTEGER column instead of storing it.

    Without STRICT, SQLite would accept "not-a-number" into messages.seq and the
    corruption would surface much later as an ordering bug.
    """
    with Database() as db:
        db.connection.execute(
            "INSERT INTO sessions (id, created_at, metadata) VALUES (?, ?, ?)",
            ("ses_" + "a" * 24, "2026-01-01T00:00:00+00:00", "{}"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.connection.execute(
                "INSERT INTO messages (id, session_id, seq, role, content, timestamp, trust) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "msg_" + "a" * 24,
                    "ses_" + "a" * 24,
                    "not-a-number",
                    "user",
                    "hi",
                    "2026-01-01T00:00:00+00:00",
                    "user_authored",
                ),
            )
