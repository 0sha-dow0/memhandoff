"""Migration atomicity.

A failing migration must leave the database exactly as it was before that
migration began. This is not assumed from SQLite's transaction behaviour: the
first implementation used ``executescript``, which issues an implicit COMMIT
before running and left a failed migration's completed statements permanently
applied. These tests demonstrate the required behaviour directly.
"""

import sqlite3
from unittest import mock

import pytest

from open_context.storage import SCHEMA_VERSION, current_version, migrate, split_statements
from open_context.storage import database as database_module

FIRST = "CREATE TABLE alpha (x TEXT) STRICT;"
BREAKS_PART_WAY = (
    "CREATE TABLE beta (y TEXT) STRICT;CREATE TABLE gamma (z TEXT) STRICT;THIS IS NOT VALID SQL;"
)


@pytest.fixture
def connection():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def tables(conn):
    query = "SELECT name FROM sqlite_master WHERE type='table'"
    return {row["name"] for row in conn.execute(query)}


def test_failed_migration_leaves_the_database_at_the_previous_version(connection):
    with (
        mock.patch.object(database_module, "MIGRATIONS", ((1, FIRST), (2, BREAKS_PART_WAY))),
        pytest.raises(sqlite3.Error),
    ):
        migrate(connection, target=2)

    assert current_version(connection) == 1, "version must not advance"
    present = tables(connection)
    assert "alpha" in present, "the migration that succeeded stays applied"
    assert "beta" not in present, "a table created before the failure must be rolled back"
    assert "gamma" not in present


def test_failed_migration_writes_no_version_row(connection):
    with (
        mock.patch.object(database_module, "MIGRATIONS", ((1, FIRST), (2, BREAKS_PART_WAY))),
        pytest.raises(sqlite3.Error),
    ):
        migrate(connection, target=2)

    recorded = [row["version"] for row in connection.execute("SELECT version FROM schema_meta")]
    assert recorded == [1], "no metadata for the migration that failed"


def test_the_database_is_usable_after_a_failed_migration(connection):
    """A rolled back migration must not leave the connection in a broken
    transaction state."""
    with (
        mock.patch.object(database_module, "MIGRATIONS", ((1, FIRST), (2, BREAKS_PART_WAY))),
        pytest.raises(sqlite3.Error),
    ):
        migrate(connection, target=2)

    connection.execute("INSERT INTO alpha (x) VALUES ('still works')")
    assert connection.execute("SELECT COUNT(*) AS n FROM alpha").fetchone()["n"] == 1
    assert not connection.in_transaction


def test_retrying_after_a_fix_applies_cleanly(connection):
    """Once the broken migration is corrected, applying it must succeed with no
    residue from the failed attempt."""
    with (
        mock.patch.object(database_module, "MIGRATIONS", ((1, FIRST), (2, BREAKS_PART_WAY))),
        pytest.raises(sqlite3.Error),
    ):
        migrate(connection, target=2)

    fixed = "CREATE TABLE beta (y TEXT) STRICT;CREATE TABLE gamma (z TEXT) STRICT;"
    with mock.patch.object(database_module, "MIGRATIONS", ((1, FIRST), (2, fixed))):
        assert migrate(connection, target=2) == 2
    assert {"alpha", "beta", "gamma"} <= tables(connection)
    assert current_version(connection) == 2


def test_target_stops_before_later_migrations(connection):
    with mock.patch.object(
        database_module, "MIGRATIONS", ((1, FIRST), (2, "CREATE TABLE beta (y TEXT) STRICT;"))
    ):
        assert migrate(connection, target=1) == 1
    assert "beta" not in tables(connection)


def test_real_migrations_apply_to_the_declared_version(connection):
    assert migrate(connection) == SCHEMA_VERSION


def test_statements_are_split_individually():
    """executescript cannot be used, so the script is split and each statement
    runs inside the caller's transaction."""
    assert split_statements("CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT);") == [
        "CREATE TABLE a (x TEXT);",
        "CREATE TABLE b (y TEXT);",
    ]


def test_splitting_respects_semicolons_inside_string_literals():
    assert split_statements("INSERT INTO t VALUES ('a; b'); SELECT 1;") == [
        "INSERT INTO t VALUES ('a; b');",
        "SELECT 1;",
    ]
