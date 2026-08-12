"""Schema definition.

One list of migrations, applied in order, each one an integer version. There is
no schema autogeneration and no reflection: the DDL is written out so a diff
shows exactly what changed in a database that holds the only copy of somebody's
conversation history.

Design notes that are not obvious from the DDL:

**List fields become junction tables with an ordinal.** ``sources``,
``related_ids``, and the snapshot id lists are ordered Python lists. Without the
ordinal column a record would not round trip equal to itself.

**Polymorphic references get two nullable columns and a CHECK.** A state item's
source may be a message or an evidence record, and SQLite cannot point one
foreign key at two tables. Two nullable columns with a check that exactly one is
set gives real referential integrity in both directions.

**State subtype fields live in a JSON ``attributes`` column.** Ten tables would
mean a join on every read and a migration whenever a state type is added. One
wide table with nullable columns would mean SQLite cannot require a decision to
have a rationale, since the column has to be nullable for the other nine types.
Type-specific invariants stay in Pydantic, which is where they already are.

**Timestamps are ISO 8601 text in UTC.** SQLite has no datetime type. UTC ISO
strings sort lexicographically in chronological order, so ordering by them is
correct without parsing.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final = 1

_MIGRATION_1: Final = """
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    source      TEXT,
    created_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
) STRICT;

CREATE TABLE messages (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    seq          INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    trust        TEXT NOT NULL,
    parent_id    TEXT REFERENCES messages(id),
    tool_name    TEXT,
    tool_call_id TEXT,
    metadata     TEXT NOT NULL DEFAULT '{}',
    UNIQUE (session_id, seq),
    CHECK (seq >= 0),
    CHECK (id <> parent_id)
) STRICT;

CREATE INDEX idx_messages_session_seq ON messages(session_id, seq);
CREATE INDEX idx_messages_parent ON messages(parent_id);

CREATE TABLE state_items (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    type        TEXT NOT NULL,
    content     TEXT NOT NULL,
    status      TEXT NOT NULL,
    retention   TEXT NOT NULL,
    importance  REAL NOT NULL,
    confidence  REAL NOT NULL,
    trust       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    supersedes  TEXT REFERENCES state_items(id),
    attributes  TEXT NOT NULL DEFAULT '{}',
    metadata    TEXT NOT NULL DEFAULT '{}',
    CHECK (importance >= 0.0 AND importance <= 1.0),
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (id <> supersedes)
) STRICT;

CREATE INDEX idx_state_session_type ON state_items(session_id, type);
CREATE INDEX idx_state_supersedes ON state_items(supersedes);
CREATE INDEX idx_state_created ON state_items(created_at);

CREATE TABLE evidence (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    source_type  TEXT NOT NULL,
    trust        TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
) STRICT;

CREATE INDEX idx_evidence_session ON evidence(session_id);
CREATE INDEX idx_evidence_hash ON evidence(content_hash);

-- Evidence may cite a message or a state item (an artifact or an event).
CREATE TABLE evidence_sources (
    evidence_id   TEXT NOT NULL REFERENCES evidence(id),
    ordinal       INTEGER NOT NULL,
    message_id    TEXT REFERENCES messages(id),
    state_item_id TEXT REFERENCES state_items(id),
    PRIMARY KEY (evidence_id, ordinal),
    CHECK ((message_id IS NULL) <> (state_item_id IS NULL))
) STRICT;

-- A state item's provenance may cite a message or an evidence record.
CREATE TABLE state_sources (
    state_id    TEXT NOT NULL REFERENCES state_items(id),
    ordinal     INTEGER NOT NULL,
    message_id  TEXT REFERENCES messages(id),
    evidence_id TEXT REFERENCES evidence(id),
    PRIMARY KEY (state_id, ordinal),
    CHECK ((message_id IS NULL) <> (evidence_id IS NULL))
) STRICT;

-- 'related' comes from related_ids, 'blocks' from Task.blocked_by.
CREATE TABLE state_relations (
    state_id  TEXT NOT NULL REFERENCES state_items(id),
    kind      TEXT NOT NULL,
    ordinal   INTEGER NOT NULL,
    target_id TEXT NOT NULL REFERENCES state_items(id),
    PRIMARY KEY (state_id, kind, ordinal),
    CHECK (kind IN ('related', 'blocks')),
    CHECK (state_id <> target_id)
) STRICT;

CREATE INDEX idx_state_relations_target ON state_relations(target_id);

CREATE TABLE snapshots (
    id                 TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL REFERENCES sessions(id),
    parent_snapshot_id TEXT REFERENCES snapshots(id),
    created_at         TEXT NOT NULL,
    token_value        INTEGER,
    token_method       TEXT,
    token_exact        INTEGER,
    compiler_version   TEXT,
    reason             TEXT,
    metadata           TEXT NOT NULL DEFAULT '{}',
    CHECK (id <> parent_snapshot_id),
    CHECK (token_value IS NULL OR token_value >= 0),
    CHECK (token_exact IN (0, 1) OR token_exact IS NULL),
    -- A token count is stored with its method or not at all.
    CHECK (
        (token_value IS NULL AND token_method IS NULL AND token_exact IS NULL)
        OR (token_value IS NOT NULL AND token_method IS NOT NULL AND token_exact IS NOT NULL)
    )
) STRICT;

CREATE INDEX idx_snapshots_session ON snapshots(session_id, created_at);
CREATE INDEX idx_snapshots_parent ON snapshots(parent_snapshot_id);

CREATE TABLE snapshot_messages (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
    bucket      TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    message_id  TEXT NOT NULL REFERENCES messages(id),
    PRIMARY KEY (snapshot_id, bucket, ordinal),
    CHECK (bucket IN ('active', 'archived')),
    -- A message is active or archived in a snapshot, never both.
    UNIQUE (snapshot_id, message_id)
) STRICT;

CREATE TABLE snapshot_state_items (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
    ordinal     INTEGER NOT NULL,
    state_id    TEXT NOT NULL REFERENCES state_items(id),
    PRIMARY KEY (snapshot_id, ordinal),
    UNIQUE (snapshot_id, state_id)
) STRICT;
"""

MIGRATIONS: Final[tuple[tuple[int, str], ...]] = ((1, _MIGRATION_1),)
"""Ordered migrations. Append only; never edit one that has shipped."""


def latest_version() -> int:
    """Highest version in the migration list."""
    return max(version for version, _ in MIGRATIONS)
