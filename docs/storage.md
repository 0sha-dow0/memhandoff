# Storage

Phase 2. SQLite persistence for the Phase 1 models. Append and read only.

This is a component of the local compaction engine, not the product. It is the indexed side of a two-store design: raw conversation lives in the [archive](archive.md), and is not duplicated here. The product artifact is the portable context package described in [product-architecture.md](product-architecture.md); the database is where the engine keeps the raw history it compacts from.

## The contract

There is no `update` and no `delete` on the repository. Immutability is enforced by the absence of the operation, not by a check inside one, and there is a test asserting the method names do not exist.

A change is a new record whose `supersedes` points at the old one. Both survive.

## Supersession is derived, not written

Marking a replaced item `SUPERSEDED` would mean editing a stored historical record, which the architecture forbids. So the flag is never written. An item is superseded when some other item's `supersedes` points at it, and that is read from the graph:

```python
repo.is_superseded(state_id)  # anything replace it?
repo.superseded_by(state_id)  # what replaced it
repo.supersession_chain(state_id)  # full lineage, oldest first
repo.current_state_items(session)  # nothing has replaced these
```

`StateStatus.SUPERSEDED` still exists in the model and an extractor may author an item with it, but the repository never sets it. The edge is the truth.

Rejected items are excluded from `current_state_items` and stay reachable through `list_state_items`, because why an approach was rejected is one of the main questions people return to a long session to answer.

## Schema decisions

**One `state_items` table, not ten.** Table-per-type means a join on every read and a migration whenever a state type is added. One wide table with nullable columns means SQLite cannot require a `Decision` to have a rationale, since the column must be nullable for the other nine types. Shared fields are real columns; subtype fields go in a JSON `attributes` column and come back through `StateItemAdapter`, so type-specific invariants stay in Pydantic where they already are.

**List fields become junction tables with an ordinal.** `sources`, `related_ids`, `blocked_by`, and the snapshot id lists are ordered. Without the ordinal column a record would not round trip equal to itself, and there are tests asserting order survives.

**Polymorphic references get two nullable columns and a CHECK.** A state item's source is a message or an evidence record; an evidence source is a message or an artifact or event state item. SQLite cannot aim one foreign key at two tables, so each junction row carries two nullable FK columns with `CHECK ((a IS NULL) <> (b IS NULL))`. Both directions are genuinely enforced.

**`Task.blocked_by` is stored as relation rows, not in the JSON column.** Those are references, and references belong in a table the database can check.

**Timestamps are ISO 8601 text in UTC.** SQLite has no datetime type. UTC ISO strings sort lexicographically in chronological order, so ordering by them needs no parsing.

**Tables are `STRICT`.** Without it SQLite accepts `"not-a-number"` into `messages.seq` and the corruption surfaces much later as an ordering bug.

**`UNIQUE (session_id, seq)`.** Message order is the one thing the runtime cannot get wrong. An importer meeting a source export with genuine duplicate positions has to renumber, which is a Phase 4 problem.

## Transactions

Every multi-record write is one transaction. A batch either lands entirely or not at all, junction rows included.

Foreign keys are deferred to commit inside a transaction. Evidence cites a state item and a state item cites evidence, and a message's parent may appear later in the same import batch, so no static insert order satisfies immediate checks.

References are also checked in Python before insert. Deferred constraints surface as one opaque error at `COMMIT` with no indication of which reference broke; the explicit check produces `DanglingReferenceError` naming the record and the reference. The database constraint stays as the backstop.

## Migrations

An ordered list of `(version, sql)` in `schema.py`, applied in order, each committing its own version row. Running it twice applies nothing the second time. An interrupted run leaves the database at the last fully applied version rather than halfway through one.

Never edit a migration that has shipped. Append a new one.

Opening a database written by a newer schema raises `SchemaVersionError` rather than proceeding against a shape this build does not understand.

## Errors

| Error | Meaning |
| --- | --- |
| `RecordNotFoundError` | Requested by id, not present |
| `DuplicateRecordError` | That id is already stored. Records are immutable, so overwriting is never the intent |
| `DanglingReferenceError` | A reference points at something not stored. Names both ids |
| `SequenceConflictError` | Two messages claim the same position in one session |
| `CrossSessionReferenceError` | A reference points into a different session |
| `SupersessionCycleError` | The write would close a supersession loop |
| `SerializationError` | A `metadata` value cannot be written as JSON |
| `SchemaVersionError` | The database was written by a different schema version |

## Known limitations

`foreign_keys` is per-connection and off by default in SQLite. `Database` sets it on every connection it opens; code that connects some other way gets no integrity at all while appearing to work.

One `Database` owns one connection and is not safe to share across threads, matching `sqlite3`'s own default.

Reads materialize entire result sets. `list_messages` loads every message in a session into memory, which conflicts with the requirement that a large archive not imply proportional RAM. Streaming or paged reads are needed before the archive holds a realistically long history.

There is no deletion path at all, not even an explicit one. The architecture calls for deletion to be an explicit, audited user operation; that has no phase yet.
