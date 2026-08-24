# Archive

Phase 3. The local immutable archive: the durable record of what was actually said.

This is the source material compaction reads from. It is a component of the local engine, not the product; the product artifact is the portable context package described in [product-architecture.md](product-architecture.md).

## Archive versus SQLite

Two stores with different jobs. The conversation is not duplicated into SQLite.

| | Archive | SQLite |
| --- | --- | --- |
| Holds | Raw conversation, tool results, raw event payloads, large content | Indexed metadata, structured state, references, snapshots |
| Shape | Append-only JSONL on the filesystem | Relational, queryable |
| Access | Streaming, positional, range | Query by relationship |
| Knows about payload shape | Nothing | Everything |

The archive stores payloads whose shape nothing in this codebase understands. It deliberately does not import the Phase 1 models: those describe interpretation, and the archive holds the bytes interpretation was derived from.

## Layout

```
<root>/sessions/<session_id>/
    manifest.json     format, version, session id, creation time
    records.jsonl     the source of truth, one record per line
    records.idx       rebuildable offsets, 12 bytes per record
```

One log per session, so opening a session never touches another's bytes and a damaged session cannot cost the others.

The manifest declares which session the directory holds, and `open` refuses to proceed unless that matches the session asked for. A directory that was copied, renamed, or assembled by hand would otherwise attribute one session's conversation to another, which is a worse outcome than failing to open. Format and version are checked first, then identity: a manifest this build cannot read is not worth comparing identities against.

## Format, and why

**Append-only JSONL with a rebuildable binary sidecar index.**

Each line is a self-contained JSON object:

```json
{"hash":"a876...","id":"msg_0","kind":"message","payload":{"content":"hello"},"seq":0}
```

The hash is sha256 over the canonical serialisation of `seq`, `id`, `kind`, and `payload`. Keys are sorted and separators fixed, so the same content always produces the same bytes.

JSONL was chosen on one criterion above the others: when an archive is damaged, you can `tail -1` it and see what happened. A length-prefixed binary format would index faster and pack tighter, but it needs a purpose-built tool before anyone can debug it, and the write volume here does not justify that trade. Text also means the archive survives being copied, diffed, and grepped by people who have never heard of this project.

The index is 12 bytes per record, an unsigned 64-bit offset and an unsigned 32-bit length, little-endian. Entry N describes record N, so reading record N is a seek to `N * 12` and then a seek to the offset it names. Nothing scans.

**The index is derived data.** It can be deleted at any time and rebuilt from the JSONL, which is the only source of truth. An index that could not be rebuilt would be a second thing to keep consistent and a second thing to lose.

The length includes the terminating newline, so `offset + length` is exactly where the next record starts. Excluding it makes the end-of-index pointer land on the newline instead of past it, and recovery then eats a byte of the last good record. That bug existed during development and the tests now pin the fixed behaviour.

## Immutability

Records are appended and never rewritten. Positions are assigned by the archive and increase by one. There is no update and no delete.

Two records may share an id: a message and a later revision of it are distinct records, and position is what distinguishes them. Identity is preserved exactly as given.

## Integrity

Every record carries its own hash, so a line can be checked without consulting anything else. That is what makes verification streamable and lets damage be located precisely rather than invalidating the file.

`seq` is inside the hashed body, so records cannot be silently reordered or moved.

The hash is byte-exact and deliberately does **not** normalise Unicode. `Evidence.content_hash` normalises to NFC because its purpose is recognising duplicate content. The archive hash exists to detect corruption in stored originals, so it must notice every byte difference, including one composed form replacing another. Normalising here would hide a real change in what was written.

`verify()` streams the file and returns the positions of every damaged record rather than raising at the first one, so a caller can report the extent of the damage.

## Durability boundary

**The data file is the durability boundary.** An append is durable once `records.jsonl` has been written and `fsync`ed. Nothing is ever considered written on the strength of an index entry alone.

**The index is derived data and may lag.** It can be short, stale, deleted, or torn, and recovery rebuilds it from the data file. `rebuild_index()` does the same on demand. This is why the index needs no durability guarantee of its own: losing it costs a rescan, never a record.

Stated as one rule: if it is in the data file it happened, and if it is only in the index it did not.

## Concurrency

**A `SessionLog` has a single writer. Concurrent writes to the same session are unsupported in Phase 3.**

Nothing enforces this. There is no lock file, no advisory lock, no queue, no writer process. Two writers appending to one log will interleave their bytes and corrupt it, and neither will be told.

Reads are safe alongside other readers, and alongside a single writer: a reader only ever sees whole synced lines, and a torn tail is what recovery is for.

The write-coordination model is still deferred. Phase 4 did not need it: an import is one writer for the duration of one call, which is exactly the supported case. Choosing between a lock file, a single writer process, and a serialised queue depends on how agent adapters actually write — whether an adapter streams during a live session, whether a CLI can run against a session an agent is holding open, and whether writers share a machine. That is the adapter phases' question, and picking before then would mean picking twice.

## Crash behaviour

**Write order follows from the durability boundary.** A record is written to the data file and `fsync`ed before its index entry is written. If the process dies between the two, the index is short and the data file is whole, which recovery repairs by rescanning the tail. The reverse order would leave an index entry pointing at bytes that were never written, which is not repairable.

**Recovery is bounded.** Opening an archive scans only from the end of the last indexed record to the end of the file. A clean archive reads nothing.

**A torn tail is truncated and reported, never accepted.** An interrupted append leaves a line with no terminator, or a complete line that will not parse. Either is a failed write: the bytes are removed and `RecoveryReport` records how many were lost. The archive never treats a partial write as valid conversation history, and it never reports the loss silently.

**Damage anywhere else is corruption, not a failed write, and is never repaired.** A record in the body that fails its hash raises `CorruptRecordError` on read and appears in `verify()`. It is not truncated away, because that would be deleting history to make a file look healthy.

Tested cases: torn final record, unparseable final line, crash between data sync and index write, lost index, deleted index, torn index entry, index describing more records than the data file contains, mid-file corruption, reordered records, unknown format, future format version, manifest naming another session, manifest with no session id, repeated reopen.

## Memory

A 500k-token conversation must not need 500k tokens of RAM.

Every read path streams. `read` holds one record. `range`, `window`, `tail`, `__iter__`, and `verify` hold one record at a time regardless of how many they cover. Opening and appending do not load history.

There is no `read_all` and no `list_all_records`, deliberately.

This is enforced by tests using `tracemalloc`, not by inspection. They assert that peak memory while streaming stays well under the archive's size on disk, and that an eightfold larger archive does not cost proportionally more.

## API

```python
archive = Archive(root)
archive.create(session_id)          # new empty log, refuses to overwrite
archive.open(session_id)            # recovers a torn tail if there is one
archive.open_or_create(session_id)
archive.sessions()                  # iterator of session ids
archive.size_bytes(session_id)

log.append(record_id, payload, kind="message")   # returns the record with its seq
log.extend(pairs, kind="message")                # batch, one sync at the end
log.count

log.read(seq)                        # constant time, one record
log.range(start, stop)               # iterator, bounded
log.window(seq, before=, after=)     # iterator, clamped
log.tail(limit)                      # iterator
iter(log)                            # iterator over everything

log.verify()                         # IntegrityReport(records, corrupt)
log.rebuild_index()
log.recovery                         # RecoveryReport from the last open
```

## Errors

| Error | Meaning |
| --- | --- |
| `ArchiveNotFoundError` | No archive at that location |
| `ArchiveExistsError` | One already exists; append-only, so overwriting is never the intent |
| `RecordNotFoundError` | No record at that position |
| `CorruptRecordError` | A record fails its hash or will not parse |
| `ArchiveFormatError` | Written in a format or version this build does not understand |
| `ManifestMismatchError` | The manifest names a different session than the directory holding it |
| `DuplicateSeqError` | Defined for a rewrite attempt; unreachable while positions are archive-assigned |

## Unresolved

**Single-writer is assumed and unenforced.** Stated as an invariant under [Concurrency](#concurrency) rather than implemented. Phase 4 decides the write-coordination model, once agent adapters show what actually contends.

**No compaction of the archive itself.** It grows forever. That is correct for now, since it is the source of truth, but a very long-lived project will want segmentation or cold storage.

**Nothing links the archive to SQLite yet.** A `Message` row and its archive record refer to the same thing by id with no enforced correspondence, and no phase has yet defined which writes both. Phase 4 did not resolve it: import writes only the archive, and the connection to the session layer is a shared session id by convention. See [import.md](import.md).

**`fsync` per append is slow on some filesystems.** `extend` amortises it, but a high-frequency single-append workload may need a configurable durability level. No measurement exists yet.

**Manifest is not hashed.** Format, version, and session identity are checked on open, so a manifest that disagrees with its directory is caught. Everything else about it is not: `created_at` can be rewritten freely, and damage that still parses into the right three fields is invisible. The check is a consistency check, not an authenticity one.
