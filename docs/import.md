# Import

Phase 4. Getting conversation history from a provider's export into the archive, without the rest of the runtime learning whose export it was.

```
provider export -> importer -> RawEvent -> archive -> later phases
```

## What import is, and is not

Import answers **"what happened in this conversation"**. It does not answer **"what does this conversation mean"**.

There is no summarization here, no extraction, no retrieval, no embedding, and no model call. A message saying "We should use PostgreSQL" imports as a message saying that. It does not become a `Decision` to use PostgreSQL, and it does not become a `Goal`, a `Task`, or a `Fact`. That transformation is Phase 6's, and doing it here would mean the archive stored an interpretation instead of what was said.

| Layer | Question | Phase |
| --- | --- | --- |
| Import | What happened? | 4 |
| Extraction | What does it mean? | 6 |

## Normalized event

`RawEvent` is the smallest provider-neutral description of one thing that happened.

| Field | Holds |
| --- | --- |
| `type` | `EventType`: user, assistant, system, developer message, tool call, tool result, or `other` |
| `provider` | Which adapter produced the event, such as `generic-jsonl` |
| `raw` | The provider record as parsed, with unknown fields preserved |
| `source_id` | The provider's identifier, verbatim. `None` if it gave none |
| `source_type` | The provider's own name for this event kind, verbatim |
| `timestamp` | Timezone-aware UTC. `None` when absent or unusable |
| `text` | Human-readable content, when the source carries it as text |
| `tool_name`, `tool_call_id` | Tool identity and the call/result correlation id |
| `parent_id` | The provider's own parent reference, verbatim and unvalidated |
| `metadata` | Adapter-supplied extras. Not provider content |

The message kinds mirror `models.enums.Role`, so a later phase turning events into `Message` records does not have to invent a mapping.

**There is no ordering field.** Position comes from the archive, which assigns `seq` on append. An importer yields in source order and the pipeline appends in the order received, so archive order is source order. A position on the event as well would be a second ordering that could disagree with the first.

**Timestamps are never invented.** A source with no usable timestamp produces `None`, not the time of import. Import time is a fact about the import; writing it into the history would make an inspection weeks later unable to tell the two apart.

## Raw payload preservation

**The normalized event is an interpretation of the transport format, not a replacement for the source record.**

Every event carries `raw`, the provider record as parsed, with unknown fields preserved. When an adapter meets a field this model has no home for, the field survives there and can be read by code that understands it later. A record whose shape nothing here anticipated still imports: it becomes `EventType.OTHER`, keeps the provider's own name in `source_type`, and keeps every field in `raw`.

This duplicates content and costs disk. It is deliberate. Storing only `raw` would push provider parsing into every reader; storing only the normalized fields would discard everything the model has no field for, permanently, at the one point where it was still available.

**`raw` is the parsed record, not the source bytes.** It is what the adapter's parser produced. Values survive a round trip through the archive intact; representation does not. Key order is normalised by canonical serialisation, and anything the parser resolved before the adapter saw it — a repeated JSON key collapsed to its last value, the exact spelling of a number, the original whitespace — is already gone. Byte-level preservation of the source would be a different feature with a different cost, and nothing here claims it. What is guaranteed is that no *field* is dropped.

## Importer interface

```python
class ConversationImporter(Protocol):
    @property
    def provider(self) -> str: ...
    def detect(self, sample: str) -> bool: ...
    def read(self, stream: TextIO) -> Iterator[RawEvent | MalformedRecord]: ...
```

An importer knows one provider's transport format and nothing else. It does not open files, choose session ids, write to the archive, decide what to do about a broken record, or count anything; those are identical for every provider and live in the pipeline. A new adapter is only the part that is genuinely provider-specific.

**A read yields a union.** `read` yields `RawEvent` for a record it could normalize and `MalformedRecord` for one it could not, rather than raising. A generator that raises is finished, so a caller wanting to note the problem and continue could not. Returning the failure as a value also makes "do not silently drop data" a property of the type instead of a rule every adapter author has to remember.

**Provider neutrality.** `RawEvent`, the pipeline, and the archive contain no branch on provider and no place to put one. The pipeline sees events and failures and could not name the adapter that produced them. Anything of the form `if provider == "claude"` belongs in an adapter or nowhere.

## Adapters

| Adapter | Format | Status |
| --- | --- | --- |
| `JsonLinesImporter` | One JSON object per line | Implemented |
| Claude Code | Unverified | Not started, see below |
| Codex | Unverified | Not started |
| Gemini | Unverified | Not started |

`JsonLinesImporter` is the first adapter because it is the only shape readable without knowing whose export it is, and because it streams: a line is a record, so a source larger than memory imports in constant space.

Field names are conventions, not a schema. `id`/`uuid`, `role`/`author`, `content`/`text` and so on are ordinary JSON naming; none of it was inferred from a real provider transcript. A source that names things differently passes a `FieldMap`, and a source with extra roles passes a `roles` mapping. Neither case is handled by this file growing a branch per provider.

Roles are exactly the ones `models.enums.Role` already defines, so the adapter cannot quietly invent a vocabulary the rest of the codebase then has to honour. A role outside that set is not an error and is not dropped: the event becomes `OTHER` and keeps its original name.

### `role="tool"` is a convention, not a provider fact

The generic adapter reads a tool-role record as `TOOL_RESULT`. That is the prevailing convention in JSON conversation formats, where a tool-role message carries what a tool returned rather than a request to run one, and it is the reading `Role.TOOL` was defined against. It is a sensible default for a source whose format nobody has verified. It is not a claim about any provider.

**A provider-specific adapter must classify tool events from that provider's verified format, and must not inherit this mapping by default.** Providers differ in ways that matter here: one may put a call and its result in a single record, another may nest a call inside an assistant message, another may use a role name this adapter has never seen. Assuming the generic convention holds is exactly the kind of guess that produces an adapter which looks right and silently mislabels half a transcript.

Where a caller's own source disagrees, the `roles` mapping overrides it — `JsonLinesImporter(roles={"tool": EventType.TOOL_CALL})`. Neither route adds a provider branch to the generic adapter.

**Why not a top-level JSON array.** `json.load` builds the whole document before returning the first element, so a 500MB array costs 500MB of RAM plus the object graph on top, which breaks the constraint this phase is built around. Reading an array incrementally needs a pull parser, and that is a dependency this project does not take before a measurement demands it. An array converts with `jq -c '.[]' export.json > export.jsonl`.

## Streaming

A 500k-token conversation must not need 500k tokens of RAM.

One event is held at a time, from the source line through normalization to `log.append`. The exceptions are two sets used to notice a repeated source id and an unmatched tool result: they hold identifiers, so they grow with the number of events rather than with how much was said, and for any conversation large enough to matter they are orders of magnitude smaller than the text they came from.

Enforced by `tracemalloc` in `tests/integration/test_import_memory.py`, which asserts that peak memory stays far below the source size and does not grow when the source grows eightfold.

The report is bounded too. Counts are exact, samples are capped at 20, and `truncated_samples` says plainly when a list is not the whole story, so a broken 200MB file cannot produce a report as large as the problem it describes.

## Error behaviour

Nothing is discarded silently. Every departure from a clean read is either raised or counted in `ImportReport`.

| Situation | Behaviour |
| --- | --- |
| Malformed record | `MalformedRecordError` with its line, or counted under `on_malformed="skip"` |
| Invalid JSON | A malformed record at that position |
| Truncated input | The final line fails to parse and is reported at its own position |
| Unsupported source format | `UnsupportedSourceError`, before anything is written |
| Not valid UTF-8 | `UnsupportedSourceError` naming the file |
| Missing event id | Counted. The archive record gets a `synth-<source key fingerprint>-<position>` id; the event still records `source_id=None` |
| Duplicate event ids | Counted and sampled. Both records are kept. `on_duplicate="abort"` stops instead |
| Missing timestamp | Counted. Stored as `None`, never as the time of import |
| Unusable timestamp | Counted separately, so "this export has none" is distinguishable from "this export has some we refused" |
| Unknown event type | Stored as `OTHER` with the provider's own name kept and counted |
| Tool result with no matching call | Counted. Never paired by guesswork |
| Tool call or result with no name | Counted |

**`on_malformed="abort"` is the default.** A source this build cannot read is more likely an adapter bug than a broken file, and importing three quarters of a conversation while believing it whole is the worse failure. `"skip"` continues, and the record still appears in the report; there is no setting that drops one quietly. The value is named `abort` rather than `error` because that is what it does: it stops a running import, and what was already written stays written.

**`on_duplicate="report"` is the default.** A repeated id is legitimate: the archive stores a message and a later revision of it as two records distinguished by position. `"abort"` is for a caller who knows their source assigns ids uniquely.

**A naive timestamp is refused rather than guessed.** It could be any timezone, and guessing is how an export produced elsewhere silently reorders history. The original text stays in `raw`.

## Identity

Four concepts, and conflating any two of them produces a wrong id.

| Concept | Is | Is not |
| --- | --- | --- |
| **Source identity** (`source_key`) | Which logical source this is. A caller's judgement | Derivable from bytes |
| **Content digest** | A fingerprint of what a source contains | An identity |
| **Archive sequence** (`seq`) | Where a record sits in one local archive | Stable across archives, or identity |
| **Synthetic event id** | A deterministic encoding of source identity plus position | A provider's id |

**Source identity cannot be computed.** Two byte-identical exports may be one logical source or two, and nothing about the bytes distinguishes those cases: it depends on whether the same conversation was copied or two conversations happened to coincide. Only the caller knows, so `source_key` is theirs to state and nothing here tries to infer it.

An earlier version of this scheme used the content digest as the source key, which got this exactly wrong: two identical files silently became one logical source. Since an archive id means *identity of the underlying item* — two records sharing one are read as the same item written twice — that asserted a relationship that did not exist.

### Synthetic event ids

```
synth-<fingerprint of the source key>-<position within that source>
synth-4f2a91c6b0d3e785a1b2c3d4-00000004
```

The key is hashed rather than embedded so a key of any length, containing whatever characters a caller finds meaningful, still yields a short fixed-width id. The hash encodes the identity; it is never a substitute for it.

| Property | Why it holds |
| --- | --- |
| Same logical source and position give the same id | Nothing else is an input, on any machine or in any archive |
| Distinct logical source keys give distinct ids, collision-resistantly | A truncated SHA-256 of the key. This is a statement about how hard a clash is to find, not a proof that none exists |
| Nothing is random | No UUID, so re-running an import reproduces the ids exactly |
| Nothing depends on `seq` | Otherwise a record's name would change with what else happened to be imported first |
| It never poses as a provider id | The `synth-` prefix says so, and the stored event still records `source_id: null` |

Re-importing one source under one key repeats its ids, which is correct rather than a collision: it is the same source, so it is the same items, which is exactly what a repeated archive id means.

### Where the key comes from

`import_events` **requires** `source_key`. An event whose provider gave no id is identified by it, so a quiet default would be the bug this scheme exists to remove. A live agent stream passes something stable and meaningful, such as a run id.

`import_path` accepts one and otherwise defaults to `default_source_key(path)`:

```
file:/home/dana/export.jsonl#9f2c...e41
```

Read as **"this file, holding this content, at this location"**. Both halves matter. *Location* distinguishes two files that happen to hold identical bytes; they are two files, so by default they are two sources. *Content* means an edited file is a different logical source, because once the bytes change, position 7 no longer names the event it used to.

The consequence is that copying a source to another machine changes its default identity, since a path is local to a machine. That is a default, not a rule, and the escape hatch is the point of the parameter:

```python
# these two copies are the same logical source
import_path(archive, session, "/laptop/export.jsonl", source_key="export:2024-03-01")
import_path(archive, session, "/desktop/export.jsonl", source_key="export:2024-03-01")

# these two ingestions of one file are not
import_path(archive, session, "export.jsonl", source_key="run:1")
import_path(archive, session, "export.jsonl", source_key="run:2")
```

The key used is reported back as `ImportReport.source_key`, so a defaulted identity is still visible. It is not written into the archive; only its fingerprint appears there, inside synthetic ids.

## Imports are not atomic

**An import that fails partway leaves the records it already wrote. This is deliberate and explicit.**

The archive has `append` and `extend` and no delete, no truncate, and no rollback. Nothing in this runtime can take a written record back, and adding a staging layer to fake it would mean writing every record twice, renumbering every `seq` on commit — `seq` is inside the record hash, so a merge is a full re-hash — and holding a second copy on disk, to buy a guarantee a live agent stream could never use anyway. A failure and a crash would then also have two different recovery stories instead of one.

So the contract is the honest one:

| On failure | Guarantee |
| --- | --- |
| Records before the failure | Written, and stay written |
| The record that caused it | Never written |
| Records after it | Not read |
| What the caller is told | `IncompleteImportError` carrying an `ImportReport` with `completed=False`, `events_written`, and `last_seq` |

`IncompleteImportError` is the base of `MalformedRecordError` and `DuplicateSourceIdError`, so a caller can ask "is my archive partial" without enumerating failure types, and cannot catch either specific error without meeting the type whose name says the import stopped partway. Its message ends with how many records were already written.

`UnsupportedSourceError` is different: it is raised before anything is opened for writing, so the destination is untouched.

**Getting atomicity anyway.** Import into a session of its own and check `report.completed`. If it is false, delete the session directory — one `rmtree`, on a store that is a directory per session precisely so this works. That is the local-first answer, and it keeps the transaction where the user can see it instead of inside the runtime.

**Or continue.** A partial session is a valid archive: it verifies, it streams, and a later import appends to it. `appended_to_existing` says the session was not empty when the next import started.

## Failure semantics at a glance

```
UnsupportedSourceError   -> nothing written, destination untouched
IncompleteImportError    -> records before the failure remain; report says how many
  MalformedRecordError   -> a source record could not be read
  DuplicateSourceIdError -> a source id repeated, under on_duplicate="abort"
returns ImportReport     -> the source was read to the end; completed=True
```

`report.completed` and `report.clean` answer different questions. `completed` is "did it finish". `clean` is "is there anything to look at" and is false for a missing timestamp, an unrecognised event kind, or an unmatched tool result — none of which stop an import.

## Re-import

The archive is append-only and has no idea the same conversation arrived twice. Importing a file into a session that already holds records appends a second copy, and `ImportReport.appended_to_existing` says so. A caller wanting one copy imports into a session of its own.

There is deliberately no deduplication. Deciding that two records are the same event is a policy question with no correct answer at ingest, and an import that quietly dropped a record because it resembled an earlier one would be exactly the silent data loss this phase exists to avoid.

## SQLite

Import writes to the archive and nothing else. No message rows, no state items, no evidence.

The archive is the source of truth and the conversation is not duplicated into SQLite. Where a caller wants the two connected, the link is the session id: use one minted by `ids.new_id(ids.SESSION)` for both the archive directory and the SQLite `Session` row, and set `Session.source` to the provider name. That is the minimum metadata that connects an imported conversation to the session and state layer, and it is one row.

Pinned by tests: after an import, `messages`, `state_items`, `evidence`, and `snapshots` are all empty.

## API

```python
from open_context.archive import Archive
from open_context.importers import IncompleteImportError, import_path, stream_events

archive = Archive("~/.open-context")

try:
    report = import_path(archive, session_id, "export.jsonl")
except IncompleteImportError as failure:
    report = failure.report  # completed=False; events_written says what landed

report.completed  # whether the source was read to the end
report.clean  # whether anything needs looking at
report.events_written  # what landed
report.by_type  # counts per event kind

for event in stream_events(archive.open(session_id)):
    ...  # one event at a time, never all of them
```

`import_events(log, results, *, provider, source_key, on_malformed, on_duplicate)` takes events directly, for an adapter that is not file-shaped. `detect_importer(sample)` chooses one.

Identity helpers, kept separate because the concepts are: `default_source_key(path)` builds the identity `import_path` assumes; `content_digest_for_path` and `content_digest_for_bytes` fingerprint content; `source_key_fingerprint(source_key)` encodes an identity for an id; `synthetic_id(source_key, position)` builds the record id.

## Unresolved

**No provider transcript format has been verified.** Claude Code, Codex, and Gemini adapters are not implemented because their transcript formats are not represented in this repository and were not available to inspect. Writing them from memory would produce adapters that parse a format nobody has confirmed exists, fail silently against the real thing, and be hard to distinguish from working code. Each needs its real export or hook output in hand first: where the transcript lives, whether it is a file or an API, whether it is line-delimited, how content blocks are shaped, how tool calls and results are correlated, and whether the format is versioned. Until then the generic adapter plus a `FieldMap` covers a converted export.

**`fsync` per event.** The pipeline appends one event at a time because `SessionLog.extend` writes one `kind` per batch and imported events have mixed kinds. That keeps memory flat and the archive `kind` faithful, at one `fsync` per record. A large import is therefore slower than it needs to be. Fixing it means either an archive API that takes per-record kinds or a configurable durability level, and neither should be designed before a measurement says the cost matters.

**Content blocks are not normalized.** A source whose `content` is a list of typed parts leaves `text` as `None`, and the parts stay in `raw`. Flattening them is provider-specific and belongs in a provider adapter, once there is a real format to flatten.

**Tool correlation is within one import.** Unmatched tool results are detected against calls seen in the same import run, so a conversation split across two files reports results whose calls were in the first file. Correlating across an entire archive would mean a scan per import.

**A partial session is not marked as one.** After an `IncompleteImportError`, the records that landed are ordinary archive records with nothing distinguishing them, so "was this session imported completely" is answerable only by the caller keeping the report. Writing a marker record into the archive would put import bookkeeping inside the conversation, which is worse. If a later phase needs the answer durably, it belongs in SQLite alongside the session row.

**Streaming a source that is not seekable.** `import_path` reads the file up to three times: a sample for detection, a pass for the content digest, and the import itself. That is fine for a file and impossible for a pipe. An adapter reading a live stream uses `import_events` with its own `source_key` and is unaffected, but there is no `import_stream` for a one-shot source yet.

**A source key is not recorded in the archive.** Only its fingerprint is, inside synthetic ids, so an archive cannot say which logical source a record came from — only that two records share one. Recovering the key means keeping the report. If a later phase needs that link durably, it belongs beside the session row in SQLite, along with whether the import completed.

**Nothing links an archive record to a SQLite row.** Still open from Phase 3. An imported event and a future `Message` row would refer to the same thing by id with no enforced correspondence, and no phase has yet defined which writes both.
