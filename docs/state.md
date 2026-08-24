# State and retention

Phase 7. What makes extracted state survive a long session rather than being rebuilt from scratch every time it changes.

```
OLD STATE  +  NEW EVENTS  ->  UPDATED STATE
```

Phase 6 turns a conversation into typed records. It does it once, over a whole session, producing state that is correct and immediately stale. This phase is what lets the second hour of a session cost the second hour rather than the whole thing again.

## Incremental update

**The point is cost, not speed.** Re-extracting a long session on every update means sending the whole conversation to a model again — the expense the project exists to avoid. A run that reprocessed everything would be paying compaction's bill to do compaction's job.

A `Watermark` records the first position not yet read.

```python
update = extract_incremental(extractor, log, watermark, known_state=current)
```

### A watermark has to be earned

It advances **only when an extraction over that range actually succeeded**. One moved on a malformed reply would silently skip a region of the conversation forever, and nothing downstream could tell: the state would simply be missing whatever those events said, with no record that they were never read.

It also never moves backward. Re-extracting an earlier range is allowed — it is how a bad extraction is redone — but rewinding the record of what has been read would make everything after it be extracted twice and reconciled against itself.

**An extraction that found nothing still advances it.** Reading a range and finding no state is success. Conflating that with failure would make an uneventful stretch of conversation be read again on every pass.

**Nothing new calls no model**, and returns `result=None` rather than an empty result. Those mean different things — "nothing was read" against "a range was read and yielded nothing" — and a caller that could not tell them apart could not tell a quiet session from a broken extractor.

### Old state is an input, not just an output

`known_state` is passed *into* extraction, and this is the part that is easy to leave out. The first real incremental run did leave it out, and produced this:

```
PASS 2: read 1 of 3 events
   constraint  Do not use PostgreSQL
   decision    Use SQLite instead of PostgreSQL

RECONCILE: 2 new, 0 unchanged, 0 superseding
```

Both items correct, and no link to the decision they replaced. A model shown only the new messages has never seen the earlier decision, so it cannot name it, and the supersession claim it would need to make is unavailable to it. The reversal became two orphaned assertions sitting beside a still-active PostgreSQL decision.

With current state in the prompt, the same conversation produces:

```
PASS 2: read 1 of 3 events
   decision    The service should use SQLite...  [supersedes: The service should use PostgreSQL...]

RECONCILE: 0 new, 0 unchanged, 1 superseding
   RETIRED  'The service should use PostgreSQL...' -> superseded
   CURRENT  'The service should use SQLite...'     -> supersedes it
```

Existing content is reproduced **verbatim** in the prompt, because supersession is matched literally. A model shown a paraphrase emits a paraphrase, and the claim resolves to nothing.

## Reconciliation

`reconcile(existing, incoming)` returns a **plan, not an effect**. Nothing is written; a caller inspects it, resolves anything it cannot accept, and only then commits. Returning a plan is what lets a conflicting update be refused without having already been half-applied.

| Bucket | Meaning |
| --- | --- |
| `added` | Genuinely new items |
| `unchanged` | Already present with identical content |
| `supersedes` | `(existing id, replacement)` pairs |
| `conflicts` | Contradictions, with `blocked` for those needing a decision |

**Identity is type plus normalised content.** Not the id, which is minted per extraction and would make every re-run look like new state; not confidence or provenance, which can differ between two readings of one sentence without the sentence having changed. Crude — a re-extraction that rewords an item slightly reads as a new one — and the honest deterministic answer to a question whose better answers are all language questions.

**An item claiming a change is never filed as unchanged**, whatever its content matches. It is asserting a reversal, and filing it as a duplicate would discard the assertion.

### Nothing is deleted

`apply_supersession` returns both records: the old one marked `SUPERSEDED`, keeping its content, rationale, sources, and timestamps; the new one with `supersedes` pointing at it. A retired decision that lost its rationale would defeat the reason for keeping it at all.

This is why append-only storage underneath is a constraint the design wanted rather than one it works around.

## Conflicts

**Detection is deterministic and deliberately narrow.** Deciding whether two sentences contradict each other is a language question, and answering it with a model would make state acceptance nondeterministic — a worse version of the objection this project already raised against an LLM-based compaction gate, because here the output is not a report but the state itself.

| Kind | Found by |
| --- | --- |
| `contradictory_fact` | Two active facts sharing a `subject` and disagreeing |
| `supersession_fork` | Two items claiming the same predecessor |
| `supersedes_missing` | A claim to replace something not in the state |
| `duplicate_active` | The same content active twice |

Facts **without** a subject are not compared. Comparing arbitrary sentences is exactly the language question this refuses to guess at, and a heuristic that tried would flag every pair of sentences about the same topic.

A superseded item contradicting a current one is **not** a conflict. That is the record of a decision having changed, which supersession exists to preserve.

### Only duplicates auto-resolve

Everything else is a choice between two things a person may need to see. `Reconciliation.blocked` lists them and `safe` is false while any remain. Silently keeping the newer item would be a policy nobody chose, and it would be indistinguishable from having found no conflict at all.

`supersedes_missing` keeps the item anyway. Dropping it would lose the fact that the model saw a reversal; applying it would silently mark nothing. Surfacing both is the only option that loses neither.

## What this phase does not do

**No storage integration.** Reconciliation produces a plan against records held in memory; writing it through `Repository` is not wired up. The repository already enforces the invariants a commit would need — append-only, acyclic supersession, session-local references — so this is an assembly job rather than a design one.

**No snapshots or reconstruction-at-a-point.** `ContextSnapshot` exists from Phase 2 and nothing here creates one.

**No conflict resolution UI or policy.** Conflicts are values. What a caller does with a blocking one is undefined here, deliberately.
