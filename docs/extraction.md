# Structured extraction

Phase 6. The first layer that interprets rather than transports.

```
RAW ARCHIVE
     |
StructuredExtractor          <- this phase
     |
Goal / Constraint / Decision / Task / Fact / OpenQuestion / Entity
```

Import answers *what happened*. Compaction answers *what to keep*. Extraction answers *what it means* — that this sentence is a decision, that one a prohibition, and that the second overturns the first.

Everything before this could be checked against its source mechanically: an imported event either matches the record it came from or it does not. Extraction cannot be checked that way, which is why it reports its own losses and ships a validator.

## What it produces

The Phase 1 data model already had the targets, so this phase adds no record types. It fills them.

| Type | Notes |
| --- | --- |
| `Goal` | |
| `Constraint` | `hard` distinguishes a prohibition from a preference |
| `Fact` | Also where a failed approach lands — what was tried, and why it failed |
| `Decision` | `rationale` required, `alternatives` for options not taken |
| `Task` | With `task_status` |
| `OpenQuestion` | |
| `Entity` | |

`Preference`, `Event`, and `Artifact` exist in the model and are **not** extracted. Nothing downstream consumes them yet, and a type the prompt names is a type the model will produce whether or not it means anything.

## Provenance points at the archive, not at SQLite

This phase resolved an open question the project had carried since Phase 4: nothing linked archive records to SQLite rows.

`StateItemBase.sources` accepts `msg_` and `ev_` ids — SQLite's naming. The archive names records differently: a provider's own id, or a positional `pos-` or `synth-` id the importer minted. No correspondence has ever been established between the two.

**Extraction cites what actually exists.** An `ExtractedItem` carries the archive record ids and sequence numbers it came from, on the wrapper rather than inside the record, and `sources` is left empty.

```python
ExtractedItem(
    item=Constraint(content="Redis must not be used", ...),
    archive_record_ids=("m0",),
    archive_seqs=(0,),
)
```

Minting `msg_`-shaped ids for archive records would assert a correspondence no code maintains, and a provenance reference that resolves to nothing is worse than one that is honestly out of band. Turning archive coordinates into `Evidence` and `Message` rows belongs to whichever phase first writes both stores.

**An item that cites nothing is rejected.** Provenance is the point of the record; a claim nobody can trace is a summary sentence with a type annotation.

## Nothing the model says is trusted

Both shipped providers raise `UnsupportedCapabilityError` for `structured_output` and neither advertises the capability, so JSON is *prompted for* and parsed defensively.

| The model does this | Extraction does this |
| --- | --- |
| Wraps JSON in prose or a markdown fence | Finds the outermost braces and parses that |
| Returns truncated JSON | Records a malformed response — **never repairs it** |
| Invents a type | Counts it in `unknown_types`, rejects the item |
| Omits a decision's rationale | Rejects the item |
| Cites no source, or a source from another session | Counts it in `unattributed`, rejects the item |
| Writes `"high"` where a float was asked for | Uses 0.5 and keeps the item |
| Writes `true` for a confidence | Uses 0.5 — `bool` subclasses `int`, so this would otherwise arrive as maximum confidence |

Truncated JSON is not repaired because guessing what a half-written object meant is how an extraction invents state. A malformed reply is a **recorded failure**, never a silent empty extraction — the two look identical in a results file and mean opposite things.

**The model is shown the record ids it must cite.** The prompt demands provenance, and a model cannot cite what it was never shown; rendering without ids and then rejecting every item for citing nothing would blame the model for the harness's omission.

## What the prompt insists on

Three instructions carry most of the value, and each exists because its absence produces a specific, recognisable failure:

**Negative constraints are recorded as constraints.** "Do not use Redis" is `Constraint(hard=True)`. A prohibition dropped for being phrased negatively is the failure mode the Phase 5.8 dataset was built around.

**A decision requires a rationale.** Where the conversation records a choice but never says why, the rationale is `"not stated in the conversation"` rather than an invention. A decision without its reason cannot be revisited, which is the single question people most often reopen a long session to answer.

**A reversal records both sides.** The earlier item with status `superseded`, the later one `active` and naming what it replaced. Keeping only the winner discards the fact that a reversal happened, which is part of the state rather than noise around it.

Supersession is carried **by content, not by id**: ids are minted after the model has answered, so it could not cite one. Resolving content to an id is supersession bookkeeping, and belongs to Phase 7.

## Validation

Phase 6's exit criterion is that structured context can be validated against original source material, not merely produced. An extractor with no validator is a machine for generating confident claims.

`validate(result, log)` is **deterministic, offline, and calls no model.** A validator that asked a model whether the extraction was right would inherit the failure mode it exists to detect, and would make validity nondeterministic at the margin — the same objection this project already raised against an LLM-based compaction gate.

| Finding | Severity |
| --- | --- |
| `wrong_session` — item scoped to a different session | error |
| `dangling_citation` — cites a record not in this session | error |
| `out_of_range` — cites a position outside the extracted range | error |
| `unattributed` — cites nothing | error |
| `decision_without_rationale` | error |
| `everything_superseded` — no current state left at all | error |
| `weak_grounding` — little of the item's wording appears in its sources | warning |
| `supersedes_unknown_item` | warning |

**`weak_grounding` is a blunt instrument and is named as one.** Word overlap catches an item citing a record that plainly does not mention what it claims; it cannot catch a subtly wrong paraphrase, and it will flag a correct item written in words the source did not use. It is a warning, findings are for a human to read, and nothing is ever deleted on the strength of it.

### What a clean validation does not mean

It checks **grounding** — that nothing asserted is unsupported. It does not check **completeness**, because the conversation carries no list of what should have been found.

So an extraction that returned *nothing at all* validates clean, and correctly so: it asserts nothing unsupported. Recall is what the Phase 5.5 retention scenarios measure, and conflating the two would let an extraction that found one decision out of ten score perfectly here. The clean message says so out loud.

## Limits, recorded rather than worked around

**One model call means the whole conversation is in memory and in the prompt.** A session larger than the model's context window cannot be extracted this way, and the project's rule that a large archive must not need proportional RAM is not met here.

Chunking needs supersession and conflict resolution to reconcile what the chunks each say about the same decision. That machinery is Phase 7's, and building a worse version of it here would mean building it twice.

**Nothing is written anywhere.** Extraction produces records; deciding whether they belong in a database, and reconciling them with what is already stored, is Phase 7's. Keeping the two apart means an extraction can be run, inspected, and thrown away without touching state.

**No incremental extraction.** `first_seq` and `last_seq` are recorded so a later run can tell what has already been read, but nothing consumes that yet.
