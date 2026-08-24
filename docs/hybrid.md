# Hybrid compaction

Phase 8. Structured state, plus a verbatim recent window, plus a historical summary, under one budget.

```
RAW ARCHIVE
     |
 +---+--------------+
 |   |              |
STATE RECENT     HISTORY
 |   |              |
 +---+--------------+
     |
BUDGET ALLOCATION
     |
PORTABLE CONTEXT
```

## It exists to fix one measured failure

Not to be more sophisticated. Phase 5.8 ran a complete matrix and found two things worth building on:

1. The Phase 5 baseline scored **identically** to a plain one-call summary.
2. What compaction actually lost was **exact values** — a port number among several similar ports, a row count reported by a tool. Full context lost neither.

Structured state is the mechanism that can carry an exact value through compaction intact, because an extracted `Fact` keeps the number as written instead of hoping a summarizer chose to repeat it.

So the bet is narrow and falsifiable: **state preserves exact values that prose drops.** If the adversarial benchmark says otherwise, the answer is to delete this, not to add another layer to it.

## Where it lives, and why not in `compaction`

Beside `BaselineRecentPlusSummary` would be the obvious home, and it produced an import cycle: `compaction` would import `extraction`, which already imports `compaction` for its prompt and rendering types.

The cycle was the design telling the truth. `compaction` and `extraction` are peers; this composes both. `open_context.hybrid` sits above them, depends on both, and is depended on by neither.

## Priority is a fixed order, not a score

```
GOAL -> CONSTRAINT -> FACT -> DECISION -> TASK -> OPEN_QUESTION -> ENTITY
```

When the budget bites, items are dropped from the **end**. Truncating the rendered text instead would cut mid-sentence, and a truncated constraint can read as its own opposite.

An importance model that learned or guessed a ranking would need its own evaluation before anything built on it could be trusted, and there is no evidence yet that would justify one.

### The first ordering was wrong, and running it is what showed that

Facts were second from last, on the reasoning that *a fact without the decision it supports is a number with no reason to be trusted*. That is a fine sentence and it was the wrong call.

Against a real model on `confusable-numbers` — the scenario the Phase 5 baseline lost the port on — extraction found the fact carrying the port numbers, and the budget then dropped it, keeping a goal, a decision about formatting, and a task:

```
GOALS      - Create a firewall rule for the metrics endpoint.
DECISIONS  - Adjust the deployment formatter... (because: not stated in the conversation)
OPEN WORK  - Write up the firewall change for the metrics endpoint.

contains 8082: False
```

The phase lost precisely the thing it was built to preserve. With facts moved to third:

| Scenario | Value | Baseline | Hybrid |
| --- | --- | --- | --- |
| `confusable-numbers` | `8082` | ✗ | ✓ |
| `exact-value` | `17` | ✗ | ✓ |

**The lesson generalises past this list.** A priority order argued from plausibility will quietly contradict the measurement that motivated the work. The order is now justified by the measurement, and the test that pins it says why.

## What it costs

An extraction call **on top of** a summarization call — two per context, against the baseline's one. It is deliberately the most expensive arm.

Phase 5.8 already found compaction spending more tokens than sending the whole conversation at these lengths. This spends more still. The extraction call is added to `result.model.llm_calls` rather than left out, because a hybrid result reporting only the summarization call would understate its bill by exactly the thing the phase added.

That is why the exit criterion is a **measured** improvement rather than a plausible one.

## Budget

`state_fraction` (default 0.35) is reserved for state before anything else is allocated. The remainder goes to the Phase 5 baseline's own split between recent verbatim events and historical summary — this phase does not reimplement a decision Phase 5 already made and measured.

**The state share is capped so it cannot starve the rest below the baseline's own floor.** Without the cap, a small target raises `TargetTooSmallError` from inside the baseline, which reads as the target being too small when in fact the split was.

## Honesty about degradation

**A failed extraction says so.** When the model's reply is unusable, the result is the Phase 5 baseline with an empty state section, and it carries a warning saying it should not be read as hybrid compaction. Silently degrading would make a benchmark row a lie: an arm that is sometimes hybrid and sometimes not, averaged together under one name.

**Dropped state is warned about**, with the counts of what was extracted and what survived the budget recorded in the result's configuration.

**Superseded items are not rendered.** This shows state as it now stands; a reader given both sides of a reversal without being told which won would be worse off than one given neither. The superseded record still exists — Phase 7 keeps it — it just does not go into the context.

## What this phase does not do

No importance scoring, no retrieval, no evidence selection beyond what extraction already attached, and no reordering by relevance to the task. Each would need its own evaluation, and the benchmark that would justify one is Phase 8.5.
