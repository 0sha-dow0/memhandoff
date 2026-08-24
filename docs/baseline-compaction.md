# Baseline compaction

Phase 5. The first compaction engine.

> **Phase 5 is a baseline intended for measurement. It is not the final MemHandoff compaction algorithm.**

It is roughly what a competent developer would build in an afternoon: summarize the older part of a session with a model, keep the most recent part verbatim, and hold the whole thing to a token budget. That is the point. It is the control that later work has to beat, and the places where it loses a constraint stated once in message 40 are the evidence that later work is needed.

## What it answers

> If we take a long conversation, summarize the older portion and preserve the most recent portion, how well can another agent continue the work?

It is the comparison point for four things: full original context, ordinary LLM summarization, retrieval, and future MemHandoff compaction. Its weaknesses are research evidence, not defects to patch here.

## What it is not

No structured extraction, no importance scoring, no retention model, no retrieval, no embeddings, no graph, no contradiction resolution, no temporal reasoning, no portable package. A conversation containing "we chose Postgres because the infra already runs it" produces a summary that says so, and never a `Decision` record. Turning prose into typed state is Phase 6.

The result is an **internal experimental representation**. It is not `.ctx`, not a portable context package, and not a MemHandoff package. Serializing one for a test or a debugging session is fine; treating one as a durable interchange format is not.

## Strategy

```
                  full conversation
                          |
              +-----------+-----------+
              |                       |
        older history            recent window
              |                       |
         LLM summary            kept verbatim
              |                       |
              +-----------+-----------+
                          |
                    measure, enforce
                          |
                  baseline context
```

`BaselineRecentPlusSummary`, strategy version 1.

`CompactionStrategy` is a three-member protocol — `name`, `version`, `compact` — so a second strategy is a second class. There is no factory, no registry, and no plugin loading.

```python
strategy = BaselineRecentPlusSummary(provider, tokenizer)
result = strategy.compact(CompactionRequest(log=log, target_tokens=10_000))
```

The request carries the `SessionLog` rather than a session id and a way to find it, because a log already knows its own session and passing both creates two things that can disagree.

## Pipeline

```
allocate budget            target too small -> TargetTooSmallError
        |
count the conversation     streamed, one event at a time
        |
   fits target? ---- yes -> return everything, no model call
        |
        no
        |
select recent window       backward positional reads, whole events
        |
chunk history              sized against the model's context window
        |
summarize                  one call, or chunks then a combining pass
        |
fit summary to its budget  retry once, then deterministic cut
        |
enforce target             keep the recent window, reduce the summary
        |
     result
```

Every number is measured. The model is asked for a size and is not believed: the summary is tokenized, compared to its budget, and cut if it overshot. The same applies to the combined output against the target.

The historical summary covers only the historical range. Recent events are kept independently. Therefore dropping a recent event is genuine information loss and is treated only as an emergency fallback.

## Token budget

The target is a hard output constraint, enforced against a measured count rather than a request to the model.

| Setting | Default | Meaning |
| --- | --- | --- |
| `recent_fraction` | `0.40` | Share of the target spent on verbatim recent events |
| `minimum_target_tokens` | `50` | Below this, compaction is refused |
| `prompt_reserve_tokens` | `512` | Room set aside for the instruction prompt |
| `context_fill_fraction` | `0.85` | How much of a context window one request may occupy |
| `summary_retry_fraction` | `0.70` | How much smaller to ask for after an overshoot |

A target of 10,000 allocates 4,000 to recent context and 6,000 to the summary. Both halves are guaranteed at least one token. Allocation is fixed and computed once: there is no adaptive allocation, no importance model, and nothing that learns, because a baseline whose allocation moved by itself would be impossible to compare against. The actual allocation is on every result.

**Small targets.** 50, 100, 500, and 1,000 all allocate and all produce a result within budget. Below `minimum_target_tokens` the compaction raises `TargetTooSmallError` rather than returning something misleading: at a few dozen tokens the output is neither a usable summary nor a usable recent window, and returning one anyway would imply it worked.

## Recent window

Selected by token budget, walking backward from the end and taking whole events while they fit.

"The last twenty messages" is twenty tokens or twenty thousand depending on what those messages were, which makes the output size unpredictable and the baseline unmeasurable. So the rule is the budget, not the count.

**Whole events only.** A half-rendered tool result is not something another agent can act on, and splitting one would make the window's contents depend on tokenizer internals. The cost is that the window rarely fills its budget exactly.

**An event larger than the whole recent budget leaves the window empty.** Deterministic and documented: the event stays in the historical portion and is summarized there, so nothing is lost, and a warning says what happened. Truncating it would put a fragment of a message in the one place reserved for verbatim material.

## Historical summary

Everything before the window is summarized. The recent window is not included in the summarization request — the baseline stays explainable by keeping the two halves disjoint.

Events are rendered as labelled blocks:

```
[user_message] We need to ingest 40GB of CSV daily.
[tool_call name=psql] {"arguments":{"sql":"SELECT 1"},"tool_name":"psql"}
[tool_result name=psql] 1 row
```

An event whose adapter found no text falls back to its `raw` record, which is where a tool call keeps its arguments. Dropping it would silently remove work the agent did.

Rendering is not interpretation. A tool result that looks like noise renders exactly like one that turns out to matter, because telling them apart is the research problem.

## Prompt

`BASELINE_SUMMARY_V1`, in `compaction/prompts.py`. One prompt, used for both the single-pass summary and the pass that combines chunk summaries; only the framing of the material differs, and that lives in the user message.

It tells the model that a different agent will continue the work and will not see the original conversation, and asks it to preserve the objective, decisions and their reasoning, constraints, completed and unresolved work, failed approaches, and expensive-to-rediscover specifics — while not inventing anything and saying plainly where the conversation left something unclear.

Every result records `prompt_id` (`baseline_summary_v1`) and `prompt_hash`, a fingerprint of the prompt text. A benchmark comparing two runs can therefore tell whether the prompt changed underneath it, even if the version number did not.

## Context-window handling

Before sending history to the model, the available room is computed from `ModelInfo.context_window`:

```
available = window * context_fill_fraction - prompt_reserve_tokens - summary_budget
```

The fill fraction is below one because our count of the input is not the provider's count of it, and the margin absorbs that disagreement.

| Situation | Behaviour |
| --- | --- |
| History fits | One call |
| History does not fit | Partition into chunks that fit, summarize each, then one pass combining the chunk summaries. Records `multi_pass_summary` |
| Combined chunk summaries do not fit | Deterministically cut before the combining pass. Records `chunk_summaries_truncated` |
| Window unknown | One request with no size check, and a warning saying so |

**Two levels, never more.** This is a baseline limitation, recorded rather than engineered around. A conversation whose chunk summaries alone overflow the window gets a truncated combining pass instead of a third level. Hierarchical summarization is a design decision that deserves its own evidence.

## Output budget enforcement

**The summary against its own budget.** Tokenize; if it fits, accept. If not, retry once asking for `summary_retry_fraction` of the budget, then tokenize again. If it still does not fit, cut deterministically. One retry, not a loop: a model that overshot once usually undershoots when asked for less, and an iterative search would spend calls chasing a number a deterministic cut reaches immediately.

**The combination against the target.** The recent window is preserved and the summary gives way.

The historical summary covers only the historical range. Recent events are kept independently. Therefore dropping a recent event is genuine information loss and is treated only as an emergency fallback.

The two halves cover **disjoint** ranges. The summary is generated from `archive[0:split)` and the window holds `archive[split:end)`. An event in the window was never sent to the summarizer, so it is not condensed anywhere — dropping it removes it from the output entirely.

That makes the order:

1. If summary + recent fits the target, done.
2. Otherwise reduce the summary to the room the window leaves: regenerate it once at that smaller size, then cut deterministically if that was not enough. Since fitting a summary to a budget always succeeds, this always brings the total inside the target.
3. Emergency only, when the recent window *alone* exceeds the target: no summary of any size can help, so the summary is dropped and recent events are dropped from the oldest end until the window fits.

**Step 3 is genuine information loss and is not reachable on the normal path.** Allocation guarantees `recent_budget + summary_budget <= target`, and the window is selected against a budget strictly smaller than the target, so the window alone always fits. The branch exists so the hard output constraint survives an accounting disagreement between selection and enforcement, not as a budget strategy. A sweep of 198 target and split configurations drops nothing.

When it does fire, the result says so rather than hiding it: `recent_events_dropped` in `fallbacks`, an explicit warning that dropped events are absent from the output entirely, and `dropped_recent_event_count`. `historical_range` continues to report the range that was actually summarized, so it and `recent_range` stop meeting — **the gap between them is exactly what was lost**, and it is visible rather than inferred.

> An earlier version of this engine had the order backwards, dropping recent events first on the reasoning that "the summary already covers them". It does not. That claim is false for every event in the window, and acting on it would have silently deleted conversation.

**Truncation** is a binary search over word counts, at most `log2(words)` tokenizer calls, cutting at word boundaries so the result does not end mid-token. A `[...truncated]` marker is appended and is counted inside the budget, not added to it, so a reader can see the text was cut.

| Fallback | Meaning | Loses |
| --- | --- | --- |
| `summary_retried_smaller` | The summary overshot and was requested again | Detail |
| `summary_truncated` | The summary was cut deterministically | Detail, from the tail |
| `multi_pass_summary` | History needed chunking | Detail, through a second condensation |
| `chunk_summaries_truncated` | Intermediate summaries were cut before combining | Detail |
| `recent_events_dropped` | Emergency only: whole events, gone from the output | **Whole events, represented nowhere** |

Only the last is information loss rather than loss of detail, which is why it is last and why it is warned about.

## Token counting

Counts come from the Phase 4.5 `Tokenizer`, and an estimate is never presented as exact.

| Tokenizer state | Behaviour |
| --- | --- |
| Exact | Counts recorded with `exact=True` |
| Estimated | Counts recorded with `exact=False`, and a warning saying the budget was enforced against an estimate |
| Unavailable | `TokenizationUnavailableError` propagates |

There is no fallback for an unavailable tokenizer. Every budget decision here depends on measuring, so a compaction that could not measure has nothing to fall back to, and inventing a count would defeat the point of the abstraction.

No safety margin is applied when counts are estimated. The measured number is used and the estimate is flagged, because a margin would be an invented constant that a benchmark could not attribute. Whether one is needed is an open question below.

Counts are taken per event and summed. That matches the tokenizers in this repository exactly; a tokenizer applying a conversation-level template prefix would count a whole sequence slightly differently. Recorded below.

## Conversation below target

If the input already fits, the conversation is returned whole with `compaction_needed=False` and **no model call**. Summarizing something that already fits would spend a call to make it worse. This is also the control case a benchmark needs — full original context, produced by the same code path.

| Case | Behaviour |
| --- | --- |
| Empty archive | Empty result, no model call |
| One message | Returned whole, no model call |
| Below target | Returned whole, no model call |
| Exactly at target | Returned whole, no model call |
| Slightly over target | Compacted, one model call |

## Memory

**The baseline is not O(1), and does not claim to be.**

| Operation | Holds |
| --- | --- |
| Counting the conversation | One event |
| Selecting the recent window | The window only, one positional read per event kept |
| Summarizing | One chunk, bounded by the model's context window |
| Below-target passthrough | The whole conversation, which by definition fits the target |

The requirement is that the archive is never loaded merely to determine the recent window or basic metadata, and it is not: window selection reads backward from the end using the archive's offset index, so it costs one read per event kept and nothing per event skipped. Assembling material for a model request necessarily holds that material, which is bounded by the context window rather than by the conversation.

Pinned by tests: `tracemalloc` on the counting pass, and a counting wrapper asserting that window selection performs fewer reads than the archive has events.

## Immutability

**Compaction never modifies the source archive.** The module contains no call that appends, truncates, or rebuilds anything; the result is a separate object.

Pinned by hashing every file in the session directory before and after compaction, including after repeated compactions and after a failed one, plus `verify()` on the reopened log.

## Observability

Every result carries what a benchmark needs to attribute it:

strategy and version, prompt id and hash, full configuration, session id, target, allocated recent and summary budgets, input/output/recent/summary counts each with `exact` and `method`, archive/historical/recent event counts, historical and recent archive ranges, recent event ids, `compaction_needed`, fallbacks, warnings, tokenizer identity, model provider/model/context window, LLM call count, provider-reported usage, and elapsed time.

**Provenance is coarse on purpose.** Session id, the ranges considered, and the recent event ids are enough to know what went in. Fine-grained evidence links from a summary sentence back to a message are a later phase.

**The ranges are exhaustive unless something was lost.** `historical_range` and `recent_range` meet, and together they cover the archive. When `dropped_recent_event_count` is non-zero they stop meeting, and the gap is what is represented nowhere. Checking that the two ranges meet is therefore a check that nothing was lost.

## Reproducibility

Same input, same configuration, same fake provider gives the same result. Nothing here is random. `elapsed_seconds` is the one field that will not reproduce; the determinism test excludes it.

A real provider is nondeterministic, and a result produced with one records the model and provider that produced it.

## Metrics

```
compression_ratio = input_tokens / output_tokens
retention_ratio   = output_tokens / input_tokens
```

Zero is handled: no output gives `inf`, an empty conversation gives `1.0`.

**Neither is a quality measure.** A summary that discards the one constraint that mattered compresses beautifully. 10x compression does not mean 10x better, and whether compression cost anything is exactly what Phase 5.5 exists to find out.

## Limitations

Recorded, not solved.

- **Generic summarization.** Nothing targets decisions, constraints, or failed approaches specifically. Whether a generic summary preserves them is the open question.
- **Two-level summarization only.** Deep histories get one chunking pass and one combining pass, then truncation.
- **The 60/40 split is a guess.** Not tuned against anything.
- **Recent means recent, not important.** An important message just outside the window survives only through the summary.
- **Summary truncation is positional.** Cutting the tail assumes the end matters least, which a model writing a summary has no reason to honour. Every enforcement step above the emergency one spends this, so a tight target degrades the summary from its end.
- **The emergency fallback would delete conversation.** Unreachable on the normal path and warned about when taken, but it is a real branch, and a future accounting change could make it reachable.
- **Per-event token counting** may differ slightly from a template-aware tokenizer counting a whole sequence.
- **No safety margin on estimated counts.** The estimate is used and flagged.
- **One prompt, unmeasured.** `BASELINE_SUMMARY_V1` was written, not evaluated.

## Questions for Phase 5.5

The baseline exists to make these answerable.

- Does generic summarization preserve decisions? Their rationale?
- Does it preserve negative constraints — the things that must *not* happen?
- Does it preserve failed approaches, or does "we tried X and it did not work" vanish as unimportant?
- Does it preserve exact numbers, versions, paths, and identifiers?
- Does it preserve temporal state — what is true *now* versus what was true in March?
- Does it preserve artifact relationships?
- Does it preserve important early information, stated once and never repeated?
- How much does recent-window preservation actually help continuation?
- How much compression is achievable before continuation quality degrades?
- How does all of this change with conversation length?
- Does the 60/40 split matter, and in which direction?
- Do estimated token counts need a safety margin, and how large?
