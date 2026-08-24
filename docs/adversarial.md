# Adversarial evaluation

Phase 8.5. Dataset `v4`: conversations where compaction actually has to choose.

## Why it exists

The benchmark's job is to distinguish strategies. Through Phases 5.8 and 8 it could not, and neither failure was about compaction.

| Run | Result | What it actually showed |
| --- | --- | --- |
| 5.8, 70B model | Every arm 1.00 on every measure | A ceiling effect, indistinguishable from a measurement |
| 5.8, 8B model | Arms separated by 0.08 | Two runs of the *same* configuration also differed by 0.08 |
| 8, four arms | Hybrid 1.00/0.92 vs baseline 0.92/0.85 | Same 0.08 — the effect and the noise are one number |

The cause is arithmetic. `v3` scenarios are **220 to 385 tokens** across 18 to 32 messages. At budget 160 that is **1.4x to 2.4x compression** — so little pressure that a plain one-call summary keeps essentially everything.

> A benchmark whose conversations nearly fit inside the budget is not measuring compaction. It is measuring whether the model can read.

## What v4 changes

### Length

| | v3 | v4 |
| --- | --- | --- |
| Tokens | 220–385 | **1,400–2,300** |
| Messages | 18–32 | **63–107** |
| Compression at budget 160 | 1.4x–2.4x | **9x–14x** |

At 9x and above, something must genuinely be dropped, and *which* thing a strategy drops becomes visible.

### Filler that resists summarization

`v1`'s filler repeats one sentence with a changing index:

```
Small question 7 about logging formatting.
Answer 7: adjust the logging formatter, keep the existing pattern...
```

A summarizer collapses that to a single line at no cost. It consumes *prompt* space without consuming *summary* space, so it never forces the compactor to choose — which is the entire job of filler in an adversarial scenario.

`v4`'s filler varies across topic, package, and note, stepped on coprime cycles so combinations do not repeat within a scenario.

**An earlier version of this got it wrong**, stepping the note by three through a six-element list — a cycle of two. A test caught it producing 24 distinct messages out of 40, which is the exact flaw the filler was written to avoid. The test remains.

### Position as a controlled variable

Three scenarios plant the *same* encoding requirement at different depths:

| Scenario | Requirement sits at |
| --- | --- |
| `adv-critical-early` | ~4% through |
| `adv-critical-middle` | ~50% through |
| `adv-critical-late` | ~96% through |

They share a task, a check, and a length, and differ only in position. Anything separating them is a position effect rather than a property of the content.

This matters because **"recency wins" and "primacy wins" are different failures with different fixes**, and a dataset that scattered its critical information could not tell them apart. A recent-window strategy should ace the late variant and fail the early one; a summarizer has no such profile. That difference is a signal about mechanism, not just a score.

## The scenarios

| Scenario | The trap |
| --- | --- |
| `adv-critical-early` / `-middle` / `-late` | One stated-once encoding requirement, at three depths |
| `adv-reversed-twice` | A decision made, reversed, then reversed again |
| `adv-exact-values` | Four similar numbers, one of which is the answer |
| `adv-failed-approach` | An approach tried, failed for a stated reason, must not return |
| `adv-tool-heavy` | The answer sits in one tool result among many |
| `adv-negative-constraint` | A prohibition stated once, then buried, then invited by the task |

## What is carried over from v3

**Deterministic checks assert presence; judged questions assert absence.** Nothing in `v4` asserts that a string is missing. That rule exists because the substring form made **eight of ten** Phase 5.8 findings false: a correct answer that names what it rejected fails such a check, and does so more often the better informed it is.

Judged questions are phrased so the desirable answer is YES, and say explicitly that naming a rejected option counts as compliance — the polarity bug found in Phase 5.8, where a question asking "does the response recommend Redis?" scored every correct answer as a failure.

**Two scenarios carry judged questions only.** `adv-failed-approach` and `adv-negative-constraint` state a prohibition and never say what to use instead, so the space of acceptable answers is open and there is no positive assertion to make. Recorded rather than papered over with an invented expected answer.

## Repetitions

Phase 8 established that a single repetition cannot support a comparison: run-to-run variance was the same size as the between-arm gap. `v4` runs are executed with `--repetitions 2` or more, and each repetition is recorded separately — the harness has folded the repetition index into the run fingerprint since Phase 5.6 precisely so two runs of one configuration are never filed under one identity.

**This does not make one run conclusive.** Two repetitions estimate variance; they do not eliminate it. What they do is make the size of the noise visible in the same results file as the size of the effect, which is what Phase 8 had to reconstruct by hand from two separate runs.

## Results

`llama-3.1-8b-instant`, budget 160, 8 scenarios × 4 arms × 3 repetitions,
**96 of 96 cells successful**. Judged questions are excluded (see below), so
these are deterministic checks only — string assertions against the response,
with no model in the loop.

| Arm | rep 1 | rep 2 | rep 3 | spread | pooled |
| --- | --- | --- | --- | --- | --- |
| `full_context` (reference) | 1.000 | 1.000 | 1.000 | **0.000** | **1.000** |
| `phase_5_baseline` | 0.833 | 0.667 | 0.750 | 0.167 | 0.750 |
| `simple_summary_v1` | 0.833 | 0.667 | 0.667 | 0.167 | 0.722 |
| `hybrid_v1` | — | — | — | — | *not measurable, see below* |

**The reference arm scores 1.000 with zero variance across three repetitions.**
This is the control the benchmark never had. Everything the scenarios ask for is
recoverable from the uncompacted conversation, so every point lost below is lost
by compaction rather than by the model — which is precisely what Phases 5.8 and 8
could not establish.

### Compaction costs 25–28% at 9–14× compression

`phase_5_baseline` loses 0.250 against full context and `simple_summary_v1`
loses 0.278. Against a within-arm spread of 0.167 that is a real effect, and it
is the first one this benchmark has produced.

### The baseline still does not beat a plain summary

0.750 against 0.722 — a gap of **0.028**, against noise of **0.167**. The two
are indistinguishable.

This reproduces the Phase 5.8 finding on a dataset six times longer at ten times
the compression, and it is the more informative of the two results. The Phase 5
baseline exists to beat a one-call summary. On the hardest dataset built for it,
it does not.

### All of the loss is in two scenarios

| Scenario | full | summary | baseline |
| --- | --- | --- | --- |
| `adv-critical-early` | 1.00 | 1.00 | 1.00 |
| `adv-critical-middle` | 1.00 | 1.00 | 1.00 |
| `adv-critical-late` | 1.00 | 1.00 | 1.00 |
| `adv-reversed-twice` | 1.00 | 1.00 | 1.00 |
| `adv-exact-values` | 1.00 | **0.00** | **0.50** |
| `adv-tool-heavy` | 1.00 | **0.33** | **0.00** |

Two findings, and the second was not expected.

**What compaction loses is exact values.** The two scenarios built around exact
values are the only two anything fails. This is the Phase 5.8 diagnosis holding
up under far more pressure, and it is now measured rather than inferred from ten
hand-read failures.

**Position did not matter.** The positional trio was the most carefully
constructed part of `v4` — same task, same checks, same length, differing only
in where the requirement sits — and every arm scores 1.00 on all three. At this
budget a one-call summary keeps a stated-once requirement whether it appears at
4%, 50%, or 96% through a 2,300-token conversation. The trio is not wasted: it
is a real negative result that rules out primacy and recency effects as an
explanation for anything else here. A dataset that had scattered its critical
facts would have left that possibility open.

### Hybrid could not be measured, and the reason matters

`HybridCompaction` extracts structured state, and when extraction returns
nothing it falls back to the Phase 5 baseline and warns that the result **must
not be read as hybrid compaction**. On `llama-3.1-8b-instant` that warning fired
on **21 of 24 cells** — extraction produced zero usable items on seven of eight
scenarios.

Read naively, hybrid scored 0.667 and came last. That number was the Phase 5
baseline wearing hybrid's name.

**Phase 8's bet is therefore neither confirmed nor refuted.** Hybrid was built to
carry exact values through in structured state, and `adv-exact-values` and
`adv-tool-heavy` are exactly where the other arms fail — but on the only model
with the daily token allowance to run the full matrix, the extraction step
hybrid depends on does not work at all. What is established is narrower and
still useful: **hybrid compaction requires a model that can do structured
extraction, and an 8B model cannot.** That is a deployment constraint the phase
did not know it had.

The three genuine hybrid cells all came from `adv-reversed-twice`, where every
arm scores 1.00. They discriminate nothing.

### The judge was unusable, and the reference arm proved it

Judged questions are excluded from everything above because the judge failed its
own control. On the reference arm — handed the whole conversation, deterministic
retention 100% — the 8B judge passed **9 of 24** judged questions. On the same
questions and the same arm, the 70B judge passed **8 of 8**.

A representative failure: on `adv-failed-approach` the response opens by naming
the correlated-subquery approach and its six-hour timeout, then proposes four
alternatives and never returns to it. The question asks whether the response
avoids re-proposing it, and says explicitly that naming it only to rule it out
counts as compliance. The judge answered no, reasoning that the approach "is
listed as one of the potential approaches to consider". It is not.

`llama-3.1-8b-instant` is not usable as a judge for negation questions. The
benchmark now checks this automatically: a judge that fails the reference arm
while its deterministic checks pass is reported as unreliable, and its scores
must not be read.

### What this run cost

96 cells, about 320,000 tokens, roughly 100 minutes at 2 requests a minute.

## The first v4 run was void, and it looked like the best result yet

Worth recording in full, because the failure mode is one this benchmark is
structurally prone to.

The run used `groq/openai/gpt-oss-20b` at budget 160. After 26 cells:

| Arm | Combined |
| --- | --- |
| `full_context` | 1.00 |
| `hybrid_v1` | 0.58 |
| `phase_5_baseline` | 0.10 |
| `simple_summary_v1` | 0.09 |

Read as a score it is everything Phase 8 failed to produce: a reference arm at
ceiling, arms separated far beyond the ±0.08 noise floor, and the structured arm
well ahead of the summary arms.

**It was an artifact.** `gpt-oss-20b` is a reasoning model, and Groq bills its
thinking against `max_completion_tokens`. At a cap of 160 it spent 158 tokens
reasoning and returned `content: ""` with `finish_reason: length`. Every
compaction arm was handed an empty context. `full_context` performs no
compaction call, so it alone was untouched — and it is the arm that scored 1.00.

The ranking among the broken arms was not meaningful either: `hybrid_v1` scored
0.58 because its state section is assembled locally and survived, and
`phase_5_baseline` scored 0.10 on its verbatim recent-message floor. They were
ranked by how much of their output does not come from the model.

**Nothing in the scores could have caught it.** The signal was in the token
counts: `context_tokens` was **0** in every `simple_summary_v1` cell while
`compaction_output_tokens` was exactly 160, the cap. The same code against
`llama-3.1-8b-instant` produces 110–136.

The lesson generalises past this one model. **A benchmark that compares a
no-compaction reference against compaction arms will read any systematic failure
of the compaction call as evidence for the reference arm.** That is the shape of
a result this project would like to be true — compaction is lossy, full context
wins — which is exactly the shape that deserves the most scrutiny.

The library now raises rather than returning an empty answer, records reasoning
spend in usage, strips inline `<think>` blocks, and refuses an impossible output
cap before the request is sent. See [llm.md](llm.md). The cells are kept locally
in the git-ignored `benchmarks/results/void/`; this section is the record.

**A check worth repeating on any real run**: confirm `context_tokens` is near
the budget for every compaction arm before reading a single score.

## Cost, measured

Predicted from call counts, then measured against Groq. All-in cost per cell
includes the judge:

| Arm | Calls | Compacted context | Tokens per cell |
| --- | --- | --- | --- |
| `full_context` | 1 | 2,020 (uncompacted) | ~2,700 |
| `simple_summary_v1` | 2 | 124 | ~3,600 |
| `phase_5_baseline` | 2 | 119 | ~3,550 |
| `hybrid_v1` | 3 | 122 | ~3,700 |

The compacted contexts land at 119–124 against a budget of 160, which is the
first thing to check on any real run — the void run's tell was a zero here.

**The full matrix does not fit in a free tier's day.** 96 cells at roughly 4,400
tokens all-in is about 425,000 tokens. `llama-3.3-70b-versatile` allows
**100,000 tokens a day**, a limit that appears in no response header and is
announced only by the 429 that enforces it — so a run sized against the
documented per-minute and per-day *request* limits stops at about 22 cells.
`--resume` and per-cell recording are what make a run larger than one day's
allowance possible at all.

**The cost prediction was wrong about the expensive arm.** `hybrid_v1` was
estimated at ~5,700 tokens per cell on the reasoning that three calls cost three
times one; it measures ~3,700, barely above the two-call arms. The extra call is
extraction, whose *output* is small and whose input it shares with the
summarization call. Call count is a poor proxy for token cost when the calls are
not the same size — worth remembering before sizing a run against it.
