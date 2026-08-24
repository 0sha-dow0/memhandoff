# Benchmark

Phase 5.6. The first benchmark of MemHandoff's baselines.

> **These benchmark results establish baseline behavior. They do not yet demonstrate that MemHandoff is superior to existing approaches.**

There is no MemHandoff algorithm yet. This phase measures what exists — full context, a plain summary, and the Phase 5 compactor — so that whatever comes next has something real to be measured against.

## Status of results in this repository

**The current result is Phase 8.5, on dataset `v4`.** Everything below it is kept
as the record of how the benchmark got there; where the two disagree, `v4` is the
one with a working control. Full write-up in [adversarial.md](adversarial.md).

Dataset `v4`, budget 160, `groq/llama-3.1-8b-instant`, 8 scenarios × 4 arms ×
**3 repetitions**, 96/96 cells. Deterministic checks only — the judge failed its
own control in this run and its scores are void.

| Strategy | Cells | Deterministic | Spread over 3 reps |
| --- | --- | --- | --- |
| `full_context` (reference) | 24 | **1.000** | **0.000** |
| `phase_5_baseline` @160 | 24 | 0.750 | 0.167 |
| `simple_summary_v1` @160 | 24 | 0.722 | 0.167 |
| `hybrid_v1` @160 | 3 of 24 | not measurable | — |

**The reference arm is perfect and stable**, which is what the earlier runs
lacked: the information is demonstrably recoverable, so every point lost is lost
by compaction. Compaction costs 25–28% at 9–14x compression against a 0.167
spread — the first effect this benchmark has resolved. The Phase 5 baseline still
does not beat a plain summary. `hybrid_v1` disowned 21 of its 24 cells because
extraction returned nothing on an 8B model.

### Earlier runs

**A complete real-model benchmark has been executed.** Dataset `v3`, budget 160, `groq/llama-3.1-8b-instant`, judged, 45/45 cells with all three arms on all 15 scenarios. `check_comparable` reports no problems: this is a valid comparison.

| Strategy | Runs | Retention | Completion | Judged | Context | Compression | Total tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full_context` | 15 | **1.00** | **1.00** | 10/14 | 297 | 1.0x | **705** |
| `simple_summary_v1` @160 | 15 | 0.92 | 0.92 | 8/15 | 120 | 2.5x | 1222 |
| `phase_5_baseline` @160 | 15 | 0.92 | 0.92 | 5/14 | 115 | 2.6x | 1306 |

### Phase 8: four arms, and a variance measurement that invalidates the comparison

A second complete matrix added `hybrid_v1`. 60/60 cells, zero failures, `check_comparable` clean.

| Strategy | Runs | Retention | Completion | Context | Total tokens | Calls |
| --- | --- | --- | --- | --- | --- | --- |
| `full_context` | 15 | 1.00 | 0.92 | 297 | 719 | 0 |
| `hybrid_v1` @160 | 15 | 1.00 | 0.92 | 130 | 1299 | 2 |
| `simple_summary_v1` @160 | 15 | **1.00** | **1.00** | 124 | 1244 | 1 |
| `phase_5_baseline` @160 | 15 | 0.92 | 0.85 | 114 | 1278 | 1 |

Read naively, hybrid beats the Phase 5 baseline on both measures and Phase 8's exit criterion is met.

**That reading is wrong, and the run before it proves so.** The same configuration — same model, dataset, budget, prompt, judge — had already been run once. Comparing the two:

| Arm | Run A | Run B | Δ retention | Δ completion |
| --- | --- | --- | --- | --- |
| `full_context` | 1.00 / 1.00 | 1.00 / 0.92 | +0.00 | **−0.08** |
| `simple_summary_v1` | 0.92 / 0.92 | 1.00 / 1.00 | **+0.08** | **+0.08** |
| `phase_5_baseline` | 0.92 / 0.92 | 0.92 / 0.85 | +0.00 | **−0.08** |

**Run-to-run variance is ±0.08. The gap between hybrid and the baseline is 0.08.** With 15 scenarios of which 13 are evaluable, one scenario changing its answer moves a score by 0.077 — exactly the observed delta, in both the noise and the "effect".

So the differences between arms are indistinguishable from the differences between two runs of the *same* arm. **Phase 8's exit criterion is not met**: not because hybrid is worse, but because this benchmark cannot resolve a difference of this size.

The deterministic failure counts say the same thing more starkly — `simple_summary_v1` 0, `full_context` 1, `hybrid_v1` 1, `phase_5_baseline` 2, out of 15 scenarios each. Four numbers that small support no ranking.

**The cheapest arm scored best.** `simple_summary_v1` is one model call and took the only perfect 1.00/1.00. Whether that survives repetition is exactly what is unmeasured.

### What hybrid's single failure was

```
response: "Port 8082."
retention checks: all passed
failed: metrics-port — insubstantial, 2 words, fewer than the 12 required
```

It produced the exact value the phase exists to preserve, and failed the substance floor. Whether that is an artefact is genuinely arguable: the task asked to *write the firewall rule, naming the port*, and `"Port 8082."` names the port without writing a rule. Recorded rather than resolved.

### What Phase 8 does establish

Not a score. A **mechanism**: against `gpt-oss-120b`, hybrid preserved `8082` and `17` on the two scenarios where the Phase 5 baseline lost both. That is a direct observation of the state section carrying an exact value through compaction, and it does not depend on n or on variance.

It is also **model-dependent** — the demonstration used a 120B model and the benchmark an 8B one — which is itself worth knowing before anyone reads a compaction result as a property of the algorithm alone.

### The Phase 5 baseline does not beat a plain summary

This is what Phase 5 was built to find out. It called itself "the control that later work has to beat", and against a plain one-call summarization it scores **identically** on the deterministic metrics, **worse** on judged questions, and costs **more tokens** for a context of the same size.

Nothing here says the baseline is bad. It says the extra machinery — a preserved verbatim recent window, measured budget enforcement, chunk-then-combine summarization — bought nothing measurable at this scale over asking a model once for a summary. That is a finding about the problem, not a defect to patch: it means the interesting differences live somewhere this benchmark does not yet reach.

Treat the judged spread with care. n=15, one repetition, and a nondeterministic judge. The honest reading is *no evidence the baseline beats a plain summary, and some evidence it trails* — not a demonstration that it is worse.

### Compaction costs more than it saves here

705 tokens end to end for full context against 1222 and 1306 for the compacted arms, for a context 2.5x smaller. The summarization call is not free, and at these conversation lengths it outweighs what it saves. A compression ratio quoted without a token bill is not a result.

This does not generalise to long sessions, where the trade reverses. It does mean the regime where compaction pays for itself starts somewhere above 400 tokens of conversation, and nothing here has located it.

### What compaction actually lost

Only two deterministic checks failed across the whole matrix, both on compacted arms, and both are the same kind of loss:

| Scenario | Arm | Lost |
| --- | --- | --- |
| `confusable-numbers` | `phase_5_baseline` | the metrics port, among several similar port numbers |
| `tool-result` | `simple_summary_v1` | an exact row count reported by a tool |

**Exact values are what compaction drops.** Full context lost neither. Both survive in the original conversation and are gone from the summary — a specific, reproducible failure mode, and a better target for a next algorithm than "retention was 0.92".

The judged questions add three more differential losses, where both compacted arms fail and full context passes: `open-task`, `similar-entities`, and `temporal-state`. Judged failures shared by *all three* arms — `exact-value`, `failed-approach`, `multi-step`, `rationale` — say something about the model or the question, not about compaction, and are not counted as strategy failures.

### On the 70B run

An earlier complete matrix on `llama-3.3-70b-versatile` scored **1.00 on every metric for every arm**. That is a ceiling effect, not a result: at 2.5x compression on 15-turn synthetic conversations, a capable model recovers everything from a plain summary, and the benchmark cannot discriminate. Dropping to an 8B model is what made the differences visible.

That is itself worth recording. **A benchmark's discriminating power depends on the model under test**, and a strategy comparison run against a model strong enough to ace every arm measures nothing while looking like it measured everything.

## The question

> Given the same original conversation and the same downstream task, how well does each context representation let another agent continue the work?

```
SCENARIO + REPETITION
        |
        +----------------------+
        |                      |
FULL_CONTEXT              COMPACTED ARMS
(reference, once)              |
                         +-----+-----+
                         |           |
                     SIMPLE       PHASE 5
                         |           |
                     80/160/240  80/160/240
                         |
                SAME DOWNSTREAM TASK, SAME MODEL, SAME PROMPT
                         |
                      METRICS
```

Only the context varies. The task, prompt, model, and tokenizer are held identical, so a difference belongs to the representation. `check_comparable` reports any results set where that stopped being true.

## The reference is budget-independent

**Full context ignores the target budget, so it runs once per scenario and repetition — not once per budget.**

Running it at every budget would take three independent samples of one condition and then average them into a row labelled budget-independent. Against a deterministic fake that is merely wasteful; against a real model it is three different draws wearing one label, which is not a reference at all.

So the matrix is:

```
scenarios x repetitions x (references + compacted x budgets)
```

For the default — 15 scenarios, 1 repetition, 1 reference arm, 2 compacted arms, 3 budgets — that is `15 x 1 x (1 + 2x3)` = **105 runs**: 15 full context, 45 simple summary, 45 Phase 5.

Every compacted budget is compared against **the same reference population**. `simple_summary_v1 @ 80` and `phase_5_baseline @ 80` are both measured against the same 15 full-context results, as are the 160 and 240 rows. The reference does not move with the budget.

A reference result records `target_tokens = 0`, because it never consulted a budget and claiming one would be false. The report prints that as `ref`, and groups on the recorded budget rather than special-casing any strategy by name.

Whether an arm is budget-independent is declared on the strategy (`budget_independent`), not held in a list the orchestrator has to keep in step with reality.

## Architecture

The benchmark **orchestrates the Phase 5.5 runner; it does not replace it**. Each cell of the matrix is an ordinary `ExperimentRunner` run, so leakage prevention, the same-prompt guarantee, failure statuses, and the run fingerprint all come from code that was already reviewed. A second runner would be a second place for those guarantees to be wrong.

Each budget and each repetition is a **separate run with its own `run_id`**, because a budget change alters the fingerprint and results at different budgets are not interchangeable. The repetition index travels in the experiment configuration so two runs of one setup are never filed under one identity. With multiple repetitions the reference produces one independent sample per repetition, and reporting may aggregate those — but never across budgets, because there is nothing to aggregate across.

## Dataset

`v2`, 15 scenarios. Built by taking every `v1` scenario and attaching task-completion criteria; **`v1` is not modified**, and a test asserts its conversations, tasks, and retention checks are unchanged and that it gained no criteria.

Coverage spans all seventeen named failure modes — early and late critical information, long irrelevant history, positive and negative constraints, exact numbers, entities, decisions, reversed decisions, failed approaches, current state, temporal change, confusable entities, tool results, open tasks, rationale, multi-step work, and repeated misinformation. A test fails if any is uncovered.

## Budgets

`80 / 160 / 240` tokens — small, medium, large, chosen against the dataset rather than in the abstract. The v2 scenarios are 242 to 417 tokens, so all three sit below every scenario and compaction genuinely happens at each. A budget above the largest scenario would measure the passthrough path and call it compression; one below the harness minimum would make every arm fail and call that a finding.

**Full context is not held to the budget.** Truncating it to match would make the reference a different, unnamed strategy and flatter everything else. It appears as a single `ref` row because it is run once, not because several rows were merged.

## Metrics

### Retention and completion are never one number

| | Question | Shape |
| --- | --- | --- |
| **Retention** | Did the information survive into the response? | Fraction of checks passed |
| **Task completion** | Did the agent act correctly on it? | All criteria, or none — per scenario |

A response mentioning SQLite has retained the decision. A response mentioning SQLite *and not also* proposing PostgreSQL has acted on it. The second is what a handover is for.

Three things make a completion criterion more than a keyword search:

- **`alternatives` must be absent** — the half that catches an answer hedging across every option it can remember. This is what turns "the word appears" into "the agent chose correctly".
- **`min_words`** sets a substance floor, so a one-word reply cannot satisfy a task that asked for an approach.
- **Criteria are conjunctive** — a scenario is completed when every criterion passes. Partial credit is what retention is for.

### Failure kinds

| Status | Meaning |
| --- | --- |
| `passed` | |
| `omitted` | The right answer is not there |
| `fabricated` | Something the conversation ruled out is there |
| `insubstantial` | Present but too thin to be a continuation |
| `not_evaluable` | Needs a sandbox this project does not have |

Omission and fabrication are separated because they are different failures with different causes: forgetting versus asserting. The fabrication count is reported as **`deterministic_hallucination_checks`** — deliberately not a "hallucination rate", since it counts a narrow signal over scenario-declared alternatives and says nothing about invention in general.

`not_evaluable` is a first-class outcome. A scenario containing one cannot be scored complete or incomplete, and is excluded from the completion denominator rather than counted as a failure. Code tasks needing execution are marked this way; a sandbox is future work.

### Cost

Compaction and continuation are recorded separately: input tokens, output tokens, LLM calls, and latency for each, plus a `total_tokens`. A strategy producing an 8K context from 100K tokens of compaction work is not cheap, and comparing on context size alone would say it was.

Where a provider reports no usage, the field is `None` — never zero.

### Compression

`original_tokens / context_tokens`, measured the same way on both sides so the reference condition comes out at exactly 1.0. **Not a quality measure.** A strategy that compresses harder can continue the work worse, which is the entire reason this benchmark exists.

## Running it

```
python -m open_context_eval benchmark                              # deterministic
python -m open_context_eval benchmark -v --out results.jsonl       # with failure detail
python -m open_context_eval benchmark --real-model                 # needs a configured provider
python -m open_context_eval list-scenarios --benchmark
```

```
export OPENROUTER_API_KEY=...
python -m open_context_eval benchmark --real-model \
    --provider openrouter --model openai/gpt-oss-20b:free
```

**Free models only.** The model must be on the allowlist in `open_context.llm.free_models`, which is checked before any request leaves the machine. An unapproved or paid model fails with `ModelNotApprovedError` and costs zero network calls. There is no paid fallback: a rate-limited or unavailable free model fails rather than becoming a different one.

No Groq model is currently approved — Groq prices every chat model it serves, and its free tier is an account allowance rather than zero-priced models. See docs/llm.md.

Real-model results record `billing_class = "free"`; deterministic results record `"none"`.

`--provider` and `--model` go together; omit both to fall back to `OPEN_CONTEXT_PROVIDER` and `OPEN_CONTEXT_MODEL`. Everything goes through the existing `LLMProvider` interface — the benchmark knows nothing about any vendor's HTTP API, and there is no `if provider == "groq"` anywhere in it.

A missing credential **fails** with an explanation. It never falls back to the fakes: a real-model run that silently became a deterministic one would produce numbers labelled as measurements. Model identifiers are the caller's; nothing here names one, because free tiers and catalogues change.

The first real experiment should use one model across all three arms, for both compaction and continuation, so context representation is the only variable.

Artifacts go to `benchmarks/results/` and `benchmarks/reports/`, both git-ignored.

### Running it against a metered free tier

Phase 5.8's constraint is arithmetic. The full matrix is 105 runs and around 195 requests; OpenRouter's free tier documents **50 requests a day and 20 a minute** for an account that has never bought credits. Neither is negotiable without spending money, which the project does not do.

**The binding constraint in practice was neither of those.** The published limits set the ceiling a run is sized against, but what actually cost requests was the upstream model returning 429 at request rates far below 20 a minute — around a quarter of the first run, at one request every 3.2 seconds with 10–20 seconds of latency between them, so roughly four requests a minute. Free capacity for a specific model is shared and unpredictable in a way a documented rate limit is not. Size a run against the published numbers; expect the throttling to come from somewhere else.

```
python -m open_context_eval benchmark --real-model \
    --provider openrouter --model google/gemma-4-26b-a4b-it:free \
    --budget 160 --max-requests 46 --min-interval 3.2 \
    --resume --out benchmarks/results/run.jsonl
```

| Flag | What it is for |
| --- | --- |
| `--max-requests` | Stop after this many requests. Cells not reached are **absent**, not failed |
| `--min-interval` | Seconds between requests, to stay under the per-minute rate |
| `--resume` | Skip cells already recorded in `--out`, and append to it |

**Groq has a fourth limit, and its headers do not mention it.** Read on
2026-08-16:

| Model | Requests/day | Tokens/minute | Tokens/day | Reasoning |
| --- | --- | --- | --- | --- |
| `llama-3.1-8b-instant` | 14,400 | 6,000 | not in headers | none |
| `llama-3.3-70b-versatile` | 1,000 | 12,000 | **100,000** | none |
| `openai/gpt-oss-20b` | — | 8,000 | not in headers | separate field |
| `openai/gpt-oss-120b` | — | 8,000 | not in headers | separate field |
| `qwen/qwen3.6-27b` | — | 8,000 | not in headers | inline `<think>` |

`x-ratelimit-*` reports requests-per-day and tokens-per-*minute*. The
**tokens-per-day** ceiling appears nowhere until a 429 announces it, and on the
70B model it is what actually binds: 100,000 tokens a day is about **22 `v4`
cells**, against a full matrix of 96.

This is worth stating plainly because reading the headers and concluding there
was no daily token limit is exactly the mistake made here — a limit that is
absent from the telemetry is not a limit that is absent. Size a run against
*measured consumption per cell*, and treat the headers as a floor on what
constrains you rather than the whole picture.

For `v4`, whose requests run about 3,000 tokens and whose cells cost about
**4,400 tokens all-in** including judging, tokens-per-minute sets the pace and
tokens-per-day sets the size. `--resume` is what makes a run larger than one
day's allowance possible at all.

**Check the reasoning column before choosing a model.** A model that thinks
before it writes spends the output cap doing it, and at a compaction cap of 160
returns nothing at all. This voided a complete Phase 8.5 run — see [adversarial.md](adversarial.md). `check_output_cap` now refuses such a combination
before the request is sent, but choosing a model that answers directly avoids
the question.

**Before reading any score from a real run, check `context_tokens`.** Every
compaction arm should land near the budget. A zero, or a value far below it,
means the arms were not given what the scores claim they were given.

**A stop is not a failure, and the distinction is the whole point.** Without a ceiling, request 51 gets a 429, and so does every request after it: the results file then holds real measurements for the first fifty cells and rate-limit errors for the rest, mixed together and separable only by reading error strings. Worse, those errors *look like findings* — a strategy that failed on two thirds of the dataset — when nothing was measured about it at all. With a ceiling the run stops, and a cell nobody asked about is simply not in the file.

`RequestBudgetExhausted` is deliberately **not** an `LLMError`. The runner turns any `LLMError` into a `MODEL_ERROR` result, so if exhaustion were one, the machinery meant to prevent phantom failures would generate them itself.

**Results are written as they are produced**, not collected and saved at the end. A request against a metered tier is not repeatable on demand, so a run that accumulated privately and then stopped would discard every measurement it had already bought. `ResultWriter` flushes each line; the file is valid JSON Lines at every instant.

**Resuming does not re-measure.** `completed_cells` reads back `(run_id, scenario_id, strategy)` triples, and the run id already encodes the budget, repetition, model, tokenizer, prompt, and dataset — so a cell from a different budget is never mistaken for one already done. Only **successful** cells count as complete: a rate limit says nothing about a strategy, and treating it as done would freeze a transient outage into the results permanently.

**Pacing lives in the harness, not the provider.** Phase 4.5 kept retry policy out of the provider boundary and Phase 5.7 kept it out. How fast a *run* may go is a property of the run, so it sits next to how many requests the run may send. Nothing catches a 429 or retries: a 429 that arrives is a real provider failure and is recorded as one.

## The failure taxonomy

A number like "0.62 retention" is not something a next phase can design against. What is actionable is *which kind of thing* each strategy drops, so every observed failure is classified into one of the fifteen classes the roadmap names.

| Class | Derived from |
| --- | --- |
| `lost_constraint` | A failed constraint-retention check |
| `lost_negative_constraint` | The same, where the scenario is built around a negative constraint |
| `lost_decision` | A failed decision check |
| `lost_rationale` | A failed rationale check |
| `lost_current_state` | A failed critical-fact check in a current-state scenario |
| `lost_incomplete_work` | A failed open-task check |
| `lost_failed_approach` | A failed failed-approach check |
| `entity_confusion` | A failed entity check, or a critical fact in a confusable-entity scenario |
| `temporal_confusion` | A failed temporal-state check |
| `exact_value_loss` | A failed critical-fact check in an exact-value scenario |
| `hallucination` | A completion criterion answered with something the conversation ruled out |
| `budget_overflow` | A budgeted arm whose context exceeded its target |

**A class is a property of an observed failure, not of a scenario.** `FailureMode` says which trap a scenario sets and is fixed when the dataset is written; a class here says what actually happened in one run. The two appear together because a scenario's trap is what disambiguates a failed check whose category alone is not specific enough — a lost constraint and a lost *negative* constraint are both written as constraints.

**Three classes are declared and cannot currently be measured. They report as unmeasured, never as zero.**

| Class | Why not, and what would fix it |
| --- | --- |
| `lost_goal` | No retention check carries a goal category. Needs scenarios with an explicit goal statement and a check for it |
| `lost_completed_work` | The dataset checks open tasks, not finished ones. Needs scenarios whose task depends on not redoing completed work |
| `irrelevant_context_retention` | Every check asks whether something survived; none asks whether something that should have been dropped survived instead. Needs a judge, or checks naming what must be absent |

A zero beside `lost_goal` would assert that no goal was lost — a claim about the strategies rather than about the harness, and the wrong one, since a strategy that dropped every goal in the dataset would produce exactly the same zero.

**Every failed check lands somewhere.** One whose category and scenario do not determine a class is counted as `unclassified` rather than dropped, so the per-class counts and the total number of failures agree. A taxonomy that silently discards evidence it cannot label is how a gap in the mapping stays invisible.

**A failed run contributes nothing.** A run that ended in a model error produced no response, so it exhibits no forgetting. Counting it would turn a provider having a bad afternoon into a finding about a strategy. Run failures are reported separately.

## Interpreting a report

```
Strategy                Budget  Runs  Retention  Completion  Context   Compr  Total tok  Calls
full_context               ref    15       ....        ....      297    1.0x          -    0.0
phase_5_baseline            80    15       ....        ....       24   14.0x          -    1.0
```

`ref` is the budget-independent reference: one run per scenario and repetition, compared against by every budget row.

Read the mode line first. A `deterministic_test` report measures the harness.

Read retention and completion as separate columns; a strategy can be strong on one and weak on the other, and that difference is the most useful thing here.

Read `Calls` and `Total tok` alongside `Context`. A small context bought with several model calls is not free.

Then read the failure list, which is the point. `-v` prints, per scenario and strategy, which retention check failed and how, and which completion criterion failed and why. "What did this strategy lose, on which scenario" is what a next phase can act on; a score cannot answer it.

## Repetitions

One by default. A real model is nondeterministic and would need more. **Nothing is averaged away**: each repetition is recorded separately so the reporting layer can show spread rather than hide it, and the number used is recorded in the fingerprint.

## Statistical claims

None are made, and none should be from 15 synthetic scenarios and one repetition. This is the first benchmark; it establishes baseline behavior and the machinery to measure it. Phrases like "significantly better" require a test and a sample size that do not exist here.

## Full context is a reference, not an oracle

It is the best representation available *within the selected model's context window*. It is not an upper bound on what a context representation could achieve, and on a model with a much larger window the comparison changes. A future strategy beating full context on some scenario would be interesting, not impossible.

## What the benchmark should eventually answer

Unanswered, and the reason the benchmark exists:

1. How much does generic summarization lose?
2. Does recent-window preservation help?
3. Which failure modes cause the largest degradation?
4. Does compression correlate with continuation quality?
5. How does quality degrade as the budget falls?
6. Which information types are most fragile?
7. Does the Phase 5 baseline beat plain summarization?
8. Where do both fall short of full context?

**Do not design the next phase from one or two failures.** Produce the data first, look at the distribution, then decide.

## What the first real run found about the benchmark itself

**The most important Phase 5.8 finding is a defect in the measurement, not in any strategy. Eight of the ten failure findings the run produced are artefacts of it.**

| Evidence the scorer used | Findings | Verified correct answers wrongly failed |
| --- | --- | --- |
| A **required** term was absent | 2 | 0 — these are real losses |
| A **forbidden** term was present (`RetentionCheck.forbidden`) | 4 | 4 |
| A ruled-out **alternative** was present (`CompletionCriterion.alternatives`) | 4 | 4 |

Every negative-evidence finding in the run is false. The reported "4 responses used something the conversation had ruled out" is really zero, and the taxonomy's `lost_negative_constraint` and `hallucination` rows are entirely artefact. Across 27 successful runs the deterministic scorer found **two** genuine retention losses.

The mechanism is shared. Six of the sixteen `v2` retention checks assert a prohibited term is absent, and the completion criteria assert ruled-out alternatives are absent, both by case-insensitive substring. The reasoning is that a representation which preserved a negative constraint produces an answer not containing the prohibited thing.

Against a real model that reasoning fails, and it fails in the direction that matters. A good answer names what it rejected:

> Since **Redis** is explicitly prohibited and you cannot introduce additional stateful services, I recommend implementing an in-memory local cache...

> The event bus should be implemented as an in-process system using a SQLite-backed queue. Unlike the previously considered **Kafka** solution, this must operate on a single machine...

Both honour the constraint exactly. Both are scored as having lost it, and one of them is additionally scored as a hallucination.

This was invisible for the whole of Phases 5.5 and 5.6 because the fake provider returns one fixed string that never contains any prohibited term, so every negative-evidence check passed for every arm and the flaw could not surface. It is the clearest argument in this project so far for why a harness validated only against deterministic fakes is not a validated harness.

**The bias is not neutral between arms.** Naming and rejecting an option requires knowing the option was rejected, so the arms that retained most are likeliest to trip it. That flatters compaction against full context — the reference is penalised for being better informed, and in this run it collected three of the eight artefacts, more than either compacted arm. Any comparison drawn from negative evidence here understates the reference, in exactly the direction that would make a compaction strategy look good.

**Negative evidence by substring cannot work, and a better regular expression is not the fix.** `RetentionCheck`'s own documentation already says the honest answer to "it cannot tell a mention from a use" is a judged question rather than a cleverer pattern. A heuristic that skips terms appearing near a negation would silently reclassify real failures too.

The repair is to ask the question positively: instead of "the answer must not contain `redis`", assert that it recommends an in-process or in-memory cache. That measures whether the right approach was chosen, which is what the check was always for, and it degrades honestly — a wrong answer fails by lacking the expected content rather than by containing a word.

That is a `v3` dataset, and it is built. `v2` is left exactly as it is: a test asserts it unchanged, and editing a dataset after seeing results is how a benchmark stops meaning anything.

### Dataset v3 — presence is deterministic, absence is judged

`v3` is built from `v2` by transformation, the way `v2` was built from `v1`. Conversations, tasks, and failure-mode tags are `v2`'s verbatim; only how the scorer may reach a verdict changes.

> **Deterministic checks assert presence. Judged questions assert absence.**

Presence is decidable by substring — a required term is in the text or it is not, and a false pass needs the model to have written it by accident. Absence is not, because the same term appears in an endorsement and in a rejection.

| | v2 | v3 |
| --- | --- | --- |
| Retention checks | 16 | 13 |
| Checks asserting absence | 6 | **0** |
| Completion criteria | 15 | 13 |
| Criteria asserting absence | 14 | **0** |
| Judged questions | 1 | 7 + 14 |

A check keeping a `required` list survives with its `forbidden` dropped — the half that worked is not thrown away. A check that was *only* a `forbidden` list is removed rather than emptied, because an empty check is one that always passes.

Nothing is silently lost. Every removed assertion becomes a `JudgedQuestion`, which reports **not evaluated** without a judge rather than passed or failed. The hedging questions are derived from the criterion they replace — restating its description and naming the exact options removed — so a question cannot drift from what it stood for, and adding a criterion cannot leave an unreplaced assertion behind. Each ends by telling the judge that naming a rejected option is *not* a violation, which is the distinction the substring match could not make.

**Two scenarios now have no deterministic evidence at all.** `negative-constraint` and `failed-approach` state a prohibition and never say what to use instead, so the space of acceptable answers is open and there is no positive assertion to make. They were only ever measurable by the mechanism that turned out to be broken. Without a judge they measure nothing — the honest reading, and not a regression to fix by inventing an expected answer.

**What it costs: hedging is no longer caught deterministically.** A response saying "use SQLite or PostgreSQL" now satisfies `reversed-decision`'s criterion. That is a real loss of power, accepted rather than papered over — a check that catches hedging and also fails correct answers was not measuring hedging. Running `v3` without a judge measures less than `v2` appeared to, and more than `v2` actually did.

**A judge is now the gating dependency**, not a refinement — so one was built. `LLMJudge` grades through the ordinary `LLMProvider`, knows nothing about any vendor, and is enabled with `--judge`.

```
python -m open_context_eval benchmark --real-model --dataset v3 --judge \
    --provider openrouter --model google/gemma-4-26b-a4b-it:free \
    --budget 160 --max-requests 46 --min-interval 3.2 \
    --resume --out benchmarks/results/v3.jsonl
```

**A judge that cannot decide says so.** A grading model that errored, or answered in a shape the parser could not read, produces `evaluated=False` — recorded as not evaluated, never as a failure. A guessed NO is indistinguishable from a real one in a results file, so guessing would silently corrupt the measurements the judge exists to provide.

**The judge shares the run's request budget.** It is constructed from the already-budgeted provider, so its requests count against the same ceiling. Built from the raw provider it would send requests around the ceiling, and a run would report spending 46 of 46 while having sent far more; a test pins this.

**Judging roughly doubles the cost of a run, and that is the binding number.**

| Run | Requests |
| --- | --- |
| `v3`, one budget, no judge | ~75 |
| `v3`, one budget, with judge | **~120** |
| `v2`, one budget, no judge | ~75 |

At 50 free requests a day, a judged `v3` run at a single budget takes about **two and a half days** of quota. Three budgets would take a week. That is the real constraint on this benchmark, and it is arithmetic rather than engineering.

**Judged questions must be phrased so that YES is the desirable answer**, because the harness records `passed` straight from the verdict. This is easy to get wrong and was gotten wrong: two hand-written questions asked "does the response recommend Redis?", where a correct answer is NO, so every correct answer scored as a failure.

The judge is what caught it, and only because its reasoning was read rather than its verdict:

> `passed=False` — *"The response does not mention Redis at all, let alone recommend using it."*

A verdict whose own explanation says the answer is good, filed as a failure. Unread, it would have produced a clean-looking benchmark in which `negative-constraint` and `failed-approach` failed on every arm, and that would have been reported as a finding about compaction.

The fourteen *derived* hedging questions were unaffected — deriving a question from the criterion it replaces leaves less room for this than writing one by hand. A test now enforces the rule in its checkable form: any question that acknowledges the naming-in-order-to-reject case must map that case to YES.

**This is the second scoring defect a real model exposed that deterministic fakes could not.** The first made 8 of 10 findings false; this one would have cost two entire scenarios. A fake provider returning one fixed string exercises the plumbing and cannot exercise the measurement, and both defects lived in the gap between those.

**Its bias is uncharacterised.** It is nondeterministic even at temperature zero, and it may be the model under test — `LLMJudge.name` carries the provider and model into the run fingerprint so a reader can see when that happened, rather than the configuration being refused, since on a free tier there may be only one model available.

## Limitations

- **No real-model results.** The largest limitation, and the reason nothing here is a measurement.
- **Substring-based evaluation.** Discriminating, but still text matching; it cannot tell a correct approach from one that names the right words.
- **No code execution.** Code criteria are marked `not_evaluable` rather than approximated.
- **No LLM judge**, so rationale quality and general hallucination are unmeasured.
- **15 synthetic scenarios**, written by the same hand that wrote the strategies they test.
- **One repetition** by default, which is not enough for a nondeterministic model. The matrix supports more, and records each separately rather than averaging.
- **One tokenizer**, and an estimating one at that in the CLI's default.
- **Retention and completion are proxies** for continuation quality, not the thing itself.
