# Evaluation

Phase 5.5. The machinery for finding out whether compaction actually works.

> **The primary evaluation target is downstream task continuation, not summary quality.**

A summary that reads well to a human tells us nothing. A compact context containing many true facts tells us nothing either. The question is whether a *different agent*, given only the compacted representation, can carry the work forward.

**This phase produces no results.** It builds the apparatus. Nothing in this repository states how any strategy performs, because nobody has run it against a real model yet.

## The shape of an experiment

```
                    ORIGINAL CONVERSATION
                           |
            +--------------+--------------+
            |              |              |
       FULL CONTEXT     SUMMARY       BASELINE
            |              |              |
            +--------------+--------------+
                           |
                    SAME DOWNSTREAM TASK
                           |
                    SAME MODEL, SAME PROMPT
                           |
                        METRICS
```

Only the context representation varies. The task, the continuation prompt, the model, the tokenizer, and the target budget are held identical, so a difference in outcome is attributable to the representation rather than to the harness. Future arms — retrieval, MemHandoff — plug in at the same seam.

## Separation from the runtime

```
open_context           the runtime
open_context_eval      the harness
```

The harness depends on the runtime. **The runtime never depends on the harness**, checked by a test that parses every runtime module and fails on an import of `open_context_eval`. The harness is also excluded from the distributed wheel — `packages = ["src/open_context"]` — so installing MemHandoff does not install its own benchmark.

## Scenario model

A scenario is one sentence: *here is a conversation that happened, and here is the task the next agent must perform.*

```python
EvaluationScenario(
    scenario_id="reversed-decision",
    dataset_version="v1",
    description="A decision is made, then explicitly reversed later.",
    failure_modes=(FailureMode.REVERSED_DECISION, FailureMode.TEMPORAL_STATE),
    conversation=[...],  # what the agent may see
    task=DownstreamTask(...),  # what it is asked to do
    checks=(...),  # what only the evaluator may see
    judged=(...),
)
```

The conversation becomes a real archived session through the ordinary import path, so a scenario is not a parallel conversation format that could drift from the one the runtime actually handles.

**Task kinds**: question answering, code continuation, code modification, decision continuation, constraint-sensitive, multi-step. A new kind needs no change to the runner.

## Leakage prevention

Two things must not reach the downstream agent: material the strategy was supposed to remove, and answers only the evaluator should know. Both are prevented **structurally**, not by review.

```python
def build_continuation_request(context: str, task_instruction: str, ...) -> GenerationRequest
```

Two strings in, one request out. This function has no access to a scenario, so it cannot put an expected answer into a prompt; and it has no access to the archive, so it cannot reintroduce a conversation the strategy dropped. The prevention is that the information is not in scope.

A test asserts the downstream user message is *exactly* `CONTEXT + TASK` and nothing else.

One distinction the tests are careful about: a value a check looks for may legitimately appear in a prompt, because the conversation said it. Finding "ZEBRAFISH" in a full-context prompt is the conversation arriving, not an expectation escaping. What must never appear is the evaluator's own words — check descriptions, field names, the structure of the expectations.

**Blind evaluation.** The continuation prompt never says which strategy is being run. Telling a model it holds a summary invites hedging; telling it it has the full conversation invites confidence. Either would be measured as a property of the strategy. A test asserts the prompt contains no word like "summary", "compact", "baseline", or "strategy".

## Strategies

| Name | What the agent receives | Model calls |
| --- | --- | --- |
| `full_context` | The original conversation, unmodified | 0 |
| `simple_summary_v1` | One LLM summary of everything | 1 |
| `phase_5_baseline` | Historical summary plus verbatim recent window | 1+ |

`retrieval` and `memhandoff` are deliberately absent. A named arm with no implementation would appear in reports as a strategy that scored nothing, which is not the same as one that was never run.

**`full_context` is the reference condition and must never quietly become something else.** When the conversation exceeds the receiving model's context window it reports itself *unavailable* and the run is recorded as a `context_error`. It is not truncated: a truncated full-context arm is a different strategy wearing the reference condition's name, and every comparison against it would flatter the others.

**`phase_5_baseline` calls the Phase 5 compactor through its public interface.** Nothing in the harness modifies or reaches inside `BaselineRecentPlusSummary`. The compaction report travels into the result's `strategy_detail`, so a benchmark can see which fallbacks fired.

### `simple_summary_v1` is a baseline, not an engine

Exactly three steps and no more:

```
FULL CONVERSATION -> ONE LLM SUMMARY -> DETERMINISTIC OUTPUT-BUDGET ENFORCEMENT
```

It lives in the evaluation harness because that is what it is for: representing what a developer gets from asking a model to summarize a session, so everything else has something ordinary to be measured against. It is not exported by the runtime, is not offered as a strategy anyone should use, and should not grow features. If it starts to look good, that is a finding about the problem, not an invitation to promote it.

**It shares no compaction logic with Phase 5.** No recent window is preserved, which is precisely what makes the value of keeping recent turns verbatim measurable rather than assumed, and borrowing Phase 5's budget machinery would mean this arm was partly measuring that machinery. Its enforcement is its own: count the summary, cut at a word boundary if the model overshot. No retry, no fallback ladder.

It is versioned in its own name — alone among the arms — because it is the one most likely to be replaced by a different idea of what "a plain summary" means. A result file mentioning `simple_summary_v1` says exactly which baseline produced it.

## Metrics

### Retention checks are not task completion

This distinction is the one most likely to be misread, so it is drawn sharply.

**A. Deterministic retention checks** — `RetentionCheck`, scored as `retention_score`. They ask whether an expected fact, constraint, or decision *survived into the downstream response*. They are cheap, deterministic, and offline.

**B. Downstream task completion / continuation quality** — **not measured**. Whether the agent actually carried the work forward is a different question, and nothing in this phase answers it. A response can pass every retention check and still be a useless continuation: naming SQLite is not the same as writing working connection setup.

Retention is evidence *toward* continuation quality, not a substitute for it. Reports show `Retention` as a column and never as a verdict.

There is deliberately **no `task_completion` metric category**. A substring check cannot establish that a task was completed, and a category by that name sitting among the retention checks would invite exactly the conflation this naming exists to prevent.

**What "almost as effectively as full context" means is deliberately undefined.** No threshold is chosen — not 90%, not any number. Picking one before any measurement exists, or after seeing results, is how a benchmark becomes marketing.

### The check shape

**Objective and judged metrics are separate**, and only objective ones are implemented.

A retention check names strings that must appear, strings of which at least one must appear, and strings that must not. One shape covers a critical fact, a negative constraint, a decision taken over its predecessor, and an avoided failed approach. The metric category is a label on the check rather than a different mechanism.

| Category | How | Status |
| --- | --- | --- |
| Critical fact retention | Required value present | Deterministic |
| Constraint retention | Forbidden value absent | Deterministic |
| Decision retention | New decision present, superseded one absent | Deterministic |
| Decision rationale | Reason present | Deterministic, plus judged |
| Temporal state accuracy | Current state present, stale absent | Deterministic |
| Open task retention | Correct unfinished task named | Deterministic |
| Failed approach retention | Known failure not repeated | Deterministic |
| Entity accuracy | Correct entity of a confusable pair | Deterministic |
| Token count, compression, latency, LLM calls | Measured | Deterministic |
| Task completion | Conjunctive, discriminating criteria per scenario | Deterministic, added in Phase 5.6 |
| Continuation quality overall | Needs a judge or a human | **Not implemented** |
| Provenance accuracy | Needs provenance in the context | **Not implemented** — nothing produces provenance yet |
| Hallucination rate | Needs a judge | **Not implemented** |

**Substring matching is crude and admits it.** It cannot tell a mention from a use: a model writing "we should not use Redis" satisfies a check looking for "Redis". Scenarios are written with that in mind, and where it is not good enough the honest answer is a judged question, not a cleverer regular expression.

**The judge is a seam, and now also has an implementation.** `EvaluationJudge` is a protocol; `KeywordJudge` is a deterministic stand-in for tests; `LLMJudge` grades through the ordinary `LLMProvider`. A judged question with no judge is reported as **not evaluated** — never as passed, never as failed. Scoring an unanswerable question as a failure would penalise strategies for the evaluator's limitations.

`LLMJudge` was added in Phase 5.8 because dataset `v3` made it load-bearing: every assertion about something being *absent* moved to judged questions, since absence is not decidable by substring. See [benchmark.md](benchmark.md).

**A judge that cannot decide says so.** `JudgeVerdict.evaluated` is false when the grading model errored or answered in a shape the parser could not read, and the runner records that as not evaluated rather than as a failure. A guessed NO is indistinguishable from a real one in a results file, so guessing would silently corrupt the measurements the judge was added to provide. The parser reads YES or NO off the first meaningful line, tolerating markdown and punctuation around it, and refuses to mine a verdict out of prose — finding one there means deciding which of several words was the answer.

**Three problems it brings, none solved.** It is nondeterministic, so two runs of one experiment can disagree; temperature is pinned to zero, which reduces that and does not remove it. It may be the model under test, which is a known bias — `LLMJudge.name` carries the provider and model into the run fingerprint so a reader can see when it happened, rather than the configuration being refused outright, since on a free tier there may be only one model available. And it costs a request per question: `v3` has 21 judged questions, so a judged run at one budget costs about 120 requests against the 75 an unjudged one costs. Pass the *budgeted* provider, or the judge's requests go around the ceiling and the ceiling stops meaning what it says.

## Dataset

`DATASET_VERSION = "v1"`, 15 scenarios, each built around one way a summary loses something that mattered:

| Scenario | Trap |
| --- | --- |
| `early-critical-fact` | Requirement stated once at the start, then buried |
| `late-critical-fact` | The deciding detail arrives in the last turns |
| `negative-constraint` | "Do NOT use Redis", then Redis discussed anyway |
| `reversed-decision` | PostgreSQL, then explicitly changed to SQLite |
| `failed-approach` | An approach that ran out of memory must not be retried |
| `exact-value` | Timeout must be 17 seconds, not a rounder number |
| `similar-entities` | `billing-api` versus `billing-worker` |
| `temporal-state` | State changed twice; only the latest is true |
| `tool-result` | The row count came from a tool, not a person |
| `open-task` | Three tasks, two finished |
| `rationale` | A decision that only makes sense with its reason |
| `multi-step` | Continue the plan, do not restart it |
| `repeated-information` | A wrong figure repeated; the correction stated once |
| `confusable-numbers` | Three ports, one of them right |
| `decision-with-constraint` | A rejection and a decision that must both survive |

Written, not sampled. No private conversation is used and nothing is downloaded. Filler is generated deterministically, so a scenario can contain a long irrelevant stretch without the file containing one.

**The version means something.** Increment it whenever a scenario changes materially. A result file records the version it ran against, and comparing across a silent edit would compare two different benchmarks. A run that is handed scenarios from more than one version records `invalid_scenario` rather than mixing them.

## Result format

JSON Lines, one object per line, each independently interpretable — a results file can be concatenated, filtered, and grepped without a parser that tracks state.

```
evaluation-results/run-<id>.jsonl
```

Every line carries `format_version` and the full run metadata: run id, dataset version, provider, model, context window, tokenizer and whether it counts exactly, continuation prompt id and content hash, target budget, software version, and git commit where there is one.

**Append-friendliness is load-bearing, not incidental.** Because a line is a complete record, a run that stops partway leaves a valid file rather than a truncated document, and a later run adds to it. `ExperimentRunner.stream` yields results as they are produced and `ResultWriter` flushes each one, so a run against a metered provider never loses a measurement it has already paid for; `run` remains available and is now a wrapper over `stream`. `completed_cells` reads a file back into the `(run_id, scenario_id, strategy)` triples it already holds, which is how a resumed run avoids paying twice. See [benchmark.md](benchmark.md) for how Phase 5.8 uses this against a 50-request daily quota.

## Reproducibility

**`run_id` is a hash of a canonical serialisation of everything that can change a result**, and of nothing that cannot.

Included: dataset version, sorted scenario ids, provider, model, context window, tokenizer identity and whether it counts exactly, target budget, max output tokens, continuation prompt id and content hash, selected strategy names, strategy versions, each strategy's own configuration, the judge if any, and any extra experiment configuration.

Excluded: timestamps, filesystem paths, hostnames, and anything else that varies between machines running the same experiment. An id that moved with the clock or the working directory could not identify an experiment at all.

An id that stayed the same across a changed tokenizer, budget, or strategy configuration would let two incomparable experiments be filed under one identity — a worse failure than an id that changes too eagerly.

**The fingerprint is persisted, not just hashed.** `RunMetadata.fingerprint` carries the exact object that produced `run_id`, on every result line. A result that recorded only an opaque `run-a1b2c3` would name an experiment nobody could reconstruct; with the fingerprint beside it, a results file read back years later is self-describing, re-hashes to confirm its own identity, and can be diffed against another run to find out why the two disagree.

The fingerprint is computed once per run and both hashed and stored, so there is one object and no second derivation to drift. The named convenience fields beside it — `provider`, `model`, `target_tokens`, and the rest — are read out of the fingerprint rather than gathered again, and a validator refuses metadata whose fields contradict it. Nothing stored can say one thing in its fingerprint and another next to it.

`started_at`, `git_commit`, and `software_version` sit beside the fingerprint and deliberately outside it: they describe the occasion, not the experiment.

With the deterministic fakes, the same experiment reproduces exactly apart from `latency_seconds` and `started_at`. A real provider is nondeterministic, and a result produced with one records the model that produced it.

## Failures are not zeros

| Status | Meaning |
| --- | --- |
| `success` | The run completed; the continuation may still be wrong |
| `model_error` | The provider failed |
| `context_error` | The strategy could not produce a context |
| `tokenization_error` | No tokenizer was available |
| `evaluation_error` | Scoring failed |
| `invalid_scenario` | The scenario could not be run |

A failed run carries **no metrics** — `check_score`, `compression_ratio`, and `checks_passed` are `None`, not zero. Averaging a failure as zero silently reports a broken experiment as a bad strategy. Reports print the status instead of a number, and list failures separately from scores.

## Reporting

A text table and structured data. No plotting, no dashboard, no HTML.

```
Strategy                Runs    Tokens  Compression      Checks
---------------------------------------------------------------
full_context               2       300         1.0x         2/2
phase_5_baseline           2       204         2.1x         1/2
```

`pareto_points` returns cost against quality as data for whatever wants to plot it. Nothing here decides that fewer tokens at a lower score is better or worse — that is the question the benchmark exists to inform.

**Compression is not quality.** A strategy that compresses harder can continue the work worse, which is the entire reason this harness exists.

## Running it

```
python -m open_context_eval list-scenarios
python -m open_context_eval run --target-tokens 2000 --out results.jsonl
python -m open_context_eval compare results.jsonl
```

Deliberately a module runner rather than an installed command: the project has no CLI conventions yet and an open question about what its command should be called, and claiming a console script here would answer that by accident.

`run` uses the deterministic fakes unless `--real-model` is passed, so it works with nothing installed and never silently reaches the network. It prints, in that mode, that the numbers came from a fake model and measure the harness rather than any strategy.

## Repeated experiments

The archive is append-only, so importing a scenario twice into one session would silently double the conversation and every later experiment would evaluate a session that says everything twice. That was a real defect and is now fixed.

Materialization is **idempotent**: a session is created and imported only when it is not already there. Its id derives from the scenario's *content*, not just its id, so an edited scenario becomes a different session rather than colliding with a stale materialisation of its earlier self — which is also what makes reuse safe, since a session under that id can only have come from exactly those events. A session that exists but holds something else raises `ScenarioMaterializationError` rather than being reused or appended to.

Nothing about the production archive was changed to achieve this. Its append-only semantics are the reason for the care, not an obstacle worked around.

## The benchmark

Phase 5.6 drives this harness across a matrix of scenarios, strategies, and budgets, and adds the metric this phase deliberately left unmeasured: **task completion**, separate from retention and never averaged with it. Dataset `v2` extends `v1` with completion criteria without editing it. See [benchmark.md](benchmark.md).

## Deterministic test mode

Every test runs offline: no API key, no network, no Ollama, no model server, no database. This is a property of the design — the Phase 4.5 fakes are the whole model surface — not something the suite arranges for itself. Real-model runs go through the same `LLMProvider` interface and are a manual act; CI never depends on one.

## Limitations

- **Substring matching cannot tell a mention from a use.** The largest weakness in the deterministic metrics.
- **An LLM judge exists but nothing has been run through it**, so rationale quality, semantic correctness, and hallucination remain unmeasured. Its bias is uncharacterised.
- **Provenance accuracy is unimplementable** until something produces provenance.
- **One sample per scenario.** A stochastic model needs repetitions; nothing here repeats.
- **Scenarios are synthetic and small**, written by the same hand that wrote the strategies they test.
- **`full_context` is unavailable for conversations larger than the model**, which is the honest outcome but leaves the reference condition missing exactly where compaction matters most.
- **Retention score is not task completion.** It is a proxy, and a coarse one. Continuation quality is unmeasured.

## Unresolved

Recorded rather than answered.

**Which metrics should ultimately be human-judged?** Substring checks are cheap and shallow; humans are expensive and inconsistent; LLM judges are somewhere in between and biased in ways nobody has characterised here.

**How should code correctness be evaluated?** The code tasks currently check for strings. Running the code would be far better and needs a sandbox.

**Which models should form the standard benchmark?** Results will differ by model, and a benchmark tied to one model measures that model.

**Should the same model do compaction and continuation?** Using one model for both may flatter compaction, since the summary is written in a dialect the reader shares.

**How should stochastic outputs be handled, and how many repetitions are needed?** Both unanswered, and both required before any number from this harness means anything.

**How should LLM judge bias be controlled?** Position bias, verbosity bias, and self-preference are all documented problems.

**How should multimodal and tool-use continuation be evaluated?** Neither is represented in the dataset.

**How should provenance accuracy be measured** once there is provenance to measure?

**What counts as "almost as effectively" as full context?** The product claim needs a threshold, and picking one after seeing results is how benchmarks become marketing.

**How should benchmark difficulty be calibrated?** A dataset every strategy passes and a dataset every strategy fails are equally uninformative.
