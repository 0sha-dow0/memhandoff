# Phase plan

The project is built one layer at a time. Each phase ends with tests passing, docs updated, and a stop. Nothing advances to the next phase automatically.

Every phase runs the same loop:

```
plan -> implement -> test -> lint -> typecheck -> inspect diff -> document -> stop
```

## Position

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository, packaging, test harness, licensing, CI | Done |
| 1 | Pydantic data model | Done |
| 2 | SQLite persistence | Done |
| 2.1 | Storage integrity corrections | Done |
| 3 | Local immutable archive | Done |
| 3.1 | Archive manifest identity, invariants stated | Done |
| 4 | Conversation import adapters | Done |
| 4.1 | Import identity and failure semantics | Done |
| 4.5 | LLM provider and tokenizer abstraction | Done |
| 5 | Baseline compaction engine | Done |
| 5.5 | Evaluation framework | Done |
| 5.6 | First benchmark against summarization and retrieval baselines | Done |
| 5.7 | Real model provider layer | Done |
| 5.8 | First real-model benchmark run | Done |
| 6 | Structured extraction | Done |
| 7 | Retention and state management | Done |
| 8 | Hybrid compaction | **PARTIAL** — implemented and tested; its *measurement* exit criterion is unmet, because on the only model with the daily allowance to run the full matrix extraction returns nothing and the arm is not hybrid. A research limitation, not a V1 engineering blocker |
| 8.5 | Adversarial evaluation | Done — the benchmark now discriminates |
| 9 | Portable context package format and package builder | Done |
| 10 | Context compiler, package to agent-specific context | Done |
| 11 | First agent integration: Claude Code, cross-agent handoff | Done |
| 12 | Production hardening: security, performance, compatibility | Done |
| 13 | Open source V1: clean install, release gate, docs | Done — release gate passed against a built wheel in a clean environment |
| V2.1 | Automatic context pressure | Done |
| V2.2 | Incremental compaction | Done |
| V2.3 | Hierarchical context | Done |
| V2.4 | Local lexical retrieval | Done |
| V2.5+ | Deep provenance and beyond | Not started |

Phase 8 is the only entry that is not complete, and it is deliberately not a V1 blocker: the code exists, is tested, and behaves correctly — what is unmet is a *measurement*, because the arm could not be evaluated on the only model with enough daily allowance to run the full matrix.


Candidate integrations for Phase 11 and beyond: Claude Code, OpenAI and Codex, Gemini, local Ollama-based agents, MCP-compatible clients, other coding agents.

Automatic context-pressure detection comes only after we know which hooks each agent actually exposes. Designing interception before that means designing it twice.

## What changed in this roadmap

The product realignment moved the target from a general context runtime to a portable context package for agent handoff. See [product-architecture.md](product-architecture.md).

Three things moved as a result. Evaluation gained its own phases at 5.5 and 5.6 and now runs immediately after the first compactor, so the thesis gets measured against a plain summary before eight more layers are built on top of it. The provider and tokenizer abstraction moved from Phase 16 to 4.5, where the code that needs it actually starts. The package format and package builder became Phase 9, ahead of the context compiler at Phase 10, because the compiler consumes the package and its input shape is defined by the format rather than the other way round.

Dropped from the roadmap: memory backends, paging, and the separate MCP phase as previously scoped. Cross-session memory is a different product. Paging is an optimization with no measured need yet. MCP returns as one adapter among several in Phase 11, not as a milestone of its own.

## What V2.4 covers

The compiler asking the archive for what a task turns out to need. See [retrieval.md](retrieval.md).

**Exit criterion met**: relevant historical evidence is retrieved without loading the archive. Records stream past a bounded heap, so searching 80,000 of them costs what searching ten does — verified by a test that counts every record consumed and measures the memory it took.

**Built against a measurement.** The adversarial benchmark found that what compaction loses is exact values, and that *all* of its loss sat in the two scenarios built around them. A summary cannot keep every number and does not have to.

**Measured on four scenarios drawn from those failures: the answer was present in 0 of 4 compiled contexts without retrieval and 3 of 4 with, at +28% tokens.**

**The one it misses is the honest part.** Asked *"what encoding should the export writer use"*, it cannot find *"the downstream loader rejects UTF-8; write UTF-16LE with a BOM"* — the two sentences share no words at all. That is the structural limit of lexical matching, predictable rather than random, and it is the case that would justify *examining* embeddings under the roadmap's rule. One scenario in four, with the failure understood, is not a reason to add a vector database.

**Provenance travels with the text**: every retrieved line carries the record id it came from, under a heading naming the archive as its origin. A reader must be able to tell a record the task fetched from one the package chose to carry, since the two were selected by different things.

**Retrieval failing is not compilation failing.** An unreadable archive produces a warning and a context without the extra records — smaller, not wrong.

## Making it work on real sessions

Between V2.3 and release, the credentials changed and the pipeline met real data. Four things broke, and each was a real defect rather than a configuration problem. See [extraction-at-scale.md](extraction-at-scale.md).

**The free-model allowlist had gone stale within ten days.** `openai/gpt-oss-20b:free` stopped being free on OpenRouter, and both Groq llama models disappeared from the account — leaving no non-reasoning chat model there at all. `open-context verify-models` now re-checks the registry against the live catalogues so a reader need not trust a date.

**`--extract` never worked.** Providers register on import; the CLI never imported them. The design is right and the CLI simply had not declared itself a caller that wants a real model.

**One-shot extraction does not fit a real session** — 19,178 tokens against an 8,000-per-minute ceiling. Extraction is now windowed, carrying state forward so a later window can still name the decision an earlier one made.

**Rate limits are waited out.** `retry_after` had been carried since Phase 4.5 and deliberately never acted on, because the provider boundary refuses to choose a retry policy for its callers. Windowed extraction is a caller: with pacing the same session went from 3 items and 32 failed windows to 19 and 11.

## What V2.3 covers

A session sits inside a project, alongside other sessions that already established things. See [hierarchy.md](hierarchy.md).

**Exit criterion met**: a session now receives project-level state that earlier sessions established — verified end to end — and it does so with no new infrastructure. A test asserts the module reaches for no storage engine: the model is three levels and a parent link, which is a dictionary.

**A project is a repository, not a working directory**, and real data settled that. Of six measured sessions two ran in `open-context` and `open-context/files (4)/open-context-runtime 2` — one project and a directory inside it. Splitting them would hide a session's own history from itself. Project identity is therefore the nearest ancestor holding a repository marker, answered by looking at the filesystem rather than by parsing a path.

**There is no TASK level**, though the roadmap sketches one. Nothing in a real transcript marks where one task ends, so the boundary would have to be inferred — and an invented boundary looks exactly like an observed one.

**Inherited state is labelled, never merged.** The package builder refuses state from another session because mixing them would present one agent's work as another's; that guard is right, so inherited context got its own field rather than an exemption. It compiles into its own section under a heading naming its origin, and is dropped before the session's own state when the budget is short.

**Two bugs, both caught by tests, both from bolting a feature onto an existing invariant instead of respecting it.** Carried state was merged into `state` and the builder correctly refused it. And sibling lookup leaked across unrelated projects, because nothing recorded which project a stored session belonged to — a constraint from an unrelated project reached a package before stored state started carrying its `project_root`.

## What V2.2 covers

Extracting only what arrived since last time. See [incremental.md](incremental.md).

**Exit criterion met**: measurable efficiency, no quality regression. On a session grown to 200 turns in batches and repackaged after each — **62% fewer prompt characters, 70% less latency, identical state kept**.

**Phase 7 built this and nothing called it.** `Watermark`, `extract_incremental`, and `reconcile` all existed; every package still re-read the whole archive and paid a model for the whole conversation again. The missing half was somewhere to keep the watermark between runs.

**The shape matters more than the percentage.** The incremental prompt stays flat at ~15,000 characters as the session grows while a full read climbs from 14,746 to 64,926 — cost tracks new events rather than total ones, so the saving widens with session length, which is the case this project exists for.

**A caveat that cannot be measured away**: the numbers come from a counting stand-in, not a real model. That is the right instrument for how much work is *sent* — a property of the caller, measured exactly rather than through sampling noise — and it says nothing about whether a real model finds the same things from a partial view.

**The prompt is new events plus known state**, not new events alone, because Phase 7 already paid for that lesson: a model shown only new messages has never seen the decision being overturned and reports a reversal that resolves to nothing.

**A watermark is only as good as the state stored with it**, so they are written together through an atomic rename. Stored apart, a crash could leave a watermark claiming events were read beside state that never saw them, with nothing downstream able to tell. Damage reads as "nothing known" — recovery, rather than turning an interruption into a permanent failure.

## What V2.1 covers

Capturing a package at the moment an agent decides its window is full. See [context-pressure.md](context-pressure.md).

**Exit criterion met**: one integration receives automatic compaction events reliably — `open-context hook` wired into Claude Code's `PreCompact`, verified end to end on real transcripts, producing packages that validate at provenance depth.

**The mechanism was verified before anything was written**, which this phase requires in terms. Claude Code 2.1.195 emits `PreCompact` *before* compacting, carrying `transcript_path` and `trigger`, where `trigger: "auto"` is the agent deciding by itself that its window is full. The timing is the point: after compaction the detail is gone, so a hook that fires before is the difference between capturing a session and capturing a summary of one.

**It declines a power the protocol offers.** The binary supports blocking compaction from this hook, and the handler never does — asserted by a test. A context tool that could silently stop an agent compacting would be able to run a session out of its window, and nobody debugging that would suspect the thing they installed to help. It also never fails the hook: a non-zero exit is how a hook tells an agent something is wrong, and a failed backup is not that.

**Manual compactions are left alone by default.** A user who typed `/compact` decided what they wanted; capturing anyway asserts the tool knows better and doubles what a session costs on disk.

## What Phase 13 covers

Making the project something a stranger can actually use. See [getting-started.md](getting-started.md).

**Exit criterion met**: all ten steps run from a **clean install of a built wheel** in a fresh virtualenv, outside the repository — clone, install, understand, import, compact, generate, inspect, compile, hand off, continue. The release gate (`pytest`, `ruff check`, `ruff format --check`, `mypy`) is clean, and the CLI, `.ctx` creation, validation, and compilation for all five targets were verified against that install rather than an editable one.

**An editable install hides exactly the things a release gate is for.** Two problems only appeared once the wheel was built and installed elsewhere: there was no console script, so a user had to know to type `python -m open_context`; and the CLI help was written for the author, citing phase numbers a stranger has no reason to know.

**Two product gaps surfaced from running the stranger's own workflow.** A single long assistant turn took 476 of a 500-token budget and pushed ten other turns out of the recent window — so turns are now clipped, with the clipping marked. And a package built without extraction carries no state at all, which undersells the artifact; `--extract` now wires the extractor into handoff, opt-in because it is the only step that needs a model, and running *after* archiving so a failed model call costs the call and not the conversation.

**A warning-clobbering bug was caught by its own test**: extraction warnings were being overwritten by the packaging step's, so a failed extraction reported nothing at all.

**The README states the negative result.** The compaction thesis is unproven on this project's own benchmark, and that appears in the status block rather than in a footnote — a benchmark published only when it agrees with you is not a benchmark.

## What Phase 12 covers

What happens when the input is hostile, damaged, or very large. See [hardening.md](hardening.md).

**Exit criterion met**: realistic reliability, security, and performance tests pass — 29 new ones, on top of the 32 storage recovery and memory tests from Phases 3 and 4.

**A `.ctx` is untrusted input by design**, since the format exists to travel between machines, and its contents end up in a system prompt. The roadmap's hard rule — never execute anything because it appeared in imported context — is kept by construction and asserted against the source: no `eval`, no shell, no dynamic import, no path resolution anywhere a package can reach. `archive.location` is carried verbatim and never opened.

**Injection is reported, never claimed to be removed.** It cannot be detected reliably, so a function that stripped "the dangerous parts" would make a promise it cannot keep and a caller who believed it would stop being careful. A clean review says *"not a guarantee of safety"* in as many words, because a report implying safety would be the vulnerability.

**The compiler frames what it did not write.** In a system prompt, "the constraint we agreed" and "an instruction to you" are indistinguishable unless the model is told which it is reading, so every compiled context opens by saying its contents are a record of earlier work rather than instructions. It is budgeted like any other section — a mitigation that arrived free would mean the token accounting was wrong.

**Two performance problems were found by measuring rather than assuming.** Describing an archive called `list()` on it, which is exactly the proportional cost the archive exists to avoid; streaming removed ~80 MB on a 25 MB transcript. Branch resolution held every turn's text when it needs only identities; a two-pass survey took 51 MB to 27 MB.

**A silent detection bug surfaced while testing.** Format detection sampled a fixed 8 KB prefix, but one record can be 1.36 MB — so a large first record arrived truncated, the specific check saw invalid JSON, and the generic reader claimed the file. That would have imported a session as anonymous records, losing tool calls and branch structure, while appearing to succeed. Detection now samples whole lines.

**Four version numbers, each answering a different question**, with the envelope and its contents checked separately: a package this code can open but whose state it cannot read is a specific, fixable situation, and reporting it as an unreadable package would send a reader looking in the wrong place. Configuration is not versioned because there is no configuration file — nothing to version is a better answer than a version nothing reads.

## What Phase 11 covers

Reading a real coding agent's sessions, and handing one to a different agent. See [handoff.md](handoff.md).

**Exit criterion met**: a documented cross-agent handoff runs end to end — `sessions`, `handoff`, `validate`, `compile` — verified on this machine's own Claude Code transcripts and pinned by tests against the real on-disk format.

**The format was inspected, not assumed**, as the roadmap requires. Six real transcripts, 8,890 conversation records. Three findings each contradicted the obvious reading, and one of them was a genuine error caught only by real data:

* **A session file is a graph.** Forks are rewinds; the abandoned attempts stay on disk. 5 of 6 files had them.
* **The parent chain runs through records that are not conversation** — 804 attachments and 62 system records between turns in one session. Following `parentUuid` while indexing only turns breaks the chain at the first attachment.
* **A second root is a continuation, not a rival.** Compaction starts a *new tree*. An earlier `active_thread` kept only the chain ending at the last record — the obvious reading — and discarded **78% and 86%** of two long sessions: every pre-compaction turn, which is exactly what a handoff exists to carry. One of those files has no forks at all, so nothing was ambiguous about it.

**Adapters do not decide.** `handoff` reads, archives, and packages; selection stays in the compactor, budgeting in the compiler, the format in the package. The archive is written before anything is interpreted, so a failed extraction costs the expensive step and not the conversation.

**State is passed in, not extracted here.** Called without it the package says so in a warning rather than implying it carries goals and constraints it does not.

**Registry order is load-bearing**: the generic JSON Lines reader accepts any JSON object line, so a generic-first registry would claim every transcript and import it as anonymous records that look fine.

## What Phase 10 covers

The context compiler: a provider-neutral `.ctx` turned into context a specific agent can use. See [compiler.md](compiler.md).

**Exit criterion met**: the same package compiles to useful context for `openai`, `anthropic`, `gemini`, `local`, and `generic`, each in that family's own request shape, proven from a file on disk through `python -m open_context compile`.

**The portable layer decides; targets only render.** A renderer receives a context that is already selected and already inside the budget, with no tokenizer, no budget, and no package — so it cannot drop, reorder, or re-fit anything even by accident. This is the roadmap's "do not duplicate compaction logic in adapters", enforced by what a renderer is handed rather than by convention. Five adapters that could each choose would become five slightly different compactors, the same `.ctx` would mean different things depending on where it was sent, and every one of them would still look reasonable.

**Fitting is imported, not reimplemented.** `fit_state` comes from the hybrid compactor unchanged; its priority order was arrived at by a benchmark result, and a second copy would drift.

**Naming a target model changes nothing about what is compiled.** It is recorded so a context can say what it was built for. A compiler that wrote different content per model would make the package's neutrality a fiction, and a test pins that it does not.

**An unknown target falls back to plain text rather than failing**, because refusing would make the format less portable than the text it is made of.

## What Phase 9 covers

The portable `.ctx` package — the product artifact the rest of the project exists to produce. See [ctx-format.md](ctx-format.md).

**Exit criterion met**: a `.ctx` file can be generated, validated, and inspected, through `python -m open_context pack | validate | inspect`, proven end to end against a real SQLite store and a real archive on disk.

**One JSON document, not a container.** Human-inspectable is a requirement of the format, and a zip you must unpack before reading does not meet it. When a package grows too large to read, the answer is to reference the archive rather than inline it — which is what the format already does.

**It carries interpretation, not history.** State items and the evidence they rest on, with a pointer back to the archive. Embedding the archive would make the package a second copy that immediately begins to diverge from the first.

**Provenance that cannot be followed is still recorded.** Dropping an unresolvable reference destroys the only trail back to the source; marking it unresolvable costs one field and is true.

**A correspondence Phase 6 left open turned out to be half-open.** Phase 6 recorded that `sources` speaks the SQLite id scheme while the archive numbers its own records, and concluded no correspondence existed. That holds for *sequence numbers*. It does not hold for *ids*: `ArchiveRecord.id` is defined as the identity of the underlying item preserved as given, so an archive fed the same messages already holds the ids `sources` names. Resolving through them is a lookup, not the invention Phase 6 rightly refused.

**Validation has two depths and says which it reached.** Structure alone, or structure plus the archive. The dangerous case is a package that passes the first and fails the second — internally impeccable, describing a different archive — and a verdict that did not name its depth would hide it.

## What Phase 8.5 covers

Dataset `v4`: conversations long enough that the budget forces a real choice. See [adversarial.md](adversarial.md).

**It exists because the benchmark could not do its job.** Phases 5.8 and 8 both failed to separate the arms, and neither failure was about compaction. On a 70B model every arm scored 1.00 on everything — a ceiling effect indistinguishable from a measurement. On an 8B model the arms separated by 0.08, and two runs of the *same configuration* also differed by 0.08, so the separation was noise.

The cause was arithmetic. `v3` scenarios are 220 to 385 tokens across 18 to 32 messages, which at budget 160 is **1.4x to 2.4x compression**. A benchmark whose conversations nearly fit inside the budget is not measuring compaction; it is measuring whether the model can read.

`v4` conversations run 1,400 to 2,300 tokens across 63 to 107 messages — **9x to 14x compression at the same budget**, and something must genuinely be dropped.

**Filler that resists summarization.** `v1`'s filler repeats one sentence with a changing index, which a summarizer collapses to a single line at no cost: it consumes prompt space without consuming *summary* space, so it never forces a choice. `v4`'s filler varies across topic, package, and note on coprime cycles. An earlier version stepped the note by three through a six-element list, which cycles after two, and a test caught it producing 24 distinct messages out of 40 — the exact flaw it was written to avoid.

**Position is a controlled variable.** The same encoding requirement is planted early, in the middle, and late in three otherwise identical scenarios. "Recency wins" and "primacy wins" are different failures with different fixes, and a dataset that scattered its critical information could not tell them apart.

**The `v3` rule is kept.** Deterministic checks assert presence; judged questions assert absence, phrased so that naming a rejected option counts as compliance. Two scenarios carry judged questions only, because they state a prohibition and never say what to do instead — recorded rather than papered over with an invented expected answer.

**The exit criterion is met: the benchmark discriminates.** 96 of 96 cells on `llama-3.1-8b-instant`, three repetitions. The reference arm scores **1.000 with zero variance**, which is the control Phases 5.8 and 8 never established — everything the scenarios ask for is recoverable from the uncompacted conversation, so every point lost below is lost by compaction. Compaction costs **25–28%** at 9–14x compression, against a within-arm spread of 0.167. That is the first real effect this benchmark has produced.

**Two results, one of them unwelcome.** The Phase 5 baseline does not beat a plain one-call summary — 0.750 against 0.722, a gap of 0.028 inside noise of 0.167, so the two remain indistinguishable. Phase 5.8 found this on 220–385 token scenarios; it survives on scenarios six times longer at ten times the compression. And **all** of the loss sits in the two exact-value scenarios; the positional trio scores 1.00 for every arm, ruling out primacy and recency as an explanation for anything else here.

**Hybrid could not be measured at all**, and the reason is a deployment constraint the phase did not know it had: `HybridCompaction` disowned **21 of 24** of its own cells because extraction returned nothing on an 8B model. Its apparent score was the Phase 5 baseline wearing hybrid's name. The benchmark now excludes cells a strategy disowns rather than averaging them in.

**The judge failed its own control.** On the reference arm — deterministic retention 100% — the 8B judge passed 9 of 24 judged questions where the 70B judge passed 8 of 8. All judged scores from that run are void, and the report now says so automatically: a judge that cannot grade the arm holding the whole conversation cannot grade anything below it.

**The first `v4` run was void, and it looked like the best result the project had produced.** On `gpt-oss-20b` the reference arm scored 1.00 and every compaction arm collapsed to 0.09–0.58 — arms separated far beyond the noise floor Phase 8 could not clear. The model is a reasoning model whose thinking is billed against the output cap; at 160 tokens it spent 158 reasoning and returned empty content, so every compaction arm was handed an empty context and the only untouched arm was the one that makes no compaction call. Nothing in the scores could have caught it; `context_tokens` was 0. Fixed in the provider layer (see [llm.md](llm.md)), quarantined at `benchmarks/results/void/`, and rerun on a model that answers directly.

The general shape is worth carrying forward: **a benchmark comparing a no-compaction reference against compaction arms reads any systematic failure of the compaction call as evidence for the reference.** That is a result this project would like to be true, which is what makes it the one to distrust.

## What Phase 8 covers

Hybrid compaction: structured state, a verbatim recent window, and a historical summary under one budget. See [hybrid.md](hybrid.md).

**Built against a measured failure rather than a principle.** Phase 5.8 found the Phase 5 baseline scoring identically to a plain one-call summary, and found that what compaction actually lost was exact values — a port number among similar ports, a row count from a tool result — which full context kept. Structured state is the mechanism that can carry an exact value through intact, because an extracted `Fact` keeps the number as written instead of hoping a summarizer repeats it. The bet is narrow and falsifiable: if the adversarial benchmark disagrees, the answer is to delete this rather than add to it.

**It lives outside `compaction`, because trying to put it inside produced an import cycle** — `compaction` importing `extraction`, which already imports `compaction` — and the cycle was the design telling the truth. `compaction` and `extraction` are peers; hybrid composes both and sits above them.

**The first priority order was wrong, and only running it showed that.** Facts sat second from last, reasoned as "a fact without its supporting decision is a number with no reason to be trusted". On `confusable-numbers`, extraction found the fact carrying the port numbers and the budget then dropped it, keeping a goal, a formatting decision, and a task — losing precisely what the phase existed to preserve. With facts moved to third, hybrid keeps `8082` and `17` where the baseline loses both. A priority order argued from plausibility will quietly contradict the measurement that motivated the work.

**The exit criterion is not met, and the reason is the benchmark rather than the algorithm.** A four-arm matrix ran 60/60 cells clean. Read naively it says hybrid beats the Phase 5 baseline — 1.00/0.92 against 0.92/0.85. Then the same configuration was compared against an earlier identical run, and **run-to-run variance came out at ±0.08 on arms that did not change at all**: `simple_summary_v1` moved +0.08 on both measures, `full_context` and `phase_5_baseline` moved −0.08 on completion. With 13 evaluable scenarios one flipped answer is 0.077, which is both the noise and the entire claimed effect. Deterministic failure counts say it more plainly: 0, 1, 1, and 2 failures across four arms of 15 scenarios support no ranking. The cheapest arm — a single-call plain summary — took the only perfect score.

**What Phase 8 does establish is a mechanism, not a score.** Against `gpt-oss-120b`, hybrid preserved `8082` and `17` on the two scenarios where the baseline lost both, which is a direct observation of state carrying an exact value through compaction and does not depend on sample size. It is also model-dependent: the demonstration used a 120B model, the benchmark an 8B one, and hybrid's single deterministic failure in the benchmark was answering `"Port 8082."` — the right value, failed on the substance floor for being two words.

**It is deliberately the most expensive arm**: two model calls per context against the baseline's one, and the extraction call is added to the reported bill rather than omitted. A failed extraction produces the baseline with an empty state section *and a warning saying it must not be read as hybrid*, because an arm that is sometimes hybrid and sometimes not, averaged under one name, would make a benchmark row a lie.

Phase 8 also found a Phase 6 bug that no Phase 6 test caught: `Entity` requires a `name` the extraction prompt never asks for, so **every extracted entity was being silently rejected** as an invalid record — silently because a rejection is an ordinary counted outcome rather than an error. Found by being the first code to construct an `Entity` directly.

## What Phase 7 covers

State and retention: incremental update, supersession resolution, and deterministic conflict detection. See [state.md](state.md).

Phase 6 turns a conversation into typed records once, over a whole session, producing state that is correct and immediately stale. This phase is what lets the second hour of a session cost the second hour rather than the whole thing again.

**A watermark has to be earned.** It advances only when an extraction over that range actually succeeded, because one moved on a malformed reply would skip a region of the conversation permanently with nothing downstream able to tell. It never moves backward, an extraction that found nothing still advances it — reading a range and finding no state is success — and nothing-new returns `None` rather than an empty result, since "nothing was read" and "a range was read and yielded nothing" are different facts.

**Old state is an input, not only an output, and leaving that out was the phase's real mistake.** The first incremental run against a real model read one new event, produced a correct new decision and a correct new constraint, and linked neither to the decision they replaced — because a model shown only the new messages has never seen the earlier decision and cannot name it. The reversal became two orphaned assertions beside a still-active predecessor. Feeding current state into the prompt, verbatim because supersession is matched literally, produced the correct result: `0 new, 0 unchanged, 1 superseding`, PostgreSQL retired and SQLite current.

**Reconciliation returns a plan, not an effect.** Nothing is written; a caller inspects, resolves what it cannot accept, and commits. That is what lets a conflicting update be refused rather than half-applied. Identity is type plus normalised content — not the id, which is minted per extraction and would make every re-run look like new state.

**Nothing is deleted.** A superseded item keeps its content, rationale, sources, and timestamps; a retired decision that lost its reason would defeat the point of keeping it.

**Conflict detection is deterministic and narrow.** Contradictory facts sharing a subject, supersession forks, missing predecessors, and duplicates. Facts without a subject are not compared, because comparing arbitrary sentences is a language question and answering it with a model would make state acceptance nondeterministic — worse than the LLM-gate objection this project already raised, since the output here is the state itself rather than a report. **Only exact duplicates auto-resolve**; everything else blocks, because silently keeping the newer item is a policy nobody chose and looks identical to having found nothing.

Not done here: storage integration, snapshots, and reconstruction-at-a-point. The repository already enforces what a commit would need — append-only, acyclic supersession, session-local references — so wiring it is assembly rather than design.

## What Phase 6 covers

Structured context extraction: conversation to typed work state, and a validator that checks it against the source. See [extraction.md](extraction.md).

The first layer that **interprets**. Import asks what happened; compaction asks what to keep; extraction asks what it means — that this sentence is a decision, that one a prohibition, and that the second overturns the first. Everything before it could be checked against its source mechanically, and this cannot, which is why it reports its own losses and ships a validator.

The Phase 1 data model already had the targets, so no record type was added. `Goal`, `Constraint` with hard/soft, `Fact`, `Decision` with a required rationale, `Task`, `OpenQuestion`, `Entity` are filled; `Preference`, `Event`, and `Artifact` deliberately are not, because nothing consumes them and a type the prompt names is a type the model will produce whether or not it means anything.

**Provenance points at the archive, which resolved a question open since Phase 4.** `StateItemBase.sources` speaks SQLite's `msg_`/`ev_` naming; the archive names records its own way, and nothing ever linked the two. An extracted item now carries archive record ids and sequence numbers on its wrapper, and `sources` stays empty. Minting `msg_`-shaped ids would assert a correspondence no code maintains, and a reference that resolves to nothing is worse than one that is honestly out of band. An item citing nothing is rejected outright.

**Nothing the model says is trusted.** Both providers refuse `structured_output`, so JSON is prompted for and parsed defensively: prose and fences are stripped, unknown types counted, decisions without rationales rejected, `true` as a confidence turned into 0.5 rather than maximum certainty. **Truncated JSON is never repaired** — guessing what a half-written object meant is how an extraction invents state — and a malformed reply is a recorded failure rather than a silent empty result, because those look identical in a file and mean opposite things.

**Validation is deterministic and calls no model.** One that asked a model whether the extraction was right would inherit the failure it exists to detect. It checks grounding, not completeness, and says so: an extraction that found nothing validates clean, correctly, because it asserts nothing unsupported. Recall is what the Phase 5.5 scenarios measure.

Verified against a real model on benchmark scenarios. `reversed-decision` produced both sides of the reversal — PostgreSQL superseded, SQLite active, each with its rationale — and `negative-constraint` produced "Do NOT use Redis" as a hard constraint. Both validated clean.

Known limits, recorded rather than worked around: one model call means the whole conversation sits in memory and in the prompt, so a session larger than the context window cannot be extracted this way; nothing is written to storage; and there is no incremental extraction. All three need the supersession and conflict machinery that is Phase 7's.

## What Phase 5.8 covers

The first benchmark run against a real model, the machinery that made running one against a metered free tier possible, and a failure taxonomy over the results. See [benchmark.md](benchmark.md).

**A complete matrix was executed and a comparison is claimed — a negative one.** Dataset `v3`, budget 160, `groq/llama-3.1-8b-instant`, judged, 45/45 cells with every arm on every scenario, `check_comparable` clean.

**The Phase 5 baseline does not beat a plain summary.** Identical deterministic scores, worse on judged questions, more tokens spent for a context of the same size. Phase 5 called itself the control that later work has to beat; against one-call summarization it has not beaten anything. That is a finding about the problem rather than a defect to patch, and it is exactly what building the control was for.

**Compaction cost more than it saved.** 705 tokens end to end for full context against 1222 and 1306 for the compacted arms, at 2.5x smaller context. At these conversation lengths the summarization call outweighs what it saves; where that reverses is unmeasured.

**What compaction lost was exact values.** Two deterministic failures in the whole matrix, both on compacted arms: a port number among similar ports, and a row count from a tool result. Full context lost neither. A specific reproducible failure mode is a better target for a next algorithm than "retention was 0.92".

**A benchmark's discriminating power depends on the model under test.** An earlier complete matrix on a 70B model scored 1.00 on everything for every arm — a ceiling effect that looks exactly like a measurement. Dropping to an 8B model is what made the differences visible.

**Free-tier limits differ in kind, not only in size.** OpenRouter caps requests per day; Groq caps requests per day, tokens per day, and tokens per minute, per model. Only the per-minute limit is solvable by pacing, and a run that assumes otherwise discovers it at request fifty-one. Groq models are approved on `ACCOUNT_TIER` evidence — the owner's word about their own plan — which is weaker than a published zero price and is marked as such, and is not portable to another account.

The machinery: a request budget that stops a run at a ceiling rather than letting the remainder collect 429s and be filed as failed strategies; results streamed and flushed so a stop keeps what it paid for; resumption keyed on `(run_id, scenario_id, strategy)` counting only successes; a scenario-coverage check in `check_comparable`; and the fifteen-class failure taxonomy, three of whose classes report as unmeasured rather than zero.

Two scoring defects were found, both invisible to deterministic fakes and both caught only by a real model. Negative evidence by substring made 8 of 10 findings artefacts, repaired as dataset `v3`. A judged question phrased so that the desirable answer was NO scored correct answers as failures, caught by reading a judge's reasoning rather than its verdict.

## What Phase 5.7 covers

The first real implementations behind the Phase 4.5 boundary: Groq and OpenRouter, plus the free-model allowlist that decides which of them the benchmark may call. See [llm.md](llm.md).

The interface did not change. Both providers implement `LLMProvider` as written in Phase 4.5, post JSON over `urllib` rather than adding a vendor SDK, and share their wire dialect through a private helper rather than a second abstraction. Transport is injectable, so every provider test exercises the real translation and error mapping against canned responses without opening a socket. The suite still runs with no key and no network.

**A credential is not a licence to spend.** `open_context.llm.free_models` is an explicit allowlist, each entry recording the date its price was read as zero and the endpoint it was read from, and the check runs in the provider constructor before the credential is read — so an unapproved model costs zero HTTP requests, proven by a counting transport. A `:free` suffix is never consulted; it is a naming convention and a model that stopped being free would keep its name. There is no paid fallback and no fallback mechanism of any kind: an unavailable free model fails, and choosing another approved one is the caller's explicit act.

**No Groq model is approved, and that is a finding rather than an omission.** Groq prices every chat model it serves; its free tier is an account-level allowance against priced models, and nothing in the API distinguishes "within my allowance" from "billed". The provider works; the policy is what stops it, so `approved_models("groq")` is empty and the smoke test skips rather than reaching for something priced.

**The smoke test is one real request, and it ran.** On 2026-08-13, five of the six approved OpenRouter models answered a live generation through the ordinary interface — `openai/gpt-oss-20b:free`, `liquid/lfm-2.5-2.6b:free`, `nvidia/nemotron-3-nano-30b-a3b:free`, `google/gemma-4-26b-a4b-it:free`, and `nvidia/nemotron-3-super-120b-a12b:free`. `google/gemma-4-31b-it:free` returned 429 and was recorded as rate-limited, not retried elsewhere. All six were re-confirmed at `pricing.prompt == 0` and `pricing.completion == 0` on the same date. Groq skipped. The test is marked `llm`, skips without a credential, and asserts no key reaches a result, a `repr`, or an error.

**No benchmark was run against a real model in this phase, and no result is claimed.** A working provider is the unblocking of Phase 5.8, not a measurement.

## What Phase 5.6 covers

The first benchmark: a matrix of scenarios, strategies, and token budgets driven through the Phase 5.5 runner, with task-completion metrics, cost accounting, and a report that separates what survived from whether the agent acted on it. See [benchmark.md](benchmark.md).

**No real-model benchmark was executed in this phase, and no result is claimed.** When 5.6 was built the repository had no verified provider — Phase 4.5 shipped fakes only, because no vendor API could be verified offline — so the benchmark runs in deterministic mode, every row records `evaluation_mode`, and every report says at the top that fake-provider numbers measure the harness rather than any strategy. Phase 5.7 supplied the provider; running the matrix through it is Phase 5.8.

Dataset `v2`, 15 scenarios, built by attaching task-completion criteria to `v1` without editing it; `v1` is asserted unchanged. Budgets 80/160/240, chosen below every scenario so compaction genuinely happens. Three arms: full context, `simple_summary_v1`, and the Phase 5 baseline, the latter two untouched.

**The reference is budget-independent.** Full context ignores the target, so it runs once per scenario and repetition rather than once per budget — otherwise three independent samples of one condition get averaged into a row claiming to be budget-independent, which against a real model is three draws wearing one label. The matrix is `scenarios x repetitions x (references + compacted x budgets)`, giving 105 runs by default: 15 full context, 45 simple summary, 45 Phase 5. Every budget is compared against the same reference population.

**Retention and task completion are never one number.** Retention asks what survived; completion asks whether the agent chose correctly, and is conjunctive, discriminating against the options the conversation ruled out, with a substance floor. Omission and fabrication are distinguished, and criteria needing code execution report `not_evaluable` rather than being guessed.

Nothing was designed from the results, because there are none to design from.

## What Phase 5.5 covers

The evaluation harness: scenarios, strategies, a continuation runner, deterministic metrics, a versioned synthetic dataset, machine-readable results, and a small module CLI. See [evaluation.md](evaluation.md).

**The primary evaluation target is downstream task continuation, not summary quality.** A summary that reads well, or that contains many true facts, is not evidence. The question is whether a different agent given only the compacted representation can carry the work forward, so every arm runs the same task through the same model with the same prompt and budget, and only the context representation varies.

Three properties carry the phase. **The runtime does not depend on the harness** — `open_context_eval` sits beside `open_context`, imports it, is never imported by it, and is excluded from the wheel; checked by a test that parses every runtime module. **Leakage is structurally impossible** — the function that builds a downstream request takes two strings and has no access to a scenario or an archive, so neither an expected answer nor a removed conversation can reach a model. **A failure is not a zero** — every way a run can fail has a status and carries no metrics, so a broken experiment never reads as a bad strategy.

Three arms exist: full context, `simple_summary_v1` — a plain-summarization baseline that shares no compaction logic with Phase 5 — and the Phase 5 baseline called through its public interface and not modified. Retrieval and MemHandoff are deliberately absent rather than stubbed.

**Retention checks are not task completion.** The deterministic checks ask whether an expected fact, constraint, or decision survived into the response. Whether the agent continued the work well is a different question, is unmeasured, and no threshold for "almost as effectively as full context" is chosen.

**No results are claimed.** The harness produces the machinery to generate numbers and ships none.

## What Phase 5 covers

The first compaction engine: `BaselineRecentPlusSummary`, which summarizes the older part of a session through `LLMProvider` and keeps the most recent part verbatim, holding the whole result to a measured token budget. See [baseline-compaction.md](baseline-compaction.md).

**Phase 5 is a baseline intended for measurement. It is not the final MemHandoff compaction algorithm.** It is roughly what a competent developer would build in an afternoon, which is the point: it is the control that later work has to beat, and where it loses a constraint stated once in message 40, that loss is a finding rather than a defect to patch here.

No structured extraction, no importance scoring, no retention, no retrieval, no embeddings, no graph, no portable package. A conversation saying "we chose Postgres because the infra already runs it" produces a summary that says so and never a `Decision` record.

Three properties carry the phase. **The budget is enforced against a measurement, never against the model's word**: the summary is tokenized, compared to its budget, retried once, and cut deterministically if it still overshoots. **The archive is never modified**, pinned by hashing every file in the session directory before and after. **An estimate is never presented as exact**, inherited from Phase 4.5 and surfaced as a warning when the budget was enforced against one.

Reads the archive through range and positional access rather than loading it, reaches a model only through `LLMProvider`, and runs entirely offline against the deterministic fakes.

## What Phase 4.5 covers

The provider-neutral boundary future phases will call a model through: `LLMProvider`, `Tokenizer`, `ModelInfo` with a capability set, provider-neutral errors, a schema-based structured-output contract, and a small registry and environment configuration. See [llm.md](llm.md).

At the end of this phase Open Context has a provider-neutral interface through which future verified model implementations can be used, along with token counting and model capability abstractions. Only deterministic fakes are implemented, so nothing here reaches a real model. It cannot compact a conversation either, and nothing here tries to: no compaction, no summarization, no extraction, no retention, no retrieval, no ranking, no message selection, no context-pressure detection.

Two invariants carry the phase. **An estimate is never presented as an exact token count**: every count records how it was produced, and adding an exact count to an estimate yields an estimate, so the guarantee is enforced by arithmetic rather than by discipline. **The core never depends on a provider**: `archive`, `models`, `storage`, and `importers` do not import the LLM layer, the LLM layer does not import them, and no package imports a vendor SDK — all checked by tests that read the source rather than by convention.

No real provider was implemented. No vendor API could be verified from this environment, and an HTTP client written against a remembered request shape looks correct and fails against the real service. Only deterministic fakes ship, which is why the suite runs with no key, no network, and no model server. The registry makes a verified provider an additive change.

## What Phase 4.1 covers

Four corrections to Phase 4, no new capability.

**Synthetic identity is scoped to a logical source.** An event with no id of its own previously took a position counted from the start of the import, so the first unidentified event of every import into a session got the same id — which the archive reads as one item written twice. Identity is now `synth-<fingerprint of the source key>-<position>`.

The source key is the identity of the *logical* source, which is a caller's judgement and not a property of any bytes: two byte-identical exports may be one source or two, and only the caller knows which. A first attempt used the content digest as the key and so silently merged identical files; `source_key` is now explicit, with `import_path` defaulting it to "this file, holding this content, at this location" and documenting exactly that. Distinct keys give distinct ids collision-resistantly, one key keeps its ids everywhere, nothing is random, and nothing derives identity from `seq`.

**Imports are explicitly non-atomic.** The archive has append and extend and no delete, truncate, or rollback, and a staging layer would mean writing every record twice and renumbering every `seq` on commit to buy a guarantee a live stream could never use. A failure now raises `IncompleteImportError` carrying the report of what was written, and the record that caused it is never among them. Atomicity, where a caller wants it, is a session of its own and an `rmtree`.

**The generic tool-role mapping is documented as a convention.** `role="tool"` still reads as a tool result; a provider adapter must classify tool events from its own verified format rather than inherit that.

**`raw` is the provider record as parsed, with unknown fields preserved.** Wording implying byte-level source preservation is corrected everywhere. No field is dropped; representation is not kept.

## What Phase 4 covers

Conversation import adapters: a provider-neutral normalized event, an importer interface, a generic JSON Lines adapter, and a streaming pipeline that writes events into the archive. See [import.md](import.md).

Ingestion, not understanding. An importer answers "what happened in this conversation" and never "what does this conversation mean". No message becomes a `Decision`, a `Goal`, a `Task`, or any other state item, and no LLM is called.

Every event keeps the provider record as parsed in `raw`, so a field the normalized model has no home for survives ingest rather than being discarded at the one point where it was still available. `raw` preserves unknown fields, not the source bytes: representation such as key order and number spelling is lost in parsing, and byte-level preservation is deliberately not claimed.

No provider-specific adapter was written. Claude Code, Codex, and Gemini transcript formats are not represented in this repository and were not available to inspect, and an adapter written from memory parses a format nobody has confirmed exists. They are recorded as open questions below rather than guessed at.

No compaction, no extraction, no retrieval, no LLM provider, no package format.

## What Phase 3.1 covers

Two corrections to Phase 3, no new functionality.

`Archive.open` now checks that the manifest's `session_id` matches the session asked for, and raises `ManifestMismatchError` when it does not. The manifest's format and version were already validated; its identity was not, so a copied or renamed session directory would open and serve one session's conversation under another's name.

The single-writer and durability invariants are now stated explicitly, in the module, on the class, and in [archive.md](archive.md). They are documented rather than implemented on purpose: no lock, writer process, or queue was added, because the right coordination model depends on how agent adapters write and that is Phase 4's to design.

## What Phase 3 covers

The local immutable archive: append-only JSONL on the filesystem, one log per session, with a rebuildable offset index, per-record hashing, bounded tail recovery, and streaming reads. See [archive.md](archive.md).

The archive holds raw conversation; SQLite holds indexed metadata and interpreted structure. The conversation is not duplicated into SQLite.

Two invariants are stated rather than enforced, and both are Phase 4's to revisit. **A `SessionLog` has a single writer; concurrent writes to the same session are unsupported in Phase 3.** **The data file is the durability boundary; the index is derived data and may lag, and recovery can rebuild it from the data file.** No lock, no writer process, and no queue exists, because the right coordination model depends on how agent adapters write and that is not yet designed.

No compaction, no extraction, no retrieval, no LLM provider, no package format, no adapters.

## What Phase 2.1 covers

Three corrections to Phase 2, no new functionality.

Session locality is now enforced on every reference. Migrations are genuinely atomic: the first implementation used `executescript`, which issues an implicit COMMIT before running and left a failed migration's completed statements applied. Each migration now runs as individually executed statements inside one explicit transaction. The `raw source -> evidence -> state` provenance layering needed no schema change and is now pinned by tests and documented.

A supersession cycle inside a single batch was also found and rejected.

## What Phase 2 covers

SQLite persistence for the Phase 1 models. Append and read only: no update, no delete. Referential integrity, atomic multi-record writes, and explicit versioned migrations. See [storage.md](storage.md).

No compaction, no retrieval, no embeddings, no LLM provider, no package builder, no compiler, no MCP, no CLI.

## What Phase 1 covers

The record types and their validation rules. See [data-model.md](data-model.md) for the hierarchy and the decisions taken.

No storage, no LLM provider, no compaction, no retrieval, no MCP, no CLI.

## What Phase 0 covers

Packaging, the test harness, lint and type configuration, CI, licensing, and the third-party reuse process.

Phase 0 deliberately contains no application logic. There is no database, no MCP surface, no LLM call, no CLI entry point.

## Resolved

**Is this a memory system or a handoff tool?** A handoff tool. The portable package is the product; the archive and state graph are engine components. Cross-session memory is out of scope.

**When does evaluation happen?** Phases 5.5 and 5.6, directly after the baseline compactor.

**When does the provider abstraction land?** Phase 4.5, before anything needs an LLM call or a token count. Done: `LLMProvider` and `Tokenizer`, with the core's independence from any provider enforced by tests.

**Phase 2 question: are cross-session references allowed?** No, resolved in Phase 2.1. The session state graph is session-local and `CrossSessionReferenceError` is raised otherwise. Knowledge crosses sessions as a portable context package instead.

**Phase 1 question: is `Message.seq` unique per session?** Yes. `UNIQUE (session_id, seq)` in the schema, with `SequenceConflictError` raised on collision. This never arose in Phase 4: import writes archive records, whose positions the archive assigns and which allow a repeated source id outright. It returns whenever a phase first turns imported events into `Message` rows.

## Open questions

Product-level questions live in [product-architecture.md](product-architecture.md). These are engine-level.

These affect layers beyond the one being built. They are recorded rather than answered so the decision is deliberate and dated.

**Which providers count tokens exactly, and how are non-text tokens counted?** Phase 4.5 built the honest representation — exact, estimated, or unavailable — but verified nothing. Which providers offer reliable exact counting, how image and tool-call tokens should be represented, and how a provider's own chat template changes a count are all unmeasured. Recorded in full in [llm.md](llm.md).

**Does generic summarization preserve what matters?** The whole reason Phase 5 is a baseline rather than a contribution. Phases 5.5 and 5.6 built the apparatus to answer it, and Phase 5.7 supplied the verified provider it was blocked on, so the question is no longer blocked — only unanswered. Whether a generic summary keeps decisions and their rationale, negative constraints, failed approaches, exact numbers, temporal state, and information stated once early is still unmeasured. The full list is in [baseline-compaction.md](baseline-compaction.md); Phase 5.8 runs the matrix that addresses it.

**Negative evidence by substring scores correct answers as failures.** Found in Phase 5.8, against a real model, and the most consequential thing that run turned up. Both halves of the deterministic scorer assert that something is *absent*: `RetentionCheck.forbidden` and `CompletionCriterion.alternatives`. A model that honours a constraint by naming what it rejected ("Since Redis is explicitly prohibited, I recommend an in-memory cache") contains the term and fails both. Eight of the run's ten failure findings are this artefact, manually verified; only two were real. Invisible through Phases 5.5 and 5.6 because the fake provider's one fixed reply never contained any prohibited term.

**The bias favours compaction over the reference.** Naming a rejected option requires knowing it was rejected, so the arms that retained most are likeliest to trip it, and full context is penalised for being better informed.

**Repaired as dataset `v3`, which is built.** A cleverer pattern was rejected: `RetentionCheck`'s own documentation already says the honest answer to "it cannot tell a mention from a use" is a judged question, and a heuristic skipping terms near a negation would silently reclassify real failures too. `v3` applies one rule — deterministic checks assert presence, judged questions assert absence — removing all 20 negative substring assertions and replacing each with a question that reports *not evaluated* without a judge. `v2` and `v1` are left unedited and asserted so by test.

**The exact-value ledger is built and mechanically verified; its benchmark effect is unmeasured.** 2026-08-24.

Every deterministic failure in the `v4` matrix is an exact value — the write timeout in `adv-exact-values`, the port in `adv-tool-heavy` — on every compacted arm and on neither reference arm. Nothing else fails anywhere. The mechanism is stated by a failing answer itself: "the original context does not specify the exact value", followed by an invented one.

`preserve_literals` lifts those values out deterministically, before the model is asked for anything, and carries them beside the summary. Against the real `adv-exact-values` conversation, with a summariser that returns prose containing no digits — which is what the real run produced — the baseline loses `74` and the ledger arm keeps it, labelled. That is `tests/integration/test_literal_ledger_on_dataset.py`, offline and deterministic.

*Whether it moves the benchmark score is unmeasured, and the reason is structural rather than circumstantial.* Two measurements bound it.

**The failure only exists above roughly 9x compression.** Re-running `adv-exact-values` and `adv-tool-heavy` at budget 900 — 3.5x to 4.3x on these conversations — every arm passes every check, the plain `phase_5_baseline` included. At that ratio the summariser has room to keep the number, so there is no failure for a ledger to convert. The published failures are all at budget 160, where compression is 9x to 14x.

**No approved free model can run at the budget where the failure appears.** The compaction budget doubles as the model's output cap. Groq's three models all reason before answering, so their summary share — `0.6 x budget` — must clear `REASONING_FLOOR`, which needs a budget above roughly 860. Every budget in the published matrix is less than a third of that. The two non-reasoning models on the allowlist are OpenRouter's gemmas, and both returned upstream 429s across four attempts on two days.

So the arm is registered, its mechanism is verified offline against the real conversation, and its score is recorded as unmeasured. Measuring it needs one of three things this project does not currently have: a free non-reasoning model that is not rate-limited, conversations long enough to reach 9x at a budget above the reasoning floor, or a paid model, which policy excludes.

That last option is the honest framing of the gap. It is not that the experiment failed; it is that the free tier cannot express the operating point the failure lives at.

**The judge is a capacity floor, and the harness was holding it below the floor.** Measured 2026-08-24 by rejudging the reference arm's stored answers from the `v4` run, so no compaction or continuation was re-run.

Two separate defects, one masking the other.

*The judge could not read a rejection.* On the `v4` run an 8B judge passed **9 of 24** reference questions while the same arm passed every deterministic check. Its own reasoning shows why: it failed `encoding-not-hedged` because the answer "mentions UTF-8", when the answer named UTF-8 only to say the mainframe cannot parse it, and failed `no-correlated-subquery` because the approach was "listed as one of the potential approaches", when the answer cited it as the one that timed out. That is the mention-versus-use error `v3` removed from the substring scorer, reappearing in a grader too small to make the distinction. It is not a prompt gap: `JUDGE_PROMPT_V1` states the rule outright, and every affected question restates it. A 70B judge on the identical prompt passed **8 of 8**.

*And the harness could not run a bigger judge.* `LLMJudge` asked for `max_output_tokens=200`. Six of the eight approved free models reason before answering and need at least `REASONING_FLOOR`, so the provider layer refused every one of them — including both 120B-class models. The only judges that would run were the two smallest non-reasoning models, which is exactly the class that fails the control. The instrument was structurally pinned to being unreliable, and the earlier "rejudge with a stronger model" advice could not be followed.

*Both are fixed.* The judge now derives its cap from the model's measured reasoning overhead, the way the Phase 5.7 smoke test does. Re-running the control on free models: `groq/openai/gpt-oss-120b` passes **8 of 8**, `openrouter/nvidia/nemotron-3-super-120b-a12b:free` passes **7 of 7** with one undecided — a parse failure where the model wrote prose ahead of its verdict, correctly reported as undecided rather than as a failure. Judged questions are therefore measurable again, on free models, and the gating dependency below is answered.

**What remains open is the judge.** `v3` without one measures less than `v2` appeared to, because hedging is no longer caught deterministically, and two scenarios — `negative-constraint` and `failed-approach` — carry no deterministic evidence at all, having stated a prohibition without ever saying what to use instead. A real `EvaluationJudge` is now the gating dependency for the benchmark rather than a refinement of it, and it costs a request per question, which is its own quota problem. See [benchmark.md](benchmark.md).

**Validation is a nondeterministic gate.** The compaction safety check rejects a snapshot when validation fails, and the validator is LLM-based. That makes snapshot acceptance nondeterministic at the margin and unusable as a regression signal across commits. Proposal: the hard gate is deterministic (required state item IDs present, provenance references resolve, token budget satisfied), and the LLM retention score is reported but does not block.

**Automatic compaction has no control point.** Proactive compaction assumes the runtime can observe context usage and intervene mid-session. MCP is pull-based: the agent decides when to call. This now blocks the Phase 11 adapters rather than a specific phase, and is why integration research precedes implementation.

**Short name collision.** The proposed CLI name `ocr` collides with optical character recognition in search results, package indexes, and documentation. No entry point is defined yet, so the decision is still free. Candidates: `octx`, `opencontext`, `ctxrt`.

**Cross-machine provenance breaks referential integrity.** Storage requires every provenance reference to resolve to a stored record. A package imported on another machine cites messages that do not exist there, so import fails as written. This blocks Phase 9 and is tracked as the first open question in product-architecture.md.

**Reads load whole result sets in SQLite.** `list_messages` materializes every message in a session. Phase 3 solved this for the archive, where the bulk of the content lives, but the SQLite side still needs streaming or paged reads.

**No provider transcript format has been verified.** Claude Code, Codex, and Gemini adapters are unwritten because their real exports were not available to inspect. Each needs its actual transcript or hook output in hand: where it lives, whether it is a file or an API, whether it is line-delimited, how content blocks are shaped, how tool calls and results are correlated, and whether the format is versioned. This blocks Phase 11 more than it blocks the phases in between, since a converted export imports through the generic adapter today.

**Nothing links the archive to SQLite yet.** A `Message` row and its archive record refer to the same thing by id with no enforced correspondence. Phase 4 declined to decide this: import writes only the archive, and the link is left as a shared session id by convention. Whichever phase first writes both stores has to define the order and what happens when one write succeeds and the other fails.

**The archive has no write-coordination model.** Single-writer is an invariant of Phase 3, not a guarantee it enforces. Phase 4 did not need to resolve it, since an import is one writer for the duration of one call. The choice between a lock file, a single writer process, and a serialised queue waits until agent adapters show whether writers actually contend — whether an adapter streams during a live session, and whether a CLI can run against a session an agent is holding open.

**`ContextSnapshot` is not a portable package.** It stores identifiers only and deliberately holds no content, so it is meaningless off the machine that owns the archive. The package format is a separate artifact that inlines what it needs. Both should exist; neither should be renamed into the other.
