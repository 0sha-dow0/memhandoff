# Contributing

## Before you open a pull request

The current priority is making a real Agent A to Agent B handoff easy to try,
understand, and criticize. The most useful contribution is usually a small
reproducible context-loss case, an importer for a real agent export, a usability
fix, or clearer documentation.

Open an issue before starting a large architecture or research change. V2
research and additional benchmark experiments are paused unless they directly
unblock usability or credibility. Existing negative and inconclusive results
remain published; do not retune scenarios or methodology to improve a score.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

All four must pass. CI runs the same ones.

```bash
pytest
ruff check .
ruff format --check .
mypy
```

## Code

Python 3.12 or newer. Full type annotations; mypy runs in strict mode. Small modules, single responsibility, dependencies injected rather than imported at the point of use.

Provider-specific logic stays out of the core. If a module in `models/`, `storage/`, `graph/`, or `compiler/` imports an SDK for a specific model provider, that is a design bug. Providers sit behind the `LLMProvider` protocol.

Every behavior change ships with a test. Tests that need a network call or an API key are marked `llm` and skipped by default.

## Ways to contribute

Ranked by how much they help, which is roughly the reverse of how much code they
involve:

| | |
| --- | --- |
| **Report a failure** | A session where context was lost or a receiving agent could not continue. Use the [Context Loss Report](.github/ISSUE_TEMPLATE/context-loss.yml) template. This is the most useful thing you can send. |
| **Improve the first handoff** | Remove friction from the quickstart, example, CLI messages, or feedback path. |
| **Add an adversarial scenario** | A case the current dataset cannot catch. See [docs/adversarial.md](docs/adversarial.md) for what is already covered. |
| **Improve an importer or a compile target** | Adding an agent format needs `detect` and `read`; adding a target needs one function that shapes an already-budgeted context. |
| **Propose a compaction strategy** | Discuss it first; research expansion is not the current default. If accepted, it must be evaluated against the existing baselines. |
| **Improve documentation** | Particularly anywhere the reasoning is missing rather than the facts. |

You do not need a credential for any of it. The suite is offline, and the
benchmark charts redraw from a run already in the repository:

```bash
pytest && ruff check . && ruff format --check . && mypy
python benchmarks/plot.py
```

## Compaction strategies

A strategy turns a conversation into a context that fits a budget. They live in
two places, and the split matters:

| | |
| --- | --- |
| `src/open_context/compaction/` | the Phase 5 baseline — recent window, budget enforcement, chunk-then-combine summarization |
| `src/open_context/hybrid/` | hybrid compaction, which composes the baseline and the extractor through their public interfaces |

Hybrid sits outside `compaction/` because putting it inside produced an import
cycle — `compaction` importing `extraction`, which already imports `compaction`.
The cycle was the design telling the truth: the two are peers, and anything
composing both belongs above them.

A strategy is exercised twice. Unit tests pin its behaviour offline against the
deterministic fakes in `open_context.llm.fakes`; the benchmark then runs it as an
*arm* against the same scenarios, model, prompt, and budget as every other arm.
Register it in `open_context_eval.strategies.default_strategies()` and it is
compared automatically — tests derive the arm count from the registry rather than
hardcoding it, so adding a fifth will not break them.

**A strategy that cannot say what it did is worse than one that did less.** When
an inner step fails, say so: `HybridCompaction` returns the Phase 5 baseline with
a warning that the result must not be read as hybrid, and the benchmark excludes
those cells from that arm's score. An arm that is sometimes hybrid and sometimes
not, averaged under one name, makes a benchmark row a lie.

### Evaluating a new strategy

A new strategy is compared against the **existing** baselines on the **existing**
scenarios, at the same budget, with the same model and prompt. Register it in
`default_strategies()` and the benchmark picks it up as another arm.

Do not adjust scenarios, metrics, budgets, or judge prompts so that a new
strategy looks better. If a scenario is genuinely wrong, fix it as its own change
with its own reasoning, and re-run every arm afterwards — including the ones the
fix makes look worse.

### Why a benchmark regression matters

The scores are the only evidence this project has that any of it works. A change
that improves a score is interesting; a change that moves one without an
explanation is a reason to stop and find out why.

Run the benchmark before and after any strategy change, and put both numbers in
the pull request. If the result is worse, say so — negative results are kept in
this repository on purpose, including a run that was voided entirely and
documented rather than deleted.

Do not adjust scenarios, metrics, or judge prompts to improve a score. If a
scenario is genuinely wrong, fix it as its own change, with its own reasoning,
and re-run everything.

## Benchmark methodology

Four arms are compared under identical conditions — same scenarios, model,
continuation prompt, budget, and repetition count:

| Arm | What it is |
| --- | --- |
| `full_context` | the whole conversation, uncompacted. A **reference**, not a competitor: it is budget-independent and exists to prove the information was recoverable at all |
| `simple_summary_v1` | one summarization call. The bar every strategy has to clear |
| `phase_5_baseline` | the Phase 5 compactor |
| `hybrid_v1` | structured state, a verbatim recent window, and a historical summary under one budget |

**Retention and continuation quality are different measures and are never
merged.** Retention asks whether an expected string survived into the response —
a deterministic check with no model in the loop. Completion asks whether the task
was actually done, and judged questions ask a model things a string check cannot
settle. A run reports all three separately because a response can carry every
fact and still fail the task, or complete it while dropping a constraint.

Judged questions assert *absence*; deterministic checks assert *presence*. That
rule exists because the reverse made eight of ten findings false in one run: a
correct answer that names what it rejected fails a substring check, and does so
more often the better informed it is.

**The judge is calibrated against the reference arm.** If `full_context` passes
its deterministic checks and fails its judged questions, the judge is what is
being measured — the report says so and its judged scores must not be read.

Adversarial scenarios (`dataset v4`) are long enough that the budget forces a
real choice: 1,400–2,300 tokens at 9–14x compression, with critical information
planted early, in the middle, and late in otherwise identical scenarios so a
position effect can be told from a content one. See
[docs/adversarial.md](docs/adversarial.md).

**Every run is reproducible or it is not a run.** A run id, a fingerprint of the
dataset, model, tokenizer, prompt hash, budget, and repetition index, and the git
commit are written into every result row. The repetition index is part of the
fingerprint precisely so two runs of one configuration are never filed under one
identity — Phase 8 needed exactly that to discover its between-arm gap was the
same size as its run-to-run noise.

## Internal testing policy: free models only

**The project's own automated and real-model testing uses free models only.**
Every model this repository calls comes from an explicit allowlist in
`open_context.llm.free_models`, checked before a request is built, so an
unapproved or paid model costs zero network calls and zero money. There is no
paid fallback: a rate-limited free model is recorded as blocked, never swapped
for a paid one.

**This is a development policy, not a limitation of the software.** The runtime
is provider- and model-agnostic. Users may point it at any provider they like,
paid or free, hosted or local — the allowlist governs what *this repository's own
tests and benchmarks* may call, and nothing else.

Contributors must not introduce paid API usage into automated tests or
benchmarks. In practice:

- Tests needing a network call or a key are marked `llm` and skipped by default.
- Anything else uses the deterministic fakes, so the suite runs offline with no
  credential.
- Adding a model to the allowlist requires calling it and recording what was
  observed — its price from the provider's own catalogue, and how it spends its
  output budget. `open-context verify-models` re-checks the billing half against
  the live catalogues, because a free tier can be withdrawn without notice and
  this one was: an entry stopped being free within ten days of being added.

## Proposing an architecture change

Anything that changes a format, a boundary, or a guarantee is worth discussing
before it is written. That includes the `.ctx` layout, the archive record shape,
the provider and tokenizer protocols, the compile targets, and where the portable
layer stops and a renderer begins.

Open an [Architecture Proposal](https://github.com/0sha-dow0/memhandoff/issues/new?template=architecture-proposal.yml)
saying what problem it solves, what it changes, what it breaks, what it costs,
and what you considered instead. A proposal that argues the current design is
wrong is welcome — say which constraint it fails.

Two things the design deliberately holds: the portable layer decides and the
target renderer only shapes, and no vendor SDK enters the dependency tree. A
change to either needs an argument, not just a patch.

## Never commit

Real conversation exports. Test fixtures are synthetic. A conversation you exported from your own account still contains other people's words, file paths, and often credentials.

## Reusing other projects' code

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) first. Short version: check the actual license file, preserve the notices, record the commit SHA and the exact modifications, and if the licensing is unclear, reimplement the idea instead of copying the code.

## Commits

Conventional Commits, so the changelog can be assembled from history.

```
feat(storage): add message repository
fix(compiler): respect token budget when packing evidence
docs(phases): record Phase 2 completion
```

## License

Contributions are licensed under Apache-2.0, per section 5 of the license.
