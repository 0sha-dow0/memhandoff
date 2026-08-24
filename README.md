# MemHandoff

> Local-first context handoff and compaction for AI agents.

![MemHandoff — context that travels, work that continues](docs/assets/memhandoff-hero.png)

You spend two days working with one agent. The conversation outgrows the context
window. The agent compacts it with its own provider-specific mechanism, and that
compacted state is stuck inside that provider. Moving to another agent means
starting the explanation from scratch.

Summarising the transcript and deleting the original does not solve it either. It
loses the API contract agreed in message 40, the reason Kafka was rejected in
message 180, and the constraint stated once and never repeated.

MemHandoff compacts a session into a **portable context package** — a structured,
provider-neutral artifact another agent can continue from. The original
conversation stays on your machine.

*The Python package is `open-context-runtime` and the command is `open-context`.*

## Why MemHandoff?

Provider-specific compaction is a dead end for the user. When an agent runs out
of window it summarises the session with its own mechanism, and whatever it kept
now lives inside that provider in a shape nobody else reads. Switching agents
means explaining the project again.

The bet MemHandoff is testing is that the useful residue of a long session —
what you are trying to do, what you must not do, what you decided and why, what
is still open — is **structured, small, and provider-neutral**, and can therefore
be handed over.

**Whether that bet pays off is not settled.** The benchmark below is where it is
being tested, and it currently says the compactor does not beat a plain summary.

## How It Works

![A session leaving one agent and arriving at another](docs/assets/memhandoff-identity.svg)

Three layers stay distinct, and the middle one is the only part that travels:

| Layer | What | Portable |
| --- | --- | --- |
| Raw conversation | the full history, hundreds of thousands of tokens | No — stays on your machine |
| Portable package | goals, constraints, decisions and why, open work, evidence | **Yes** |
| Agent context | what a specific model actually receives, compiled per agent | No — generated on demand |

The package is not an LLM summary. It carries typed records with sources,
statuses, and provenance, so you can ask why something was kept and get message
ids back.

## Architecture

```
Claude Code session
        |
   import + archive          append-only, hashed per record, never destroyed
        |
   extract + compact         typed state: goals, constraints, decisions
        |
    project.ctx              one JSON file, versioned, hashed, inspectable
        |
   compile per target        OpenAI / Anthropic / Gemini / local / generic
        |
another agent continues
```

## Integrations

| | |
| --- | --- |
| **Read from** | Claude Code sessions, generic JSON Lines |
| **Trigger on** | Claude Code `PreCompact` — capture automatically when the agent's window fills |
| **Extract with** | any allowlisted free model on Groq or OpenRouter |
| **Compile for** | OpenAI, Anthropic, Gemini, local, generic |

Compiling produces a request body in that family's shape. **This tool never calls
a model on your behalf** except during `--extract`.

Adding a reader means implementing `detect` and `read`. Adding a target means one
function that shapes an already-budgeted context — target adapters get no budget
and no tokenizer, so they cannot decide what to include, only how to lay it out.
See [docs/handoff.md](docs/handoff.md) and [docs/compiler.md](docs/compiler.md).

Detailed design notes live in [docs/](docs/README.md) — one document per layer,
each explaining what it does and which decisions were made against a measurement
rather than a preference.

## Portable Context

One JSON file, versioned, hashed, and readable by a person:

```json
{
  "manifest": { "format": "open-context-ctx", "format_version": 1, ... },
  "state":    [ { "type": "constraint", "content": "Never write to the NFS mount", ... } ],
  "evidence": [ { "state_id": "con_...", "archive_seqs": [41], "excerpt": "..." } ],
  "recent":   { "messages": [ ... ] },
  "archive":  { "record_count": 2772, "record_hashes": { "41": "9f2c..." } },
  "content_hash": "38164681a6a497eb..."
}
```

It **references** the archive rather than embedding it. Embedding would make the
package a second copy of the conversation that immediately begins to diverge; the
archive already exists on your machine and is already verifiable record by record.

A receiver holding the archive can prove the package describes *that* archive. A
receiver without it can still verify the package is internally intact and knows
exactly what it is missing. See [docs/ctx-format.md](docs/ctx-format.md).

## Quick Start

```bash
pip install open-context-runtime

open-context sessions                                    # find a session
open-context handoff <transcript>.jsonl \
    --store ./ctx --out project.ctx                      # package it
open-context inspect  project.ctx                        # read it
open-context validate project.ctx --archive ./ctx        # check it
open-context compile  project.ctx \
    --target anthropic --budget 4000 --task "Continue."  # compile for another agent
```

Add `--extract` to `handoff` to turn the conversation into typed goals,
constraints, and decisions. That is the one step needing a model, so it is
opt-in and reads `OPEN_CONTEXT_PROVIDER` / `_MODEL` / `_API_KEY`. On a real
session it extracts in windows and waits out rate limits, and reports what it
could not read rather than presenting a partial result as a whole one.

Two more commands: `open-context hook --store ~/.open-context` captures a
package automatically when an agent hits context pressure, and
`open-context verify-models` re-checks the free-model allowlist against the
providers' live catalogues.

Measured on a real 24.8 MB Claude Code session: one second, a 13.7 KB package —
**0.055% of the source**. Full walkthrough in
[docs/getting-started.md](docs/getting-started.md).

Without `--extract`, a package carries recent context and provenance — and says
so, rather than implying it holds goals it does not.

## Benchmarks

**What was measured.** Four ways of preparing context are compared under
identical conditions — same scenarios, model, continuation prompt, budget, and
repetition count. The dataset is adversarial by construction: conversations of
1,400–2,300 tokens compacted into a 160-token budget, roughly 9–14x, with
critical information planted early, in the middle, and late so a position effect
can be told apart from a content one.

**The baselines.** `full_context` receives the whole conversation and is a
*reference, not a competitor* — it is budget-independent, and it exists to show
whether the information was recoverable at all. `simple_summary_v1` is a single
summarisation call, the bar anything more elaborate has to clear. Against those
sit the Phase 5 baseline compactor and hybrid compaction.

Scores below are **deterministic string checks only**. Judged questions are
excluded because in this run the judge failed its own control: the arm holding
the entire conversation passed every deterministic check and only 9 of 24 judged
questions, which measures the judge rather than the context.

![Deterministic score per arm on the adversarial dataset](docs/assets/benchmarks/benchmark-retention.svg)

**What this shows, stated plainly:**

- Compaction costs roughly a quarter of what full context retains. Full
  context scores 1.00 and is a reference, not a competitor; the compaction
  arms land at 0.72 and 0.75.
- **The Phase 5 baseline does not beat a plain one-call summary.** 0.75 against
  0.72 is a gap of 0.03, and each arm varies by 0.17 across three runs of the
  same configuration — the difference is smaller than the noise.
- **Hybrid compaction has no score here.** Its extraction step returned nothing
  usable on 21 of its 24 cells, so the compactor disowned them and what remained
  was not hybrid. Phase 8's exit criterion is unmet, and the arm is drawn as a
  stub rather than a bar so it cannot be misread as a result.

![Score per scenario, showing all loss concentrated in two scenarios](docs/assets/benchmarks/benchmark-per-scenario.svg)

Splitting by scenario locates the loss precisely: every arm is perfect on four of
six, and everything that goes wrong happens in the two scenarios built around
exact values — a port number among similar ports, a count inside a tool result.
That is a specific, reproducible failure mode rather than a diffuse "retention
was 0.75", and it is what local retrieval was later built to address.

![Context size against tokens spent producing it](docs/assets/benchmarks/benchmark-compression.svg)

Compression alone would be a flattering number. Compaction shrinks the context
about 16x and **spends more tokens overall** than sending the conversation
whole, because the summarisation call is not free. At these conversation lengths
that trade does not pay; the regime where it does starts somewhere longer, and
nothing here has located it.

**What remains inconclusive.** Whether structured state beats a summary is
unresolved — the arm meant to answer it could not be evaluated on the only model
with enough daily free-tier allowance to run the full matrix. Whether compaction
pays for itself on genuinely long sessions is untested. Lexical retrieval
recovered the answer in 3 of 4 controlled scenarios and missed the fourth for a
structural reason ([docs/retrieval.md](docs/retrieval.md)).

Runs that went wrong are kept and explained beside the ones that did not — one
whole run was voided when a reasoning model spent its entire output budget
thinking and returned empty summaries, which read as the clearest result the
project had produced. Full methodology in
[docs/adversarial.md](docs/adversarial.md) and [docs/benchmark.md](docs/benchmark.md).

### Reproducing these charts

Every value is read from the persisted run; nothing is typed into the chart
script.

```bash
python benchmarks/plot.py     # reads benchmarks/results/, writes docs/assets/
```

![Test suite: 1,285 passed, 2 skipped, 0 failed](docs/assets/benchmarks/benchmark-test-suite.svg)

## Current Status

**This is an experimental public side project, not a product.** It is pre-1.0,
maintained in spare time, and there is no support commitment or patch SLA.

What works end to end today: reading a real Claude Code session, archiving it
losslessly, extracting typed state with a model, packaging it as a `.ctx`,
validating it, and compiling it into context for five provider families. 1,285
tests, `ruff` and `mypy` clean, verified from a clean install of the built wheel.

**What is not established:** the compaction thesis. On this project's own
adversarial benchmark the Phase 5 compactor does not beat a plain one-call
summary, and hybrid compaction (Phase 8) could not be measured at all — its
extraction step returned nothing usable on 21 of 24 cells, so its exit criterion
is unmet and the arm has no score. Those results are in the section above, in
full, because a benchmark you only publish when it agrees with you is not a
benchmark.

Import reads Claude Code transcripts and generic JSON Lines. Extraction is
opt-in and needs a model; everything else runs offline, with one runtime
dependency and no database server.

**Only free models are callable** by this repository's own tests and benchmarks,
from an allowlist checked before any request leaves the machine — the project
benchmarks itself on free tiers and refuses to make a paid request by accident.
That is a policy for *our* testing, not a limitation on you: point the runtime at
any provider you like. The allowlist goes stale, so `open-context verify-models`
re-checks it against the providers rather than asking you to trust a date.

## Known Limitations

Stated plainly, because you will find them anyway:

- **Continuation quality has not been shown to match full context.** Compaction
  costs roughly a quarter of what the uncompacted conversation retains on the
  adversarial dataset.
- **Structured state has not been shown to beat a plain summary.** The Phase 5
  baseline scores 0.75 against 0.72 — a gap smaller than either arm's spread
  across three runs of the same configuration.
- **Hybrid compaction is experimentally inconclusive.** Its extraction step
  returned nothing usable on 21 of 24 cells, so the arm has no score and its
  exit criterion is unmet.
- **Lexical retrieval cannot bridge a vocabulary gap.** It recovered the answer
  in 3 of 4 controlled scenarios and missed the fourth because the question and
  the answer shared no words. That is structural, not a tuning problem.
- **Agent interception is minimal.** One hook is implemented — Claude Code's
  `PreCompact`, verified against the shipping binary. Nothing else is automatic.
- **Provider coverage is early.** Two providers, five compile targets, one
  transcript importer besides generic JSON Lines.
- **Extraction depends on model quality.** An 8B model produced no usable state
  on most windows. What you get out varies with what you point it at.
- **The whole thing is experimental** and maintained in spare time.

## Help Us Break It

MemHandoff is experimental, and the most useful thing you can send is a case
where it fails. Stars are not the point; failure cases are.

We are especially interested in sessions where:

- important context is lost during compaction
- a decision is represented incorrectly, or its reason disappears
- a constraint — particularly a negative one — vanishes
- a failed approach is forgotten and the receiving agent tries it again
- temporal state ends up wrong, so a reversed decision reads as current
- an exact value is dropped or subtly altered
- something critical established at the very start is gone by the end
- a long stretch of irrelevant conversation crowds out what mattered
- a large tool output dominates the budget and starves everything else
- the receiving agent simply cannot continue the task
- handing context between two different agents produces incorrect behaviour

**Don't just tell us it doesn't work — give us a reproducible failure case.**
A small synthetic conversation that fails the same way is worth more than a
large private one, and is safe to share.

We built this, measured it honestly, and want people to break it. The benchmark
below is published with its negative and inconclusive results intact for exactly
that reason.

[Open a Context Loss Report](../../issues/new?template=context-loss.yml).

## Challenge the Benchmark

The benchmark is published with negative and inconclusive results on purpose,
which makes it the part of this project most worth attacking. New adversarial
scenarios are welcome — including ones that make every arm look worse.

The dataset already covers early, middle, and late placement of critical facts,
reversed decisions, negative constraints, exact numbers, similar entities, failed
approaches, long irrelevant sections, temporal state, and tool-heavy sessions.
Something it misses is exactly what we want.

A proposal should say: the scenario, expected behaviour, observed behaviour,
strategy, budget, model, why the case is difficult, and how to reproduce it.
[The template asks for each](../../issues/new?template=benchmark-challenge.yml).

**One rule:** a benchmark change must not be designed to improve this project's
score. A scenario only the newest strategy passes, added by whoever wrote that
strategy, is not evidence.

Every number above comes out of one file,
[`benchmarks/results/v4-adversarial-8b.jsonl`](benchmarks/results/v4-adversarial-8b.jsonl)
— 96 records, one per arm and scenario, each carrying the completion, the token
counts, and the per-check outcome. It is in the repository so the figures can be
recomputed without a credential: `python benchmarks/plot.py` redraws all four
charts from it, byte for byte. If a chart disagrees with the data, the data is
right and the chart is a bug worth reporting.

### Open Questions

These are the things we do not know. None is a claim of solved functionality.

- Can deterministic compaction preserve enough task-critical state to continue
  work, or is a model call unavoidable?
- How close can continuation quality get to full context, and at what budget?
- Does structured state actually improve on a plain summary — and if so, where?
- Can semantic retrieval recover the vocabulary gaps lexical matching cannot,
  and is it worth the dependency?
- How well does context transfer between genuinely different agents, rather than
  between two instances of one?
- How should a decision that was made, reversed, and remade be represented so a
  reader knows which one holds?
- What is actually necessary for a successful continuation? Nobody here knows,
  and the adversarial dataset is an attempt to find out.

## What this is not

Not a vector database, not an agent framework, not a RAG library, not a hosted memory service, not a chatbot, not a replacement for any model provider. It is the compaction and interoperability layer. Where another project does a job well, this one should call it rather than reimplement it.

## Design constraints

- Local-first. No account, no hosted database, no mandatory network call for storage, search, inspection, or export. `Open Context + Ollama` must work with nothing else installed.
- Lightweight. SQLite and the filesystem. No Docker, no server, no second database. A large archive must not mean proportional RAM.
- Provider-neutral packages. Import from one agent, continue in another.
- Source-recoverable compaction. The original history is never destroyed, and important compacted information points back at its source through provenance.
- Inspectable. Ask why something was kept, get message IDs back.

## Requirements

Python 3.12 or newer. One runtime dependency: pydantic.

## Reproduce Everything

Nothing here needs a credential. The suite is offline and the charts are drawn
from a persisted run already in the repository.

```bash
git clone <repository-url>
cd open-context-runtime
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The four checks CI runs, verbatim:

```bash
pytest
ruff check .
ruff format --check .
mypy
```

Redraw the benchmark charts from the stored results — every value is read from
the run, nothing is typed into the script:

```bash
python benchmarks/plot.py
```

Then try it on one of your own sessions:

```bash
open-context sessions                                   # find a transcript
open-context handoff <transcript>.jsonl \
    --store ./ctx --out project.ctx                     # archive + package it
open-context inspect  project.ctx                       # read what it kept
open-context validate project.ctx --archive ./ctx       # check it
open-context compile  project.ctx \
    --target anthropic --budget 4000 --task "Continue."  # context for another agent
```

Two integration tests skip unless a provider credential is present, and no test
makes a network call without one.

### The contribution path

1. Install, and run the four checks above.
2. Redraw the charts, so you have seen the numbers come out of the data.
3. Hand off one of your own sessions and read the package.
4. Find a case where it loses something that mattered.
5. Open a [Context Loss Report](../../issues/new?template=context-loss.yml)
   with a reproducible example — synthetic is fine and often better.
6. Or propose a [benchmark scenario](../../issues/new?template=benchmark-challenge.yml)
   it cannot catch.
7. Or send an [architecture proposal](../../issues/new?template=architecture-proposal.yml)
   with the experiment that would test it.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers development setup, where compaction
strategies live and how they are evaluated, the benchmark methodology, and the
free-models-only policy for this repository's own testing.

New compaction strategies are evaluated against the existing baselines on the
existing scenarios; benchmark methodology is not changed to accommodate them.

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) applies to everyone taking part.
Architecture questions and research ideas are welcome as issues — there is no
separate forum, and an issue that turns out to be a conversation is fine.

## Reusing other projects

This project is Apache-2.0 and reuses compatible open-source code where it makes sense. It does that at the point the relevant component is being built, not up front, so the repository does not become a pile of borrowed subsystems that never fit together. The rule and the review checklist are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Security

[SECURITY.md](SECURITY.md) covers how to report a vulnerability, and the threat
model: a `.ctx` package is untrusted input by design, nothing in this project
executes anything because it appeared in imported context, and archive locations
inside a package are never resolved.

## License

Apache-2.0. See [LICENSE](LICENSE).
