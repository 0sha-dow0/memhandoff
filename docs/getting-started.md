# Getting started

Take a long session from one coding agent and continue it in another.

## Install

```bash
pip install open-context-runtime
```

Python 3.12 or newer. The only dependency is pydantic. No database server, no
account, no network call for anything except extraction — which is opt-in.

## The five-minute version

```bash
# 1. find a session
open-context sessions

# 2. turn one into a portable package
open-context handoff ~/.claude/projects/<project>/<uuid>.jsonl \
    --store ./ctx --out project.ctx

# 3. read what you got
open-context inspect project.ctx

# 4. check it
open-context validate project.ctx --archive ./ctx

# 5. compile it for whoever continues the work
open-context compile project.ctx \
    --target anthropic --budget 4000 --task "Finish the export writer."
```

Step 5 prints a request body in that provider's shape. Paste the `system` text
into another agent, or send the JSON yourself — this tool never makes the call.

## What you actually get

`handoff` tells you, every time:

```
session ses_912bcf79732042d7b702749c
  read      192 events
  kept      187 (5 abandoned as rewound)
  state     0 items
  warning: no state was extracted, so this package carries recent context and
           provenance but no goals, constraints, or decisions; pass a provider
           to extract it
```

Read that warning. **Without `--extract`, a package carries recent conversation
and provenance and nothing else.** That is useful — it is a compact, verifiable
record of where the work stood — but it is not the goals-and-constraints summary
the project is ultimately for.

## Getting real state out of it

Extraction is the one step that needs a model, so it is opt-in and needs a
provider configured:

```bash
export OPEN_CONTEXT_PROVIDER=groq          # or openrouter
export OPEN_CONTEXT_MODEL=llama-3.3-70b-versatile
export OPEN_CONTEXT_API_KEY=...

open-context handoff session.jsonl --store ./ctx --out project.ctx --extract
```

Now the package carries typed goals, constraints, decisions with their
rationale, tasks, and open questions, each pointing back at the messages it came
from.

Only models on an explicit free-model allowlist are callable. That is deliberate:
this project runs its own benchmarks on free tiers and refuses to make a paid
request by accident. See [llm.md](llm.md).

## What the pieces are

| | |
| --- | --- |
| **archive** | the original conversation, append-only, hashed per record. Never destroyed. |
| **`.ctx` package** | one JSON file: state, evidence, recent turns, a pointer to the archive |
| **compiled context** | what a specific model receives, budgeted and in its own request shape |

The archive stays on your machine. The package is the portable part. The
compiled context is generated per agent and never stored.

## Honest limits

- **Import reads Claude Code transcripts and generic JSON Lines.** Other agents
  need an importer; the protocol is `detect` plus `read`.
- **Compaction has not been shown to beat a plain summary.** The project's own
  adversarial benchmark says so — see [adversarial.md](adversarial.md). What it
  did establish is that compaction costs 25–28% of what full context retains at
  9–14× compression, and that what it loses is exact values.
- **A `.ctx` from someone else is untrusted input.** `validate` reports what a
  package carries; nothing in this tool executes anything from one. See
  [hardening.md](hardening.md).
- **Nothing sends your conversation anywhere** unless you pass `--extract`, and
  then only the archived session goes to the model you configured.

## Where to go next

[The docs index](README.md) maps the rest.
