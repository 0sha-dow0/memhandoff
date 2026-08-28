# Getting started

MemHandoff moves useful working context from Agent A to Agent B. The quickest
way to see the whole path is the bundled offline example; after that, replace
its transcript with one of your own.

## Requirements

- Python 3.12 or newer
- macOS, Linux, or Windows
- no account, server, API key, or model call for the basic handoff

The runtime has one direct dependency, pydantic.

## Complete a handoff in under five minutes

The current public release is `v0.1.0`. To try the current `main` branch and its
bundled example, install from source.

On macOS or Linux:

```bash
git clone https://github.com/0sha-dow0/memhandoff.git
cd memhandoff
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
./examples/cross-agent-handoff/run.sh
```

On Windows PowerShell, install the same way and run the three commands directly:

```powershell
git clone https://github.com/0sha-dow0/memhandoff.git
cd memhandoff
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install .

New-Item -ItemType Directory -Force memhandoff-demo\archive
open-context handoff examples\cross-agent-handoff\agent-a.jsonl --store memhandoff-demo\archive --out memhandoff-demo\export.ctx --title "Customer export writer"
open-context validate memhandoff-demo\export.ctx --archive memhandoff-demo\archive
open-context compile memhandoff-demo\export.ctx --target generic --budget 800 --task "Continue the export work." | Out-File -Encoding utf8 memhandoff-demo\agent-b-context.json
```

Open `memhandoff-demo/agent-b-context.json`. Its `text` value is the context for
Agent B. Paste it into a different agent and ask:

> What should you do next, and what must you not change?

The receiving agent should recover all seven facts listed in the
[example walkthrough](../examples/cross-agent-handoff/README.md). If it does not,
open a [Context Loss Report](https://github.com/0sha-dow0/memhandoff/issues/new?template=context-loss.yml).

This example is deliberately small enough to fit in the recent-context window.
It proves the packaging and cross-agent workflow, not long-session compaction
quality.

## Install the published release

```bash
pip install open-context-runtime
```

Package versions and source-branch features can differ between releases. Check
what you installed with:

```bash
python -c "import open_context; print(open_context.__version__)"
open-context --help
```

## Hand off your own session

Claude Code sessions are discovered automatically:

```bash
open-context sessions
```

Use the returned path, or any generic JSON Lines transcript:

```bash
open-context handoff ~/.claude/projects/<project>/<uuid>.jsonl \
  --store ./ctx --out project.ctx --title "My work"

open-context inspect project.ctx
open-context validate project.ctx --archive ./ctx
open-context compile project.ctx \
  --target generic --budget 4000 --task "Continue this work." \
  > agent-b-context.json
```

Paste the `text` value from `agent-b-context.json` into Agent B. For a provider
request body instead, choose `anthropic`, `openai`, `gemini`, or `local` as the
target. MemHandoff shapes the payload but never sends it.

A generic JSON Lines transcript has one object per line. The importer recognizes
common field names including `id`, `parent_id`, `role`, `content`, and
`timestamp`:

```json
{"id":"1","role":"user","content":"Keep port 9443 reserved."}
{"id":"2","parent_id":"1","role":"assistant","content":"Understood."}
```

## Read the handoff report

Without model extraction, a typical report says:

```text
session ses_912bcf79732042d7b702749c
  read      192 events
  kept      187 (5 abandoned as rewound)
  state     0 items
  warning: no state was extracted, so this package carries recent context and
           provenance but no goals, constraints, or decisions
```

That warning is a contract. A recent-only package is useful for a short transfer
but does not pretend to have inferred older goals or decisions. Up to 12 spoken
turns are carried verbatim; tool traffic is excluded from that window.

The three artifacts have different jobs:

| Artifact | What it is |
| --- | --- |
| Archive | Original records, append-only and hashed; stays local |
| `.ctx` | Portable state, evidence, recent turns, and an archive reference |
| Compiled context | Budgeted input shaped for one receiving agent family |

## Optional model extraction

Extraction asks a configured model to infer typed goals, constraints, decisions,
rationale, tasks, and open questions from the archived conversation:

```bash
export OPEN_CONTEXT_PROVIDER=groq       # or openrouter
export OPEN_CONTEXT_MODEL=<model>
export OPEN_CONTEXT_API_KEY=<key>

open-context handoff session.jsonl \
  --store ./ctx --out project.ctx --extract
```

This is the only handoff step that sends conversation text over the network. It
is opt-in, reports failed windows, and does not hide a partial extraction.

The repository's own tests and benchmarks may call only explicitly allowlisted
free models and never fall back to paid usage. Users can configure any compatible
model. See [llm.md](llm.md).

## Honest limits

- Recent-only handoff loses older facts once a conversation exceeds its window.
- Extraction quality varies with the configured model; one measured 8B model
  produced no usable state for most benchmark cells.
- The published compactor does not clearly beat a simple summary, and hybrid
  compaction remains inconclusive.
- Lexical retrieval helps with exact evidence but fails across vocabulary gaps.
- Only Claude Code and generic JSON Lines importers exist today.
- A `.ctx` is untrusted input. Validation does not make its text safe or correct.

Read [hardening.md](hardening.md) before accepting packages from other people and
[benchmark.md](benchmark.md) before interpreting quality claims.

## If context is lost

Do not share a private transcript just to report a bug. Remove credentials,
private conversations, proprietary code, customer data, and personal information.
Prefer a small synthetic transcript that fails in the same way.

The [Context Loss Report](https://github.com/0sha-dow0/memhandoff/issues/new?template=context-loss.yml)
asks for source and receiving agents, budget, expected context, actual context,
and reproduction commands. Those failures are the feedback this project most
needs.

The [documentation index](README.md) maps the formats, architecture, compiler,
retrieval, security model, experiments, and contribution guide.
