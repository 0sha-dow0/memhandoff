# Cross-agent handoff

Phase 11. Take a real session from one coding agent and continue it in another.

```
Agent A  ->  runtime  ->  project.ctx  ->  Agent B  ->  continue the work
```

## Do it

```bash
# 1. find a session on this machine
python -m open_context sessions

# 2. hand it off
python -m open_context handoff \
    ~/.claude/projects/<project>/<session-uuid>.jsonl \
    --store ./ctx --out project.ctx --title "Export writer"

# 3. check what you got
python -m open_context validate project.ctx --archive ./ctx
python -m open_context inspect  project.ctx

# 4. compile it for whoever continues the work
python -m open_context compile project.ctx \
    --target anthropic --budget 4000 --task "Finish the export writer."
```

Step 4 prints a request body for the target family. Paste the `system` text into
another agent, or send the JSON yourself — the runtime does not make the call.

`handoff` reports what it did:

```
session ses_912bcf79732042d7b702749c
  read      192 events
  kept      187 (5 abandoned as rewound)
  state     0 items
  warning: no state was extracted, so this package carries recent context and
           provenance but no goals, constraints, or decisions
```

## What the format actually looks like

The roadmap says: *do not assume integration APIs; inspect the current, actual
mechanisms*. Everything below was measured on **6 real transcripts, 8,890
conversation records**, in `~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl`.
Three findings each contradicted the obvious assumption.

### A session file is a graph, not a list

Records carry `uuid` and `parentUuid`. In **5 of 6** files some parent had more
than one child — up to three. Those forks are rewinds and retries, and the
abandoned attempts stay on disk. Reading the file top to bottom yields a
conversation containing work that was taken back, presented as though it happened.

### The chain runs through records you do not import

Between two turns sit attachments and system records — **804 attachments and 62
system records** in one session alone. `parentUuid` points at whichever record
came last, usually an attachment. Following it while indexing only conversation
records breaks the chain immediately.

Each event therefore carries `conversation_parent` in its metadata: the nearest
ancestor that is actually a turn. `parent_id` keeps the provider's own reference
verbatim, because a derived value in a field documented as verbatim is a quiet
lie.

### A second root is a continuation, not a rival

**This is the one that mattered.** When Claude Code compacts a session it starts
a *new tree* rather than extending the old one. Measured on two long transcripts,
the trees are strictly sequential and split exactly at the `isCompactSummary`
record: 2,407 records then 388, and 3,294 then 927.

An earlier version of `active_thread` kept only the chain ending at the last
record — the obvious reading of a DAG. It discarded **78% and 86%** of those two
sessions: every pre-compaction turn, which is precisely the history a handoff
exists to carry. One of the two files contains no forks at all, so nothing about
it was ambiguous. Only real data showed the error.

So: **forks are resolved, roots are kept.** `segments()` exposes the split.

### Content is blocks, except when it is a string

`message.content` is a list of typed blocks — `text`, `thinking`, `tool_use`,
`tool_result` — for most records, and a plain string for **225 of 3,323** user
records. Both are handled.

One record can become several events. A turn holding text and three tool calls is
four things that happened, and collapsing them loses the tool calls — which Phase
8.5 measured to be where answers hide.

**Thinking is skipped by default.** It is the model's deliberation, not its
answer, and this project has already stored one as the other: `strip_inline_reasoning`
exists because a compactor saved a truncated `<think>` block as a summary.
`include_thinking=True` carries it, and a turn that was *only* thinking still
produces an empty event so the chain keeps a link where a turn occurred.

## What the adapter does not do

Per the roadmap, an adapter receives context, invokes the runtime, and exports or
compiles. It does not decide what to keep.

`handoff` reads, archives, and packages. Selection lives in the compactor,
budgeting in the compiler, and the format in the package. A handoff with its own
idea of what matters would be a fourth compactor nobody asked for.

**The archive is written before anything is interpreted.** Extraction needs a
model and a model can fail; when it does, the conversation is already on disk
losslessly and only the expensive part has to be retried.

**Malformed records are skipped rather than fatal here**, unlike an ordinary
import, which aborts. A live session file can have a torn tail simply because the
agent writing it is still running, and refusing the whole conversation over its
last line would break handoff exactly when it is most wanted. The records are
counted and located.

## State

`handoff` takes state as an argument and does not extract it. Extraction needs a
model, a budget, and a choice of which — none of which belongs to a function whose
job is moving a conversation from one place to another.

Called without it, the package carries recent context and provenance, no goals or
constraints, and **says so in a warning**. That is a smaller claim honestly made
rather than a larger one faked. To fill it, run the extractor over the archived
session and pass the result in.

## Other agents

Only Claude Code is implemented, because it is the one whose real format could be
inspected on this machine. The importer protocol is the extension point:
`detect` plus `read`, registered in `importers/registry.py`.

**Order in that registry is load-bearing.** `JsonLinesImporter.detect` accepts any
line that parses as a JSON object, which every Claude Code transcript also does.
Generic-first would claim every session file and import it as anonymous records —
losing tool calls, the parent chain, and the branch structure, while appearing to
succeed. Specific formats come first; generic is the fallback it was always meant
to be.
