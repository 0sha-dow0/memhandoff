# Architecture

> How the engine is built. [product-architecture.md](product-architecture.md) defines what the product is; where the two disagree, that document wins. Most of what follows is not implemented yet. Where an implementation contradicts this document, the implementation is right and this document is stale.

## Where the engine sits

The engine produces one thing: a portable context package that another agent can continue from.

```
RAW HISTORY
    |
COMPACTION ENGINE          <- everything in this document
    |
PORTABLE CONTEXT PACKAGE   <- the product artifact
    |
AGENT ADAPTER
    |
RECEIVING AGENT
```

The archive, the state graph, SQLite, and the retrieval index are components of the compaction engine. They are not the product. A user never has to care that SQLite is involved.

## The layer model

The context window is a working set, not memory:

```
LLM
 |
Active context      what the model sees right now
 |
Working state       what the session is currently about
 |
Archive             everything that was ever said, on disk
```

Information moves between layers and is not destroyed on the way down.

Cross-session durable memory sat in this stack in earlier drafts. It has been removed. The product is session handoff, and a general memory layer is a different product that would pull the runtime toward being a platform. Session state stays session-local.

## Data flow

```
RAW HISTORY
     |
     v
  ARCHIVE  <---------------------------+
     |                                 |
     +-------------+                   |
     |             |                   |
     v             v                   |
STRUCTURED     RETRIEVAL               |
   STATE           |                   |
     |             |                   |
     +------+------+                   |
            |                          |
            v                          |
        COMPACTION  <-- task, budget    |
            |                          |
            v                          |
  PORTABLE CONTEXT PACKAGE --- provenance +
            |
            v
      AGENT ADAPTER
            |
            v
           LLM
```

The step that used to be labelled "context compiler producing active context" is now two: the package builder produces a provider-neutral package, and the context compiler turns that package into one agent's representation. Keeping them separate is what makes the same package usable by more than one agent.

## Invariants

These hold across every phase. A change that breaks one is an architecture change and needs discussion, not a pull request.

**The archive is the source of truth.** Compaction, pruning, and paging never delete source history. Deletion is an explicit user operation and is logged.

**Compaction is source-recoverable.** The original history is never destroyed, and important compacted information points back to its source through provenance. This is a claim about the archive and the provenance links, not about the compacted text: a semantic summary cannot regenerate the conversation it came from, and nothing here should be read as suggesting it can. Recovery means going back to the stored original, not inverting the compaction.

**Stored state and model context are different things.** The state graph is a durable structured store. The context a model receives is generated output. There is no stored artifact called "the summary."

**The portable package is provider-neutral.** No provider's prompt conventions, message shapes, or token counts appear in it. A package that only one agent can read defeats the point of having one.

**History stays on disk.** A large archive must not mean proportional memory use. Only the relevant portion is loaded.

**Context is task-dependent.** The same history compiles to different contexts for different tasks and different budgets. That is the whole thesis.

**State is model-independent.** Nothing in storage or the state graph refers to a specific provider's message format, token count, or model name. Providers are adapters at the edge.

**Every important state item has provenance.** A decision without source message IDs is not trustworthy and cannot be audited when it turns out to be wrong.

**The session state graph is session-local.** Every reference inside a session, message parents, provenance, supersession, relations, and snapshot membership, resolves within that session. Knowledge that should move between sessions moves as a portable context package, which is an explicit, inspectable artifact, not as a reference reaching sideways out of one session's history into another's.

**Provenance points backwards only.** A provenance reference must resolve to a record that is already stored, so the graph is acyclic by construction and the layering `raw source -> evidence -> state` cannot be inverted.

**Local by default.** Storage, search, snapshots, inspection, and export work with no network. Only LLM-backed operations need a provider, and a local model counts.

**The runtime stays small.** No background server, no second database, no service to operate. It runs as one local process.

## Components

Nine pieces, built in dependency order.

| Component | Question it answers |
| --- | --- |
| Session store | What was actually said? |
| Archive | Can I still get the original? |
| Context graph | What does the project know? |
| Retention policy | How valuable is this piece? |
| Compaction engine | How do I shrink this without losing what matters? |
| Package builder | What belongs in the portable package for this task and budget? |
| Validator | Did compaction destroy something important? |
| Context compiler | What does this package look like to the receiving agent? |
| Agent adapter | How does that context reach the agent? |

The last three run on opposite sides of the machine boundary and must not be conflated.

**Package builder** produces the portable package. It selects state, evidence, and recent context against a budget and writes the provider-neutral artifact. It knows nothing about any model.

**Context compiler** consumes that package and produces the representation one receiving agent gets: Claude-shaped, OpenAI-shaped, Gemini-shaped, or a local model's prompt format. It is the only place where a provider's conventions appear.

**Agent adapter** is transport. It translates between an agent's interface, whether MCP, CLI, or library, and the runtime. It performs neither compaction nor compilation.

```
state -> PACKAGE BUILDER -> project.ctx -> CONTEXT COMPILER -> agent context -> ADAPTER -> agent
                            provider-neutral            provider-specific
```

The package builder and the package format are what the project lives or dies on. Everything before them is preparation; the compiler and adapters after them are comparatively thin, and one package must serve all of them.

## The four compaction operations

| Operation | Active context | Archive |
| --- | --- | --- |
| Keep | Original stays | Original stays |
| Prune | Removed | Retained |
| Summarize | Replaced by a semantic representation | Original retained |
| Extract | Replaced by structured state items | Original retained |

What this never does:

```
100k tokens -> 5k summary -> delete the 100k
```

What it does instead:

```
100k tokens -+-> archive (all of it)
             +-> structured state
             +-> compact narrative
             +-> active context
```

## Retention classes

| Class | Examples | Treatment |
| --- | --- | --- |
| Critical | Explicit constraints, API contracts, exact numbers, security requirements | Exact source preserved, plus structured form, plus provenance |
| Important | Architecture decisions, rationale, rejected approaches, current state | Structured form, compressed explanation, provenance |
| Reconstructable | Long discussions, repeated explanations, intermediate reasoning | Semantic compression, archived |
| Ephemeral | Repetitive tool output, duplicate logs, greetings | Pruned to archive |

## Temporal state

Facts change. Overwriting them loses the history and the ability to answer why.

```
January: PostgreSQL
March:   MongoDB
June:    PostgreSQL
```

All three are stored. Current state is derived from ordering, not from the last write. "What database are we using?" answers PostgreSQL, and can add that MongoDB was used briefly in March, because that is often the thing the person actually wanted to know.

## Storage

Two stores. The archive holds raw conversation as append-only JSONL on the filesystem; SQLite holds indexed metadata and interpreted structure. The conversation is not duplicated into SQLite. See [archive.md](archive.md) and [storage.md](storage.md).

Not PostgreSQL, not Redis, not a vector database. Retrieval starts with FTS5 and adds a local vector index only when a benchmark shows FTS5 is the bottleneck. Introducing a vector database before that measurement means carrying an operational dependency to solve a problem nobody has demonstrated.
