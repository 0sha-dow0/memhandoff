# Product architecture

> This document defines what the product is. [architecture.md](architecture.md) describes how the engine is built, and where the two disagree, this one wins.

## What this is

A lightweight, local-first context-compaction plugin that turns a long-running AI session into a portable representation another agent can pick up and continue.

## The problem

You spend two days working with one agent. The conversation outgrows the window. The agent compacts it using its own provider-specific mechanism, and that compacted state is locked inside that provider. Switching to a different agent means explaining the entire project again from scratch.

```
Claude Code
    |
long-running conversation
    |
Open Context
    |
portable context package
    |
Codex, Gemini, another Claude, a local model
    |
continue the same work
```

The user should not have to re-explain the project.

## The artifact

The fundamental artifact is a **portable context package**, not a memory database. The database is an implementation detail of the local engine.

A package captures enough to continue the work:

goals, constraints, decisions and their rationale, rejected decisions and why, current state, work completed and work outstanding, open tasks, open questions, important entities, relevant artifacts, previous attempts and failed approaches, relevant recent context, provenance, temporal relationships, and references to source material still sitting in the local archive.

It is structured and loss-aware. It is not an LLM-generated summary. A summary asserts; a package carries typed records with sources, statuses, and a record of what was dropped.

Compaction is source-recoverable rather than reversible. The original history is never destroyed and important compacted information points back to its source through provenance, but the compacted text cannot regenerate the conversation it came from. Recovery means returning to the stored original.

## Three layers, kept distinct

| Layer | What it is | Scale | Portable |
| --- | --- | --- | --- |
| **A. Raw conversation** | Complete original history, the source material | 500k tokens | No, stays local |
| **B. Portable context package** | Compacted structured representation | 5k to 20k tokens | Yes, this is the artifact |
| **C. Agent-specific context** | What the context compiler produces from B for one receiving model | 5k to 20k tokens plus the user's request | No, generated per agent |

```
RAW HISTORY
    |
COMPACTION ENGINE
    |
PACKAGE BUILDER
    |
PORTABLE CONTEXT PACKAGE      <- provider-neutral, this crosses machines
    |
CONTEXT COMPILER              <- provider-specific starts here
    |
AGENT ADAPTER                 <- transport only
    |
RECEIVING AGENT
```

Layer B is never provider-specific. The moment a Claude-shaped prompt appears in the package format, the same package stops working for Gemini and the product's reason to exist is gone.

## Archive versus package

The local archive holds everything: every message, every tool result, every artifact, the full history. The package holds the minimum useful representation needed to continue.

```
LOCAL MACHINE                     TRANSFERRED              RECEIVING AGENT
full archive, 500k tokens   ->    package, 8k to 15k   ->  package + user request
```

A package may reference archived material that stays on the original machine. Whether those references are usable on a different machine is unresolved; see the open questions.

## Local-first

No cloud dependency is required for storage, search, inspection, compaction bookkeeping, or export. The archive and the compacted state live on the user's machine.

A cloud LLM may receive the compiled context when the user chooses a cloud model. That is the user's choice at compile time, not an architectural requirement. `Open Context + Ollama` must work with nothing else installed.

That promise is now structural rather than aspirational. Every model call goes through the `LLMProvider` and `Tokenizer` interfaces, providers register themselves by name, and no vendor SDK is a dependency of the base package — so a user who only wants local models installs nothing belonging to a cloud provider. See [llm.md](llm.md).

Not designed around: AWS, S3, cloud databases, hosted vector databases, SaaS infrastructure, remote memory services.

## Lightweight

The default install must not require Docker, PostgreSQL, Redis, Kafka, Elasticsearch, Neo4j, Qdrant, Milvus, or any cloud infrastructure. SQLite and the local filesystem are the persistence layer. Anything else needs a profile showing a concrete need first.

A large history must not mean proportional RAM. History stays on disk and only the relevant portion is loaded. This is a real constraint on the code, not an aspiration: see the conflicts section.

```
small runtime + local archive + local state and index + agent adapters
```

## Plugin architecture

The core is headless. Adapters are thin.

```
                  OPEN CONTEXT CORE
                         |
      +------------------+------------------+
      |                  |                  |
   MCP / API         CLI / API           Library
      |                  |                  |
      +------------------+------------------+
                         |
                  core compaction
                         |
      +------------------+------------------+
      v                  v                  v
 Claude Code          Codex            other agents
```

The core owns ingestion, evidence extraction, state construction, compaction, retention, retrieval, package generation, validation, and reconstruction.

Three components sit at the boundary and are easy to confuse:

| Component | Input | Output | Provider-specific |
| --- | --- | --- | --- |
| **Package builder** | State graph, evidence, budget | `project.ctx` | No |
| **Context compiler** | `project.ctx`, target model | That agent's context | Yes, the only such place |
| **Agent adapter** | That agent's context | Delivered to the agent | Transport only |

```
state -> PACKAGE BUILDER -> project.ctx -> CONTEXT COMPILER -> agent context -> ADAPTER -> agent
                            provider-neutral            provider-specific
```

The package builder never knows which model will read its output; that is what makes one package serve every agent. The context compiler is where Claude, OpenAI, Gemini, and local prompt conventions live, and nowhere else.

An adapter translates between one agent's interface and the core. An adapter that starts making its own compaction decisions is a bug, because then two agents produce different packages from the same history and the format guarantees nothing.

## Automatic operation, eventually

The intended experience is that the user never manages memory by hand:

```
agent -> adapter -> conversation events -> context pressure detected
      -> compaction -> local archive and updated state -> session continues
```

Signals the runtime should eventually act on: context approaching the model limit, conversation size, an explicit compaction request from the agent, an explicit export from the user, a session handoff, an agent switch.

**Not being built yet, deliberately.** Every one of those signals requires an integration point that we have not confirmed exists. Which hooks Claude Code, Codex, and MCP clients actually expose determines whether interception is even possible, and designing the mechanism before knowing that means designing it twice. Integration research comes before implementation.

## Portable package format requirements

The format is not being designed yet. These are the requirements it will have to meet.

Contents, roughly: manifest, state, decisions, constraints, tasks, entities, artifacts, evidence, recent context, provenance, metadata.

Properties:

- **Provider-neutral.** No provider's prompt conventions, message shapes, or token counts.
- **Versioned.** A reader must be able to tell what it is reading and refuse a version it does not understand.
- **Deterministic where possible.** The same state should produce the same bytes, so packages can be diffed and reviewed.
- **Human-inspectable.** A user should be able to open it and see why something was kept.
- **Machine-readable.** Parseable without an LLM.
- **Extensible.** New fields must not break old readers.
- **Provenance-preserving.** Every important claim traces to a source.
- **Archive-referencing.** Able to point at material left behind on the origin machine.

## Evaluation must measure continuation

The question is not whether a summary reads well. It is whether **another agent can continue the work nearly as well as if it had the original conversation.**

Four arms on the same downstream task:

| Arm | What it is |
| --- | --- |
| A | Full original context |
| B | Ordinary LLM summary |
| C | Retrieval only |
| D | Our portable package |

Metrics: critical fact retention, constraint retention, decision retention, decision rationale retention, temporal state accuracy, open-task retention, provenance accuracy, hallucination rate, task completion, context token count, compression ratio, latency, and LLM cost where applicable.

The corpus must include the cases that break naive summarization: important information stated early and never repeated, contradictory decisions, decisions later reversed, negative constraints ("do not use X"), exact numbers, similar-looking entities, failed approaches, long irrelevant stretches, and facts that change over time.

Arm B now exists in the codebase. Phase 5 built a baseline compactor — historical summary plus verbatim recent window, held to a measured token budget — precisely so there is something concrete to measure arm D against. It is a control, not a contribution, and it produces an internal experimental representation rather than a portable package. See [baseline-compaction.md](baseline-compaction.md).

Phase 5.5 built the harness that runs the comparison: arms A and B exist, arm C is a seam, arm D is unwritten. The same task, model, prompt, and budget are held identical across arms, and the harness refuses to truncate the full-context condition to make it fit — it reports the reference condition unavailable instead. See [evaluation.md](evaluation.md).

**No arm has been measured.** The apparatus exists; nothing in this repository states how any strategy performs, and nothing should until a real model has been run against the dataset.

## What this is not

Not a general chatbot. Not a cloud memory service. Not a hosted database. Not an agent framework. Not a vector database product. Not a general RAG platform. Not a replacement for any model provider. Not a UI-heavy application. Not an IDE. Not a proprietary agent.

It is the context-compaction and interoperability layer, and nothing else.

## Unresolved

Recorded rather than answered, because guessing here would be expensive to undo.

**1. Cross-machine provenance.** A package's provenance points at messages in the origin machine's archive. On the receiving machine those records do not exist. The storage layer enforces referential integrity, so importing a package with such references fails as written. Options: strip provenance on import, keep it as opaque non-resolving strings, inline the cited evidence text, or make an imported package a read-only foreign archive. This is the largest unresolved question and it blocks the package format.

**2. What "continue the work" means for identity.** Is an imported package a new session, a continuation of the original session, or a distinct kind of record? This decides whether session-locality still holds after import.

**3. Package size targets are asserted, not measured.** The 8k to 15k figure comes from intuition. Nothing yet shows what budget preserves task performance, which is exactly what the Phase 5.6 benchmark is for.

**4. How much recent raw context to inline.** Recent turns are usually the most useful and the most expensive. The split between structured state and verbatim recent history is unknown.

**5. Which integration points actually exist.** Whether Claude Code, Codex, and MCP clients expose usable hooks for observing context pressure is unverified. Automatic operation depends entirely on the answer.

**6. Whether the package should carry a loss report.** A loss-aware format could state what was dropped and why. That is useful for trust and adds size. Undecided.

**7. Multi-session projects.** Work often spans several sessions. Whether a package covers one session or a project spanning many is undecided, and the current data model is session-scoped.
