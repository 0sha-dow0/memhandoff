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
| 4 | Conversation import adapters | Not started |
| 4.5 | LLM provider and tokenizer abstraction | Not started |
| 5 | Baseline compaction engine | Not started |
| 5.5 | Evaluation framework | Not started |
| 5.6 | First benchmark against summarization and retrieval baselines | Not started |
| 6 | Structured extraction | Not started |
| 7 | Retention and state management | Not started |
| 8 | Hybrid compaction | Not started |
| 9 | Portable context package format and package builder | Not started |
| 10 | Context compiler, package to agent-specific context | Not started |
| 11+ | Agent integrations and adapters | Not started |

Candidate integrations for Phase 11 and beyond: Claude Code, OpenAI and Codex, Gemini, local Ollama-based agents, MCP-compatible clients, other coding agents.

Automatic context-pressure detection comes only after we know which hooks each agent actually exposes. Designing interception before that means designing it twice.

## What changed in this roadmap

The product realignment moved the target from a general context runtime to a portable context package for agent handoff. See [product-architecture.md](product-architecture.md).

Three things moved as a result. Evaluation gained its own phases at 5.5 and 5.6 and now runs immediately after the first compactor, so the thesis gets measured against a plain summary before eight more layers are built on top of it. The provider and tokenizer abstraction moved from Phase 16 to 4.5, where the code that needs it actually starts. The package format and package builder became Phase 9, ahead of the context compiler at Phase 10, because the compiler consumes the package and its input shape is defined by the format rather than the other way round.

Dropped from the roadmap: memory backends, paging, and the separate MCP phase as previously scoped. Cross-session memory is a different product. Paging is an optimization with no measured need yet. MCP returns as one adapter among several in Phase 11, not as a milestone of its own.

## What Phase 3 covers

The local immutable archive: append-only JSONL on the filesystem, one log per session, with a rebuildable offset index, per-record hashing, bounded tail recovery, and streaming reads. See [archive.md](archive.md).

The archive holds raw conversation; SQLite holds indexed metadata and interpreted structure. The conversation is not duplicated into SQLite.

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

**When does the provider abstraction land?** Phase 4.5, before anything needs an LLM call or a token count.

**Phase 2 question: are cross-session references allowed?** No, resolved in Phase 2.1. The session state graph is session-local and `CrossSessionReferenceError` is raised otherwise. Knowledge crosses sessions as a portable context package instead.

**Phase 1 question: is `Message.seq` unique per session?** Yes. `UNIQUE (session_id, seq)` in the schema, with `SequenceConflictError` raised on collision. An importer meeting a source export with genuine duplicates has to renumber, which lands in Phase 4.

## Open questions

Product-level questions live in [product-architecture.md](product-architecture.md). These are engine-level.

These affect layers beyond the one being built. They are recorded rather than answered so the decision is deliberate and dated.

**Validation is a nondeterministic gate.** The compaction safety check rejects a snapshot when validation fails, and the validator is LLM-based. That makes snapshot acceptance nondeterministic at the margin and unusable as a regression signal across commits. Proposal: the hard gate is deterministic (required state item IDs present, provenance references resolve, token budget satisfied), and the LLM retention score is reported but does not block.

**Automatic compaction has no control point.** Proactive compaction assumes the runtime can observe context usage and intervene mid-session. MCP is pull-based: the agent decides when to call. This now blocks the Phase 11 adapters rather than a specific phase, and is why integration research precedes implementation.

**Short name collision.** The proposed CLI name `ocr` collides with optical character recognition in search results, package indexes, and documentation. No entry point is defined yet, so the decision is still free. Candidates: `octx`, `opencontext`, `ctxrt`.

**Cross-machine provenance breaks referential integrity.** Storage requires every provenance reference to resolve to a stored record. A package imported on another machine cites messages that do not exist there, so import fails as written. This blocks Phase 9 and is tracked as the first open question in product-architecture.md.

**Reads load whole result sets in SQLite.** `list_messages` materializes every message in a session. Phase 3 solved this for the archive, where the bulk of the content lives, but the SQLite side still needs streaming or paged reads.

**Nothing links the archive to SQLite yet.** A `Message` row and its archive record refer to the same thing by id with no enforced correspondence. Phase 4 has to decide which store is written first and what happens when one write succeeds and the other fails.

**`ContextSnapshot` is not a portable package.** It stores identifiers only and deliberately holds no content, so it is meaningless off the machine that owns the archive. The package format is a separate artifact that inlines what it needs. Both should exist; neither should be renamed into the other.
