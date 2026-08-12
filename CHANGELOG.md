# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the major version is 0, breaking changes can land in any release.

## [Unreleased]

### Added

- Phase 3. Local immutable archive.
  - Append-only JSONL on the filesystem, one log per session, with a manifest and a rebuildable 12-byte-per-record offset index.
  - Per-record sha256 over canonical JSON, byte-exact with no Unicode normalisation, so corruption in a stored original is always detected.
  - Streaming reads throughout: `read`, `range`, `window`, `tail`, iteration, and `verify` all hold one record at a time. No read-everything API exists.
  - Bounded tail recovery on open. A torn or unparseable final record is truncated and reported; damage elsewhere raises rather than being repaired.
  - Data file is synced before its index entry, so the crash window is always repairable from the data file.
  - 55 new tests, including `tracemalloc` assertions that peak memory does not scale with archive size.

### Changed

- Product realignment. Documentation only; no code, model, or schema change.
  - The product is a portable context package for agent handoff, not a general context runtime or memory system. Added `docs/product-architecture.md`.
  - Three layers made explicit throughout: raw conversation, portable package, agent-specific context.
  - Roadmap rewritten. Evaluation moved to Phases 5.5 and 5.6, directly after the baseline compactor. Provider and tokenizer abstraction moved from Phase 16 to 4.5. Package format became Phase 9, ahead of the compiler.
  - Cross-session memory backends, paging, and the standalone MCP phase dropped. MCP returns as one adapter in Phase 11.
  - Three conflicts recorded between the existing implementation and the new direction: cross-machine provenance versus referential integrity, whole-result-set reads versus the RAM constraint, and `ContextSnapshot` not being a portable package.

### Fixed

- Phase 2.1. Storage integrity corrections.
  - Migrations are now atomic. `executescript` issued an implicit COMMIT before running, so a failing migration left its completed statements permanently applied. Each migration now runs as individually split statements inside one explicit transaction.
  - Every reference is now checked for session locality; `CrossSessionReferenceError` added.
  - A supersession cycle within a single batch is now rejected; `SupersessionCycleError` added.
  - The `raw source -> evidence -> state` provenance layering is documented and pinned by tests. No schema change was required.
  - 28 new tests. No change to the Phase 1 data model or the database schema.

### Added

- Phase 2. SQLite persistence, append and read only.
  - `Database` with per-connection pragmas, WAL on file databases, and an explicit versioned migration list.
  - `Repository` covering `Session`, `Message`, `Evidence`, all ten state types, and `ContextSnapshot`.
  - Referential integrity for every reference Phase 1 could only check the format of, including polymorphic ones.
  - Atomic batch writes with deferred foreign keys, so records that reference each other in a cycle are storable.
  - Supersession derived from edges rather than written, so no historical row is ever rewritten.
  - `STRICT` tables and typed storage errors.
  - 53 new storage tests.
- Phase 1. Pydantic data model, no persistence.
  - `Session`, `Message`, `Evidence`, `ContextSnapshot`, `TokenEstimate`.
  - State items as a discriminated union on `type`: `Goal`, `Constraint`, `Fact`, `Decision`, `Task`, `OpenQuestion`, `Preference`, `Entity`, `Event`, `Artifact`.
  - Prefixed identifiers with format validation on every reference.
  - All records frozen. Change is expressed through `supersedes`, never in place.
  - Timezone-aware UTC timestamps enforced; naive datetimes rejected.
  - Content hashing over NFC-normalised text, verified on construction.
  - `pydantic>=2.8` added as the first runtime dependency.
  - 69 model tests. mypy strict clean across `src`.
- Phase 0. Repository skeleton, packaging, and project governance.
  - `pyproject.toml` with a hatchling build, src layout, and Python 3.12 minimum.
  - `open_context` package with a version and a `py.typed` marker. No application logic.
  - pytest, ruff, and mypy in strict mode, all configured in `pyproject.toml`.
  - GitHub Actions CI running tests, lint, format check, and type check on Python 3.12 and 3.13.
  - Apache-2.0 license.
  - `THIRD_PARTY_NOTICES.md` with the reuse policy and review checklist. Nothing vendored yet.
  - `docs/architecture.md` and `docs/phases.md`.
