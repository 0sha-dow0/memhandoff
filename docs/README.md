# Documentation

Start with [getting-started.md](getting-started.md).

## By topic

| Topic | Where |
| --- | --- |
| Install and first handoff | [getting-started.md](getting-started.md) |
| What the system is, and why | [architecture.md](architecture.md), [product-architecture.md](product-architecture.md) |
| The `.ctx` package format | [ctx-format.md](ctx-format.md) |
| Compiling a package for an agent | [compiler.md](compiler.md) |
| Retrieving evidence from the archive | [retrieval.md](retrieval.md) |
| Cross-agent handoff, and Claude Code's real format | [handoff.md](handoff.md) |
| Automatic capture under context pressure | [context-pressure.md](context-pressure.md) |
| Compaction | [baseline-compaction.md](baseline-compaction.md), [hybrid.md](hybrid.md) |
| Extraction and state | [extraction.md](extraction.md), [state.md](state.md) |
| Incremental extraction | [incremental.md](incremental.md) |
| Extracting a real (large) session | [extraction-at-scale.md](extraction-at-scale.md) |
| Projects, sessions, and inherited context | [hierarchy.md](hierarchy.md) |
| Providers, tokenizers, free-model policy | [llm.md](llm.md) |
| Benchmark methodology and results | [benchmark.md](benchmark.md), [evaluation.md](evaluation.md), [adversarial.md](adversarial.md) |
| Security, performance, compatibility | [hardening.md](hardening.md) |
| Storage internals | [storage.md](storage.md), [archive.md](archive.md), [data-model.md](data-model.md), [import.md](import.md) |
| Phase-by-phase history | [phases.md](phases.md) |
| Contributing | [../CONTRIBUTING.md](../CONTRIBUTING.md) |

## Reading the benchmark claims

Every number in these docs comes from a run in this repository, and the runs that
went wrong are documented beside the ones that did not — a void run is kept in
`benchmarks/results/void/` with an explanation, because a discarded run is
deleted at the cost of the lesson in it.

Where a result is negative, it is stated as a negative result. The project does
not claim to beat a plain summary, because on its own hardest dataset it does not.
