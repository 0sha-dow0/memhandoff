# Open Context Runtime

Take a long AI session and hand it to a different agent, without explaining the project again.

> **Status: Phase 2.1.** Data model and local SQLite persistence work. There is no compaction, no import, no package format, and no agent integration yet. See [docs/phases.md](docs/phases.md). Nothing user-facing works yet, and the README will say so until it does.

## The problem

You spend two days working with one agent. The conversation outgrows the context window. The agent compacts it with its own provider-specific mechanism, and that compacted state is stuck inside that provider. Moving to another agent means starting the explanation from scratch.

Summarizing the transcript and deleting the original does not solve this either. It loses the API contract agreed in message 40, the reason Kafka was rejected in message 180, and the constraint stated once and never repeated.

## The idea

Compact the session into a **portable context package**: a structured, provider-neutral artifact that another agent can continue from.

```
Claude Code
    |
long-running conversation
    |
Open Context
    |
portable context package        5k to 20k tokens
    |
Codex, Gemini, another Claude, a local model
    |
continue the same work
```

Three layers stay distinct:

| Layer | What | Portable |
| --- | --- | --- |
| Raw conversation | The full history, 500k tokens | No, stays on your machine |
| Portable package | Goals, constraints, decisions and why, open work, evidence | Yes |
| Agent context | What a specific model actually receives, compiled from the package | No, generated per agent |

The package is not an LLM summary. It carries typed records with sources, statuses, and provenance, so you can ask why something was kept and get message IDs back.

## What this is not

Not a vector database, not an agent framework, not a RAG library, not a hosted memory service, not a chatbot, not a replacement for any model provider. It is the compaction and interoperability layer. Where another project does a job well, this one should call it rather than reimplement it.

## Design constraints

- Local-first. No account, no hosted database, no mandatory network call for storage, search, inspection, or export. `Open Context + Ollama` must work with nothing else installed.
- Lightweight. SQLite and the filesystem. No Docker, no server, no second database. A large archive must not mean proportional RAM.
- Provider-neutral packages. Import from one agent, continue in another.
- Source-recoverable compaction. The original history is never destroyed, and important compacted information points back at its source through provenance.
- Inspectable. Ask why something was kept, get message IDs back.

## Requirements

Python 3.12 or newer.

## Development

```bash
git clone <repository-url>
cd open-context-runtime
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest
ruff check .
ruff format --check .
mypy
```

## Reusing other projects

This project is Apache-2.0 and reuses compatible open-source code where it makes sense. It does that at the point the relevant component is being built, not up front, so the repository does not become a pile of borrowed subsystems that never fit together. The rule and the review checklist are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
