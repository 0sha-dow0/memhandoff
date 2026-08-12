# Contributing

## Before you open a pull request

This project is built in phases, one layer at a time. See [docs/phases.md](docs/phases.md) for the current position and what is next.

The most useful thing you can do before writing code is open an issue describing the problem. A pull request that implements three phases ahead of the current one will be hard to review and will probably be declined, not because the code is bad but because the layer underneath it is not settled yet.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

All four must pass. CI runs the same ones.

```bash
pytest
ruff check .
ruff format --check .
mypy
```

## Code

Python 3.12 or newer. Full type annotations; mypy runs in strict mode. Small modules, single responsibility, dependencies injected rather than imported at the point of use.

Provider-specific logic stays out of the core. If a module in `models/`, `storage/`, `graph/`, or `compiler/` imports an SDK for a specific model provider, that is a design bug. Providers sit behind the `LLMProvider` protocol.

Every behavior change ships with a test. Tests that need a network call or an API key are marked `llm` and skipped by default.

## Never commit

Real conversation exports. Test fixtures are synthetic. A conversation you exported from your own account still contains other people's words, file paths, and often credentials.

## Reusing other projects' code

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) first. Short version: check the actual license file, preserve the notices, record the commit SHA and the exact modifications, and if the licensing is unclear, reimplement the idea instead of copying the code.

## Commits

Conventional Commits, so the changelog can be assembled from history.

```
feat(storage): add message repository
fix(compiler): respect token budget when packing evidence
docs(phases): record Phase 2 completion
```

## License

Contributions are licensed under Apache-2.0, per section 5 of the license.
