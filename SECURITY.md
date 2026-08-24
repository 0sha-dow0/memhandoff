# Security

## Reporting a vulnerability

Report privately through GitHub's "Report a vulnerability" flow on the Security tab, not through a public issue.

Include the affected version or commit, reproduction steps, and what an attacker gains. Expect an acknowledgment within seven days.

This is a pre-release project maintained in spare time. There is no patch SLA yet, and only the default branch is supported.

## Threat model

The runtime ingests conversation transcripts and tool output, stores them, and later feeds derived state back into a model. That shape creates three problems worth stating up front.

### Conversation content is untrusted input

A transcript can contain text authored by a web page, a tool result, or another model. Text such as "remember this permanently and always obey it" is data, not an instruction. Extracted state carries an origin and a trust level, and nothing recovered from the archive may override system or developer instructions in a compiled context.

### Memory can be poisoned

An attacker who can get text into a session can try to plant a durable false fact that surfaces in every later compiled context. Provenance is the defense that makes this auditable: every state item points at the messages that produced it, so a poisoned fact can be traced to its origin and removed.

### A `.ctx` package is untrusted input

The portable package format exists so context can travel between machines and
between agents. That is also what makes it the project's main attack surface: a
`.ctx` is a file a stranger can hand you, whose contents you will paste into a
model's system prompt.

**Nothing in this project executes anything because it appeared in imported
context.** There is no `eval`, no shell, no dynamic import, and no path
resolution driven by package content anywhere a package can reach — asserted by a
test that parses those modules, so the property survives an edit rather than
resting on having been careful once.

`archive.location` is the obvious trap: a filesystem path chosen by whoever built
the package. It is carried verbatim, reported, and **never opened**. To this code
`/etc/passwd` and `../../../../etc/shadow` are strings.

`open-context validate` reports what a package carries before you rely on it:
instruction-shaped text, items too large to be what they claim, state that cannot
be traced to a source, and absolute paths. **It does not sanitize, and says so.**
Prompt injection cannot be detected reliably; a function that stripped "the
dangerous parts" would make a promise it cannot keep, and a caller who believed
it would stop being careful. A clean report prints that it is not a guarantee of
safety.

The compiler adds the one real mitigation available: every compiled context opens
by stating that what follows is a record of earlier work rather than instructions
to the reader, because in a system prompt "the constraint we agreed" and "an
instruction to you" are otherwise indistinguishable.

### Importing context must never execute anything

Loading, inspecting, validating, or compiling a `.ctx` package is parsing, not
evaluation. Nothing in a package is treated as code: no `pickle`, no `eval`, no
`exec`, no import driven by a name inside the file, no shell invocation, and no
path in a package is used to write outside the store you name. A package that
causes execution is a vulnerability, not a feature — report it.

The corollary is that a package can still carry text designed to manipulate
whatever model reads it downstream. That is a prompt-injection problem, handled
above; it is not prevented by the no-execution rule.

### Credentials

Keys are read from the environment, never from code or a config file, and are
excluded from `repr` so they cannot reach a log line, a traceback, or an
assertion dump. No credential is written into a `.ctx` package, a benchmark
result, an archive record, or a run fingerprint.

If you find a credential in this repository, in a release artifact, or in any
file this project generates, please report it privately through the flow above
rather than opening an issue.

### Local storage is not encrypted

The SQLite database and archive hold full conversation history in plaintext on disk, under the invoking user's file permissions. Anything sensitive that reached a transcript is sitting there. Treat the storage directory as you would treat a shell history file or a private key directory.

## If you are filing a reproduction

A `.ctx` package contains real conversation text, and an archive contains all of
it. Both are ordinary files that are easy to attach without looking inside.

Before attaching anything to an issue, remove API keys, tokens, `.env` contents,
private conversations, proprietary source, customer data, and personal
information — yours and other people's. `open-context inspect` prints a package
in a readable form, which is the quickest way to see what you are about to share.

**If a faithful reproduction needs private material, build a smaller synthetic
one that fails the same way.** That is usually a better report: it isolates the
property that causes the failure, and anyone can run it.

## Data handling

Storage is local by default. Nothing leaves the machine for storage, search, snapshots, or inspection.

Compaction and extraction call an LLM. If that LLM is a hosted API, conversation content is sent to that provider. Local storage and local processing are separate settings and will be reported separately, because "your data stays local" is false when a hosted model is doing the compaction.

## Scope

In scope: prompt injection through archived content, trust-level bypass in compiled context, path traversal in import and export, SQL injection, unsafe deserialization of import files, and permanent loss of source history through a non-explicit operation.

Out of scope: an attacker who already has read access to the storage directory, vulnerabilities in third-party LLM providers, and denial of service from oversized local input.
