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

### Local storage is not encrypted

The SQLite database and archive hold full conversation history in plaintext on disk, under the invoking user's file permissions. Anything sensitive that reached a transcript is sitting there. Treat the storage directory as you would treat a shell history file or a private key directory.

## Data handling

Storage is local by default. Nothing leaves the machine for storage, search, snapshots, or inspection.

Compaction and extraction call an LLM. If that LLM is a hosted API, conversation content is sent to that provider. Local storage and local processing are separate settings and will be reported separately, because "your data stays local" is false when a hosted model is doing the compaction.

## Scope

In scope: prompt injection through archived content, trust-level bypass in compiled context, path traversal in import and export, SQL injection, unsafe deserialization of import files, and permanent loss of source history through a non-explicit operation.

Out of scope: an attacker who already has read access to the storage directory, vulnerabilities in third-party LLM providers, and denial of service from oversized local input.
