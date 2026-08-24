# The `.ctx` format

Phase 9. The product artifact: what one agent knows, written down so another can
pick it up.

```
python -m open_context pack     --session ses_… --store ./ctx --out project.ctx
python -m open_context validate project.ctx --archive ./ctx/archive
python -m open_context inspect  project.ctx
```

## One JSON document, not an archive

A zip would let a package carry the whole conversation cheaply. It is rejected
for that reason.

**Human-inspectable is a requirement of this format**, and a container you must
unpack before you can read it does not meet it. A `.ctx` opens in an editor,
diffs in a code review, and survives being pasted into a bug report. When a
package grows too large for that, the answer is to reference the archive rather
than inline it — which is the answer the format already gives.

The file is indented and key-sorted for reading. The *hash* is computed over a
canonical serialisation, so readability costs nothing.

## What is in it

| Section | Holds | Required |
| --- | --- | --- |
| `manifest` | format, version, package id, session, provenance of the package itself | yes |
| `state` | goals, constraints, facts, decisions, tasks, open questions, entities | — |
| `evidence` | one reference per state item, back to the archive records it rests on | — |
| `recent` | verbatim recent turns, when worth carrying | no |
| `archive` | a pointer to the archive, with hashes for the records cited | no |
| `content_hash` | sha256 over everything above | yes |

### It carries interpretation, not history

State items and the evidence they rest on — not the conversation. Embedding the
archive by default would make a package a second copy that immediately begins to
diverge from the first, and the archive already exists and is already verifiable.

### Superseded items are excluded

A package is what is believed **now**. A reader handed both sides of a reversal
without being told which won is worse off than one handed neither. The count of
what was dropped survives in `manifest.metadata`, so a reader can see that
history existed without having to reason about it.

An item that supersedes something outside the package is flagged as a warning
rather than an error — that is the expected shape, and recording it tells the
reader the item replaced something.

## Provenance

Every state item gets an `EvidenceReference`, and a reference that cannot be
followed is still recorded. **Dropping it would destroy the only trail back to
the source**; saying it is unresolvable costs one field and is true.

References are resolved two ways:

1. An explicit mapping from the caller, who may have tracked the derivation.
2. Otherwise, the item's `sources` matched against archive record **ids**.

The second deserves a note, because Phase 6 concluded no correspondence existed
between the two id schemes. That is true of **sequence numbers** and not of
**ids**: `ArchiveRecord.id` is defined as *"identity of the underlying item,
preserved as given"*, so an archive fed the same messages the state was extracted
from already holds exactly the ids `sources` names. Matching on an identifier
both sides agreed on is a lookup. Nothing is minted, and a source naming
something this archive does not hold simply does not resolve.

Each reference may carry a short excerpt, capped at 240 characters, so a package
separated from its archive can still show a reader what an item rests on rather
than only asserting that something does.

## Integrity

`content_hash` is sha256 over the canonical serialisation of every other field.
Canonical means sorted keys and fixed separators, so a package that passed
through a reformatting tool still verifies — a hash sensitive to key order would
call every reserialised package tampered with.

The archive reference hashes **only the records the package cites**. Hashing all
of them would grow the reference with the conversation and turn a pointer into a
manifest of the whole history; hashing none would leave a receiver unable to tell
one archive from another.

## Validation has two depths, and says which it reached

| Depth | Needs | Establishes |
| --- | --- | --- |
| `structure` | the package | it hashes correctly, its references are internally consistent, nothing superseded slipped in |
| `provenance` | the package **and** the archive | the records it cites exist and are the records it hashed |

**A package that passes the shallow check and fails the deep one is the dangerous
case**: internally impeccable, and describing a different archive. Reporting
"valid" without naming the depth would hide exactly that, so the depth is part of
the verdict and the CLI says on stderr when it did not have an archive to check
against.

`validate` exits non-zero when a package is invalid, and with a *different*
non-zero code when the file could not be read as a package at all. "I could not
read it" and "I read it and it is wrong" are different problems.

## Versioning

`format` and `format_version` are the two fields a reader must trust before it
understands anything else, so they are required and checked before the body is
parsed.

A package declaring a version **above** what the reader knows is refused. It may
well parse cleanly while a field has quietly changed meaning, and the failure of
that guess is silent. Older versions are fine: the format only adds.

## Determinism

Two builds of the same state at the same `created_at` are byte-identical. The
creation timestamp is the one field that legitimately varies, and `build` accepts
it explicitly so a reproducibility check is possible at all.

## What it does not do

- **No compression, no binary encoding.** Both trade the readability the format
  exists to provide for a saving that referencing the archive already gives.
- **No embedded archive.** By design, above.
- **No signature.** The hash detects alteration; it does not establish who built
  the package. Signing needs a key distribution story, and nothing has needed one
  yet.
- **No resolution of `location`.** Where the archive sat on the machine that
  built a package is meaningless on the machine that reads it. It is recorded as
  a hint and nothing follows it automatically.
