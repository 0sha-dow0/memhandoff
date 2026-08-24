# Local retrieval

V2.4. Letting the compiler ask the archive for what a task turns out to need.

```
local archive -> lexical retrieval -> relevant evidence -> context compiler
```

## Why it exists

A measurement, not a hunch. The adversarial benchmark found that what compaction
loses is **exact values** — a port among similar ports, a row count from a tool
result — and *all* of its loss sat in the two scenarios built around them.

A summary cannot keep every number. It does not have to: the archive still holds
them, and a task that names one can go and fetch it.

## Using it

```bash
open-context compile project.ctx \
    --target anthropic --budget 4000 \
    --task "Write the firewall rule for the metrics port" \
    --retrieve 3 --archive ./ctx
```

`--retrieve N` is off by default and needs `--archive`. A package that arrived
from another machine has no archive to search, so a caller who does not ask gets
exactly the behaviour they had before.

`--query` searches for something other than the task, because the two are not
always the same sentence: *"finish the export writer"* is a task, and
*"UTF-16LE port 8082"* is what finds the records it needs.

## Measured

Four scenarios drawn from the failures the adversarial benchmark identified. The
question is whether the compiled context *contains* the exact value — a property
of the compiler, checkable exactly, and a different question from whether a model
then uses it.

| Scenario | Without retrieval | With retrieval |
| --- | --- | --- |
| exact value among confusable ports | ✗ | **✓** |
| answer inside a tool result | ✗ | **✓** |
| row count | ✗ | **✓** |
| encoding requirement | ✗ | ✗ |
| **total** | **0 / 4** | **3 / 4** |

Token cost: **208 → 266, +28%**.

### The one it misses, and why

The encoding scenario asks *"what encoding should the export writer use"*. The
record that answers it reads *"the downstream loader rejects UTF-8; write
UTF-16LE with a BOM"*.

Those two sentences share **no words at all**. After stopwords, the query is
`encoding, should, export, writer, use` and the record is `downstream, loader,
rejects, utf, 8, write, utf, 16, le, bom`. Overlap: none.

**This is the structural limit of lexical retrieval, and it is predictable rather
than random.** It finds a record when the query shares vocabulary with it, and it
cannot find one that answers a question in different words. Asking *"what UTF
encoding does the loader need"* retrieves it immediately.

That is also the honest case *for* embeddings — and the roadmap's rule is that
they go in only when a benchmark demonstrates material benefit. This is the
benchmark that would justify examining them: one scenario in four, with the
failure understood. It is not a reason to add a vector database today.

## What it does not do

**No embeddings, no vector index, no second store**, per the roadmap. For finding
`8082` beside `8080` and `8081`, exact matching is the right instrument anyway —
a port is a token, not a concept.

## How it behaves

**Streamed, never loaded.** Records pass a bounded heap, so peak memory is set by
how many results were asked for and not by how large the archive is. Searching
80,000 records costs what searching ten does. An archive that had to be read into
a list to be searched would have given up the property that made it worth keeping.

**Provenance travels with the text.** Every retrieved line carries the record id
it came from:

```
From the archive, found for this task
- [m00041] "The app listens on 8080, the admin panel on 8081, and metrics on 8082."
```

A record cited without saying which record it was is an assertion, not evidence.
The heading names the origin too, because a reader must be able to tell a record
the task fetched from one the package chose to carry — they were selected by
different things and are worth different amounts of trust.

**Budgeted like every other section.** Retrieved records are fitted to what is
left and truncated deterministically, never appended wholesale. When the budget
is short they are dropped before the session's own state: what the session
concluded outranks what a task might have wanted.

**Deterministic.** Ties resolve by archive position, so the same package, archive,
and query give the same context — including when the records arrive in a
different order.

**Duplicates are suppressed.** Retrieval searches the same archive the package's
evidence came from and will find those records again; printing one twice spends
budget to tell a reader nothing.

**Failure is not fatal.** An archive that cannot be read produces a warning and a
context without the extra records. That context is smaller, not wrong, and an
unreadable archive should not deny a caller the package they already hold.
