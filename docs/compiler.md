# The context compiler

Phase 10. Turning a provider-neutral `.ctx` package into context a specific
agent can use.

```
.ctx  ->  CompiledContext  ->  target renderer
                               ├── openai
                               ├── anthropic
                               ├── gemini
                               ├── local
                               └── generic
```

```bash
python -m open_context compile project.ctx \
    --target anthropic --budget 2000 --task "Finish the export writer."
```

## The one rule: decisions happen once

**The portable layer decides. Targets render.**

`compile_context` selects what fits the budget, on a structure that knows nothing
about any provider. A renderer then receives a `CompiledContext` that is *already
selected and already inside the budget* — with no tokenizer, no budget, and no
package.

That is not a convention. A renderer takes exactly one argument, and that
argument carries no way to measure anything, so it cannot drop, reorder,
summarise, or re-fit even by accident.

**The failure this prevents is quiet.** If each adapter chose what to include,
five adapters would grow into five slightly different compactors. The same `.ctx`
would mean different things depending on where it was sent, and every adapter
would still produce something that looked entirely reasonable. A test asserts the
renderers never gain the ability.

Nothing here reimplements compaction either. Fitting state to a budget is
`fit_state` from the hybrid compactor, imported unchanged — the priority order
that decides what survives a small budget came from a benchmark result, and a
second copy of it would drift from the first.

## What differs between provider families

Less than it looks. The substantive difference is where standing context goes:

| Family | Standing context | Turns |
| --- | --- | --- |
| `openai` | a `system` message, first in the list | `messages` |
| `anthropic` | a top-level `system` string, outside the list | `messages`, first must be `user` |
| `gemini` | a top-level `systemInstruction` | `contents`, roles `user`/`model`, text in `parts` |
| `local` | as `openai` | `messages` |
| `generic` | inlined above the task | one text block |

`local` is a separate name rather than an alias for `openai`. "A local model" is
a deployment choice, not a promise that it will always accept OpenAI's schema;
when one does not, that is where it diverges.

**An unknown target falls back to `generic` rather than failing.** Refusing would
make the format less portable than the plain text it is made of.

**These are request shapes, not an integration.** Nothing here sends anything or
imports a vendor SDK. Choosing an integration mechanism is Phase 11's job, and
guessing at it here is what that phase explicitly warns against.

## Budgeting

Sections are placed in priority order, each given what the previous ones left:

1. **Task** — reserved first, and if it does not fit, compilation fails. A
   context without its task is not a smaller context, it is a different one.
2. **State** — capped at `state_fraction` (0.6 by default) of the budget. A cap,
   not a reservation: state that does not need its share leaves the rest behind.
3. **Evidence** — excerpts, as many as fit.
4. **Recent context** — taken from the **end**, because it exists to show what
   was just happening.

Only evidence references carrying an excerpt are rendered. One that resolves to
a sequence number and nothing else is real provenance and useless here: a model
cannot follow a pointer into an archive it does not have, so printing the number
would spend budget on something only a validator can use.

**What was dropped is part of the result**, not a log line. A context that
silently omitted half the state looks exactly like one built from half as much
state, and a caller handing this to an agent needs to tell those apart.

## Neutrality

`target_model` is recorded and never acted on. Nothing in the compiler changes
what it writes based on which model was named — a compiler that quietly produced
different content per model would make the package's provider-neutrality a
fiction. A test pins it: two requests differing only in `target_model` compile to
identical sections.

## Token counts are estimates, and say so

The CLI counts with a word-based tokenizer and labels the number estimated.
Counting exactly needs the target model's own tokenizer, which the command cannot
obtain without calling a provider — and compiling a context is not a reason to
make a network request. An estimate a caller knows is an estimate is the honest
offer; one presented as exact is how a compiled context overflows.
