# Hierarchical context

V2.3. A session is not the top of the world: it sits inside a project, alongside
other sessions that already established things.

```
PROJECT
 ↓
SESSION
 ↓
EVENT
```

## The logical model, and nothing else

The roadmap is explicit that a graph-shaped model is not a reason to install a
graph engine. This is three levels and a parent link — a dictionary. No second
store, no service, no index. A test asserts the module reaches for no storage
engine at all.

**There is no TASK level**, although the roadmap's sketch has one. Nothing in a
real transcript marks where one task ends and the next begins, so a task boundary
would have to be inferred — and an invented boundary looks exactly like an
observed one. It goes in when something real delimits it.

## A project is a repository, not a working directory

The obvious reading — one `cwd`, one project — is wrong, and real data says so
immediately. Of six sessions measured here, two ran in:

```
/Users/…/open-context
/Users/…/open-context/files (4)/open-context-runtime 2
```

That is one project and a directory inside it. Splitting them hides a session's
own history from itself, which is precisely what this level exists to prevent.

So a project is the nearest ancestor holding a repository marker — `.git`, `.hg`,
`.svn` — and the working directory only when there is none. That is a filesystem
question, answered by looking rather than by parsing a path for something that
resembles a project name, and a test pins that distinction.

**On the machine these were measured on, none of the six directories is a
repository**, so each is correctly its own project. The merging behaviour is
verified against a real marker rather than asserted.

## What each level is for

| Level | Default retention | Holds |
| --- | --- | --- |
| project | critical | goals, constraints, decisions — things that outlive a session |
| session | important | facts, tasks, open questions belonging to this stretch of work |
| event | reconstructable | a detail of one exchange; the archive still has it |

Nothing here deletes. It is an ordering for what to carry when there is not room
for everything.

**An item that states its own importance is believed.** A `CRITICAL` retention
set by extraction or by a person is a judgement about *that* item; the type-based
default is a guess about its kind in general, and the specific claim wins. The
data model separately refuses a critical item with no source — the most
load-bearing claims are exactly the ones that must be checkable.

## What it buys

Monday's constraint is still true on Tuesday. A session that could only see
itself would ask again, or decide differently.

```
session ses_22222…
  read      20 events
  carried   1 item established earlier on this project
```

Only project-scoped state crosses. Carrying everything would make the second
session read the first one whole, which is the thing this project exists to
avoid. The same constraint restated in three sessions is carried once — a
receiving agent shown it three times learns nothing the third time.

## Inherited state is labelled, never merged

The package builder refuses state belonging to another session, on the ground
that mixing them would present one agent's work as another's. **That guard is
right**, so inherited context gets its own field rather than an exemption:

```json
{
  "state":     [ /* what this session concluded */ ],
  "inherited": [ /* what other sessions on this project established */ ]
}
```

It compiles into its own section, under a heading that says where it came from.
*"We decided X"* and *"another session on this project decided X"* are both worth
carrying and are not the same sentence; a reader who cannot tell them apart
cannot judge which to revisit.

When the budget is short, inherited state is dropped before the session's own.

## What went wrong on the way

Two failures, both caught by tests rather than by review, and both from the same
mistake — bolting carry-forward on without respecting an invariant that already
existed.

**Foreign state was merged into `state`.** The builder refused it, correctly. The
fix was a separate field, not a weaker check.

**State leaked between unrelated projects.** Sibling lookup read every stored
session in the directory, because nothing recorded which project a stored session
belonged to. A constraint from an unrelated project reached a package. Stored
state now carries its `project_root`, and a session whose root differs is skipped.
