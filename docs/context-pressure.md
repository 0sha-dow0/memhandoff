# Automatic context pressure

V2.1. Capture a portable package at the moment an agent decides its window is
full — the last instant the whole conversation still exists.

## Set it up

```bash
open-context hook --store ~/.open-context
```

Wire that into Claude Code's `PreCompact` hook, in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "open-context hook --store ~/.open-context",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

From then on, every time the agent compacts under pressure, a validated `.ctx`
lands in `~/.open-context/packages/`.

## The mechanism, verified

The roadmap for this phase says: *do not implement until actual agent hooks are
verified*. They were, against the shipping binary — Claude Code 2.1.195 — rather
than against documentation or assumption.

**`PreCompact` fires before compaction** and carries:

```json
{"hook_event_name": "PreCompact",
 "session_id": "912bcf79-…", "transcript_path": "/…/<uuid>.jsonl",
 "cwd": "/…", "trigger": "auto", "custom_instructions": ""}
```

**`trigger` is the whole signal.** `"auto"` means the agent decided by itself
that its window was full — that is context pressure. `"manual"` means somebody
typed `/compact`.

The moment matters. After compaction the detail is gone; before it, the full
transcript is still on disk. A hook that fires *before* is the difference between
capturing a session and capturing a summary of one.

## What this deliberately does not do

**It never blocks compaction**, though the protocol lets it. The binary contains
the string *"Compaction blocked by PreCompact hook; continuing uncompacted"*, so
a `decision` in the response would stop the agent compacting.

That power is not used, and a test asserts the handler never votes. A context
tool able to silently prevent an agent from compacting could run a session out of
its window, and a user debugging that would have no reason to suspect the thing
they installed to help. Backing something up is not a licence to interfere with
it.

**It never fails the hook.** The response says `continue` whatever happened, and
the command exits 0 even when it captured nothing — a non-zero exit is how a hook
tells the agent something is wrong, and a failed backup is not that.

**It ignores manual compactions by default.** A user who typed `/compact` has
decided what they want; capturing anyway asserts the tool knows better and
doubles what a session costs on disk. `on_manual` turns it on.

**It skips transcripts under 4 KB.** A session that hit compaction while tiny is
a strange state, and a directory of near-empty packages is how a useful feature
becomes one people turn off.

## What you get

```
open-context: captured 303 events to ~/.open-context/packages/912bcf79-….ctx
```

A package that validates at provenance depth against the archive written beside
it, holding recent conversation and a verifiable pointer to the full history.
Add `--extract` to the equivalent `handoff` call to turn it into goals and
constraints; the hook path does not extract, because a hook has seconds and no
credential of its own.

Measured on a real transcript: **303 events captured, 4.4 KB package**, well
inside the hook timeout.
