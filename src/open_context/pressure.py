"""Reacting to an agent's own context pressure.

**Verified against the real mechanism before a line was written**, which is what
the roadmap requires of this phase — and the reason it is implementable at all.
Claude Code 2.1.195 emits a `PreCompact` hook *before* it compacts, carrying:

    {"hook_event_name": "PreCompact",
     "session_id": "...", "transcript_path": "/…/<uuid>.jsonl",
     "cwd": "/…", "trigger": "auto" | "manual",
     "custom_instructions": "…"}

`trigger: "auto"` **is** the context-pressure signal. The agent has decided its
window is full and is about to discard detail; that instant is the last moment
the full conversation exists, and therefore the right moment to capture it.

**This observes and never interferes.** The binary supports blocking compaction
from this hook — *"Compaction blocked by PreCompact hook; continuing
uncompacted"* — and that power is deliberately not used. A context tool that
could silently prevent an agent from compacting would be able to run a session
out of its window, and a user debugging that would have no reason to suspect the
thing they installed to help.

So the handler writes a package and returns control. It never blocks, and it
never fails the hook: an exception here would surface to the user as their agent
misbehaving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRE_COMPACT = "PreCompact"
SESSION_START = "SessionStart"

AUTOMATIC = "auto"
"""The trigger value that means context pressure rather than a user's request."""


@dataclass(frozen=True)
class HookEvent:
    """What an agent handed us, normalised and never trusted.

    Every field is optional because a hook payload is input from another
    program: a future version may add fields, rename them, or send none of
    them, and a handler that raised on an unexpected shape would break the
    user's agent rather than its own feature.
    """

    event: str = ""
    session_id: str = ""
    transcript_path: str = ""
    cwd: str = ""
    trigger: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pressure(self) -> bool:
        """A compaction the agent decided on by itself."""
        return self.event == PRE_COMPACT and self.trigger == AUTOMATIC

    @classmethod
    def parse(cls, text: str) -> HookEvent:
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(document, dict):
            return cls()
        return cls(
            event=str(document.get("hook_event_name") or ""),
            session_id=str(document.get("session_id") or ""),
            transcript_path=str(document.get("transcript_path") or ""),
            cwd=str(document.get("cwd") or ""),
            trigger=str(document.get("trigger") or ""),
            raw=document,
        )


@dataclass(frozen=True)
class HookResponse:
    """What we hand back.

    ``continue_`` is always true. The field exists because the protocol has it,
    not because this code has a use for it — see the module docstring.
    """

    continue_: bool = True
    suppress_output: bool = True
    system_message: str = ""

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "continue": self.continue_,
            "suppressOutput": self.suppress_output,
        }
        if self.system_message:
            payload["systemMessage"] = self.system_message
        return json.dumps(payload)


@dataclass
class PressureConfig:
    """When to capture, and where to put it.

    ``on_manual`` is off by default. A user who typed `/compact` has decided
    what they want; capturing anyway is the tool asserting it knows better, and
    doubles the disk a session costs for the sake of an event nobody asked about.
    """

    store: Path
    on_automatic: bool = True
    on_manual: bool = False
    min_bytes: int = 4_096
    """Below this the transcript is too small to be worth a package.

    A session that hit compaction while tiny is a strange state, and a directory
    of near-empty packages is how a useful feature becomes something a user turns
    off.
    """

    @property
    def packages(self) -> Path:
        return self.store / "packages"

    @property
    def archives(self) -> Path:
        return self.store / "archive"


def should_capture(event: HookEvent, config: PressureConfig) -> tuple[bool, str]:
    """Whether to act, and why not when the answer is no."""
    if event.event != PRE_COMPACT:
        return False, f"not a compaction event ({event.event or 'unknown'})"
    if event.trigger == AUTOMATIC and not config.on_automatic:
        return False, "automatic capture is disabled"
    if event.trigger != AUTOMATIC and not config.on_manual:
        return False, "this compaction was requested by the user, not by pressure"
    if not event.transcript_path:
        return False, "the event named no transcript"

    source = Path(event.transcript_path)
    if not source.is_file():
        return False, f"no transcript at {source}"
    if source.stat().st_size < config.min_bytes:
        return False, f"transcript is {source.stat().st_size} bytes, below the floor"
    return True, ""


def handle(event: HookEvent, config: PressureConfig) -> tuple[HookResponse, str]:
    """Capture a package if this event calls for one. Never raises.

    Returns the hook response and a line for the log. The response says
    ``continue`` whatever happened, because a failure to make a backup is not a
    reason to interfere with the agent that was running.
    """
    wanted, why_not = should_capture(event, config)
    if not wanted:
        return HookResponse(), f"skipped: {why_not}"

    try:
        from open_context.handoff import handoff
        from open_context.models import ids

        config.packages.mkdir(parents=True, exist_ok=True)
        session = event.session_id or ids.new_id(ids.SESSION)
        safe = "".join(c for c in session if c.isalnum() or c in "-_")[:64] or "session"
        out = config.packages / f"{safe}.ctx"

        package, report = handoff(
            source=event.transcript_path,
            session_id=f"{ids.SESSION}_{safe.replace('-', '')[:24]}",
            archive_root=config.archives,
            title=f"before compaction in {Path(event.cwd).name}" if event.cwd else None,
        )
        out.write_text(package.to_json(), encoding="utf-8")
    except Exception as exc:
        return (
            HookResponse(),
            f"capture failed ({type(exc).__name__}): {exc}",
        )

    return (
        HookResponse(),
        f"captured {report.events_kept} events to {out}",
    )


__all__ = [
    "AUTOMATIC",
    "PRE_COMPACT",
    "SESSION_START",
    "HookEvent",
    "HookResponse",
    "PressureConfig",
    "handle",
    "should_capture",
]
