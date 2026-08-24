"""Target renderers: one compiled context, five request shapes.

**A renderer decides nothing.** It receives a ``CompiledContext`` that is already
selected and already inside its budget, and turns it into the shape one provider
family expects. It gets no budget and no tokenizer, so it cannot drop, reorder,
summarise, or re-fit anything — which is the point. Five adapters each choosing
what to include would be five slightly different compactors, and the same `.ctx`
would quietly mean different things depending on where it was sent.

**What actually differs between families is where standing context goes.**
That is a smaller difference than it looks, and it is the only one that matters
here:

| Family | Standing context | Turns |
| --- | --- | --- |
| `openai` | a `system` message, first in the list | `messages` |
| `anthropic` | a top-level `system` string, outside the list | `messages`, first must be `user` |
| `gemini` | a top-level `systemInstruction` | `contents`, roles `user`/`model`, text in `parts` |
| `local` | a `system` message, like `openai` | `messages` |
| `generic` | inlined above the task | one text block |

**These are request shapes, not an integration.** Nothing here sends anything or
imports a vendor SDK, and the runtime's own providers speak chat-completions over
raw HTTP. Choosing an integration mechanism is Phase 11's job, and doing it here
on an assumption would be the mistake that phase explicitly warns against.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from open_context.compiler.portable import CompiledContext

TargetRenderer = Callable[[CompiledContext], dict[str, Any]]

GENERIC = "generic"
"""The fallback, and the one that always works.

A family nobody has written an adapter for still gets a usable context, because
plain text is the intersection of every chat API in existence. An unknown target
falls back here rather than failing: refusing to compile would make the format
less portable than the plain text it is made of.
"""


def render_openai(context: CompiledContext) -> dict[str, Any]:
    """`messages`, with standing context as the first `system` message."""
    messages: list[dict[str, str]] = []
    if context.instructions:
        messages.append({"role": "system", "content": context.instructions})
    if context.task:
        messages.append({"role": "user", "content": context.task})
    return {"messages": messages}


def render_local(context: CompiledContext) -> dict[str, Any]:
    """Same shape as `openai`.

    A separate name rather than an alias, because "a local model" is a
    deployment choice and not a claim that it will always accept OpenAI's
    schema. When one does not, this is where that diverges — and until then,
    saying so is more honest than implying the two are the same thing.
    """
    return render_openai(context)


def render_anthropic(context: CompiledContext) -> dict[str, Any]:
    """`system` outside the message list, which must begin with a user turn.

    An empty `messages` list is not valid, so a context with no task still gets
    one turn. The alternative is emitting a request that cannot be sent, which
    would push the problem to whoever tried to use it.
    """
    payload: dict[str, Any] = {"messages": []}
    if context.instructions:
        payload["system"] = context.instructions
    payload["messages"].append(
        {"role": "user", "content": context.task or "Continue the work described above."}
    )
    return payload


def render_gemini(context: CompiledContext) -> dict[str, Any]:
    """`contents` with `parts`, and standing context as `systemInstruction`."""
    payload: dict[str, Any] = {}
    if context.instructions:
        payload["systemInstruction"] = {"parts": [{"text": context.instructions}]}
    payload["contents"] = [
        {
            "role": "user",
            "parts": [{"text": context.task or "Continue the work described above."}],
        }
    ]
    return payload


def render_generic(context: CompiledContext) -> dict[str, Any]:
    """One text block: standing context, then the task.

    The lowest common denominator, and deliberately so. Anything that accepts a
    string accepts this.
    """
    blocks = [context.instructions, context.task]
    return {"text": "\n\n".join(block for block in blocks if block)}


TARGETS: Mapping[str, TargetRenderer] = {
    "openai": render_openai,
    "anthropic": render_anthropic,
    "gemini": render_gemini,
    "local": render_local,
    GENERIC: render_generic,
}


def render_for(target: str, context: CompiledContext) -> dict[str, Any]:
    """Render for a named family, falling back to `generic` for an unknown one."""
    return TARGETS.get(target.lower(), render_generic)(context)


def known_targets() -> tuple[str, ...]:
    return tuple(sorted(TARGETS))


__all__ = [
    "GENERIC",
    "TARGETS",
    "TargetRenderer",
    "known_targets",
    "render_anthropic",
    "render_for",
    "render_gemini",
    "render_generic",
    "render_local",
    "render_openai",
]
