"""Prompts, versioned, and the one place a downstream request is built.

**This module is the leakage boundary.** ``build_continuation_request`` takes a
context string and a task string. It does not take a scenario, so it cannot put
an expected answer into a prompt, and it cannot reach the original conversation
when the strategy under test was supposed to remove it. The prevention is that
the information is not in scope, not that the code remembers to omit it.

**The downstream agent is not told which strategy it is running.** The
instruction is identical for every arm; only the context differs. Telling a model
it is reading a summary invites it to hedge, and telling it it has the full
conversation invites confidence, either of which would be measured as a property
of the strategy.
"""

from __future__ import annotations

from open_context.compaction import Prompt
from open_context.llm import ChatMessage, GenerationRequest

CONTINUATION_PROMPT_V1 = Prompt(
    name="continuation",
    version=1,
    text="""\
You are continuing work that another agent previously performed. You did not see \
the earlier session; what follows is the context you have been given, and it is \
all you have.

Use only that context. If it does not settle something you need, say so plainly \
rather than filling the gap with a plausible guess — an invented detail is worse \
here than an acknowledged gap.

Do the task you are given directly. No preamble about what you were provided and \
no commentary on the handover itself.""",
)

SIMPLE_SUMMARY_PROMPT_V1 = Prompt(
    name="simple_summary",
    version=1,
    text="""\
Summarize this working session so that a different AI agent can pick up the work \
and continue it. That agent will see only your summary.

Keep what the work depends on: the objective, decisions and why they were made, \
constraints, what is done, what is unfinished, approaches that failed, and \
specific values that would be expensive to rediscover. Do not invent anything. \
Where the session left something unresolved, say so.

Output prose and short lists. No preamble.""",
)

CONTEXT_HEADER = "CONTEXT FROM THE EARLIER SESSION"
TASK_HEADER = "YOUR TASK"


def build_continuation_request(
    context: str,
    task_instruction: str,
    *,
    prompt: Prompt = CONTINUATION_PROMPT_V1,
    max_output_tokens: int | None = None,
) -> GenerationRequest:
    """The downstream request, identical across strategies except for ``context``.

    Two strings in, one request out. Nothing else is reachable from here, which
    is what makes leakage a structural impossibility rather than a review item.
    """
    body = f"{CONTEXT_HEADER}\n\n{context}\n\n{TASK_HEADER}\n\n{task_instruction}"
    return GenerationRequest.of(
        ChatMessage.system(prompt.text),
        ChatMessage.user(body),
        max_output_tokens=max_output_tokens,
    )


def build_summary_request(
    conversation: str,
    *,
    prompt: Prompt = SIMPLE_SUMMARY_PROMPT_V1,
    max_output_tokens: int | None = None,
) -> GenerationRequest:
    """The request behind the plain-summary arm."""
    return GenerationRequest.of(
        ChatMessage.system(prompt.text),
        ChatMessage.user(conversation),
        max_output_tokens=max_output_tokens,
    )


__all__ = [
    "CONTEXT_HEADER",
    "CONTINUATION_PROMPT_V1",
    "SIMPLE_SUMMARY_PROMPT_V1",
    "TASK_HEADER",
    "build_continuation_request",
    "build_summary_request",
]
