"""Versioned prompts.

A benchmark result is meaningless if nobody can tell which prompt produced it.
Each prompt carries a name, a version, and a hash of its own text, so an edit
that changes results is visible in the results rather than only in the git log.

One prompt. The same instruction serves both the single-pass summary and the
pass that combines chunk summaries, because the instruction — summarize so
another agent can continue — is the same either way; only the material differs,
and the material is described in the user message. A prompt per situation would
multiply the things a benchmark has to control for.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """A named, versioned instruction."""

    name: str
    version: int
    text: str

    @property
    def identifier(self) -> str:
        return f"{self.name}_v{self.version}"

    @property
    def content_hash(self) -> str:
        """Fingerprint of the prompt text.

        Recorded on every result, so a benchmark comparing two runs can tell
        whether the prompt changed underneath it even if the version did not.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


BASELINE_SUMMARY_V1 = Prompt(
    name="baseline_summary",
    version=1,
    text="""\
You are compacting the earlier part of a working session so that a DIFFERENT AI \
agent can pick up the work and continue it. That agent will not see the original \
conversation. It will see your summary plus the most recent messages.

Write for continuation, not for a reader. This is a working handover, not a story \
and not minutes of a meeting.

Preserve, where the conversation actually contains them:
- what is being worked on and what the current objective is
- decisions that were made, and the reasoning behind them
- constraints and requirements, especially ones stated once
- what has been completed
- what is in progress or still unresolved
- approaches that were tried and rejected, and why they failed
- specific facts that would be expensive to rediscover: names, versions, numbers, \
paths, identifiers, error messages
- files, artifacts, and commands that matter

Rules:
- Do not invent anything. If something was not said, do not write it.
- Do not resolve ambiguity by guessing. If the conversation left something unclear \
or contradictory, say so plainly.
- Prefer specifics over description. "Chose Postgres over MongoDB because the \
existing infra already runs Postgres" is useful; "discussed database options" is not.
- Omit pleasantries, restatements, and reasoning that led nowhere.
- Keep the ordering that makes the work understandable, not necessarily the order \
things were said.

Output prose and short lists. No preamble, no closing remarks, no meta-commentary \
about the summary itself.""",
)

SINGLE_PASS_HEADER = "Here is the earlier part of the session, in order:"
"""Framing for raw conversation material."""

CHUNK_HEADER = "Here is one section of the earlier part of the session, in order:"
"""Framing for one chunk when the history did not fit in a single request."""

COMBINE_HEADER = (
    "Here are summaries of consecutive sections of the earlier part of the session, "
    "in order. Combine them into one summary under the same rules. Material already "
    "condensed here must not be condensed further than the rules allow:"
)
"""Framing for the pass that merges chunk summaries."""


__all__ = [
    "BASELINE_SUMMARY_V1",
    "CHUNK_HEADER",
    "COMBINE_HEADER",
    "SINGLE_PASS_HEADER",
    "Prompt",
]
