"""The extraction prompt, versioned.

```
conversation -> typed work state
```

**Not a summary prompt.** Phase 5's asks a model to write prose for another
agent to read. This one asks it to identify discrete items and say what kind
each is, so the result is records with types, sources, and statuses rather than
a paragraph that asserts things.

**JSON is prompted for, not negotiated.** Both shipped providers raise
``UnsupportedCapabilityError`` for ``structured_output`` and neither advertises
the capability, because whether a given model honours a schema varies and
OpenRouter routes to many. So the contract is stated in the prompt and enforced
by a parser that expects to be disappointed. A malformed reply is a recorded
failure, never a silent empty extraction.
"""

from __future__ import annotations

from open_context.compaction import Prompt

EXTRACTION_PROMPT_V1 = Prompt(
    name="extraction",
    version=1,
    text="""\
You are reading a working session between a person and an AI agent, and \
recording the WORK STATE it arrived at, so a different agent can continue \
without the original conversation.

Extract discrete items. Do not write a summary, do not editorialise, and do not \
invent anything that is not supported by the conversation.

Return ONLY a JSON object of this shape, with no prose before or after it and \
no markdown fence:

{
  "items": [
    {
      "type": "goal|constraint|fact|decision|task|open_question|entity",
      "content": "one self-contained sentence stating the item",
      "source_ids": ["the id of every message this came from"],
      "status": "active|superseded|rejected",
      "confidence": 0.0 to 1.0,
      "rationale": "decisions only: WHY this was chosen",
      "alternatives": ["decisions only: options considered and not taken"],
      "hard": true or false,
      "supersedes_content": "if this replaces an earlier item, that item's content",
      "task_status": "tasks only: open|in_progress|blocked|done|cancelled"
    }
  ]
}

Rules that matter more than completeness:

EVERY item must carry the source_ids of the messages it came from. An item you \
cannot attribute to a specific message does not belong in the output.

A CONSTRAINT that says what must NOT happen is as important as one that says \
what must. Record "do not use Redis" as a constraint with hard=true. Never drop \
a prohibition because it is phrased negatively.

A DECISION requires a rationale. If the conversation records a choice but never \
says why, use the rationale "not stated in the conversation" rather than \
inventing one.

When a later part of the conversation REVERSES an earlier decision or fact, \
record BOTH: the earlier one with status "superseded" and supersedes_content \
empty, and the later one with status "active" and supersedes_content set to the \
earlier item's content. Do not silently keep only the winner — which decision \
was overturned, and that it was overturned, is part of the state.

An approach that was TRIED AND FAILED is a fact, with content saying both what \
was tried and why it failed. It is not a task and not a decision.

Prefer fewer, well-attributed items over many vague ones. Say nothing rather \
than guess.""",
)
"""The instruction. Versioned and content-hashed, so an extraction result can
record which prompt produced it and a later run can tell the prompt changed."""

CURRENT_STATE_HEADER = """\
CURRENT STATE ALREADY RECORDED FROM EARLIER IN THIS SESSION.

These items are already stored. Do not repeat one unless the new messages \
change it. If a new message overturns one of them, emit the replacement and set \
supersedes_content to the earlier item's content COPIED EXACTLY as written \
below — it is matched literally, and a paraphrase will not resolve.
"""
"""Shown to the model when extracting a later range of a session.

**Without it, incremental extraction cannot express a reversal.** A model given
only the new messages has never seen the decision being overturned, so it cannot
name it, and the supersession claim it would need to make is unavailable to it.
The first incremental run against a real model produced exactly that: a correct
new decision, a correct new constraint, and no link to the decision they
replaced.
"""


__all__ = ["CURRENT_STATE_HEADER", "EXTRACTION_PROMPT_V1"]
