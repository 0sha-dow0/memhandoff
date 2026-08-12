"""Identifier scheme.

Every identifier carries a type prefix, so a reference is self-describing:
``msg_9f2c...`` is a message and ``ev_41ab...`` is an evidence record. That lets
provenance be a flat list of strings instead of a list of (kind, id) pairs, and
lets a message id pasted where an evidence id belongs fail validation instead of
dangling until something dereferences it.

Identifiers are random, not ordered. Ordering comes from explicit fields
(``Message.seq``, ``Session.created_at``), never from an id.
"""

from __future__ import annotations

import re
from typing import Final
from uuid import uuid4

MESSAGE: Final = "msg"
SESSION: Final = "ses"
EVIDENCE: Final = "ev"
SNAPSHOT: Final = "snap"

GOAL: Final = "goal"
CONSTRAINT: Final = "con"
FACT: Final = "fact"
DECISION: Final = "dec"
TASK: Final = "task"
OPEN_QUESTION: Final = "oq"
PREFERENCE: Final = "pref"
ENTITY: Final = "ent"
EVENT: Final = "evt"
ARTIFACT: Final = "art"

KNOWN_PREFIXES: Final[frozenset[str]] = frozenset(
    {
        MESSAGE,
        SESSION,
        EVIDENCE,
        SNAPSHOT,
        GOAL,
        CONSTRAINT,
        FACT,
        DECISION,
        TASK,
        OPEN_QUESTION,
        PREFERENCE,
        ENTITY,
        EVENT,
        ARTIFACT,
    }
)

_SUFFIX_LENGTH: Final = 24
_ID_PATTERN: Final = re.compile(rf"^([a-z]+)_([0-9a-f]{{{_SUFFIX_LENGTH}}})$")


def new_id(prefix: str) -> str:
    """Mint an identifier for the given prefix."""
    if prefix not in KNOWN_PREFIXES:
        raise ValueError(f"unknown id prefix {prefix!r}")
    return f"{prefix}_{uuid4().hex[:_SUFFIX_LENGTH]}"


def prefix_of(value: str) -> str:
    """Return the type prefix of a well-formed identifier."""
    match = _ID_PATTERN.match(value)
    if match is None:
        raise ValueError(f"malformed identifier {value!r}")
    return match.group(1)


def is_valid(value: str) -> bool:
    """Whether the value is a well-formed identifier with a known prefix."""
    match = _ID_PATTERN.match(value)
    return match is not None and match.group(1) in KNOWN_PREFIXES


def validate_id(value: str, *, expected: str) -> str:
    """Validate an identifier and require a specific prefix."""
    if prefix_of(value) != expected:
        raise ValueError(f"expected an id with prefix {expected!r}, got {value!r}")
    return value


def validate_ref(value: str, *, allowed: frozenset[str]) -> str:
    """Validate a reference to another record, restricted to allowed prefixes."""
    found = prefix_of(value)
    if found not in allowed:
        permitted = ", ".join(sorted(allowed))
        raise ValueError(f"reference {value!r} has prefix {found!r}; expected one of {permitted}")
    return value
