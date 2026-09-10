"""Exact values a summary will not keep.

Every deterministic failure in the `v4` adversarial matrix is one of these, on
every compacted arm and on no reference arm: a write timeout of `74` seconds
among `47`, `7`, and `4700`, and a port `9443` among other ports. Nothing else
fails anywhere. The mechanism is visible in the answers themselves — one reads
"the original context does not specify the exact value" and then invents a
plausible one, which is worse than admitting the gap.

A summariser drops these because they are, to it, low-salience detail: the
sentence reads fine without the number. But an exact value is the one thing a
reader cannot reconstruct and cannot safely guess, and it costs a handful of
tokens to keep. So the values are lifted out deterministically, before the model
is asked for anything, and carried alongside the summary rather than through it.

This is a ledger, not a retrieval index. It answers "what exact values were
stated" and nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VALUE = re.compile(
    r"""
    (?<![\w.])
    (?:
        \d{1,3}(?:\.\d{1,3}){3}(?::\d+)?   # an address, optionally with a port
      | \d+(?:\.\d+)+                       # a version, or a decimal
      | 0[xX][0-9a-fA-F]+                   # hex
      | \d[\d,_]*                           # a plain number, grouped or not
    )
    (?!\w)        # not the head of an identifier
    (?!\.\d)      # and not cut out of the middle of a decimal
    """,
    re.VERBOSE,
)
"""Numeric literals, kept whole.

`10.0.0.5` is one address and not four numbers, `1.2.3` is one version, and
`4,700` is one quantity. The lookarounds keep it from biting into identifiers
like `utf8` or `v2beta`, where the digits are part of a name rather than a value
someone chose — while still allowing a value that ends a sentence, since a
trailing full stop is punctuation and `74.` is still `74`.
"""

_LABEL = re.compile(r"[A-Za-z][\w-]*")
_SENTENCE_END = re.compile(r"[.!?;:\n]")
_TRIVIAL = frozenset({"0", "1", "2"})
"""Ordinals and counts that carry no configuration meaning on their own."""

_FILLER = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "then",
        "this",
        "to",
        "was",
        "were",
        "we",
        "you",
        "i",
        "they",
        "he",
        "she",
        "there",
        "here",
        "with",
        "set",
        "setting",
        "use",
        "using",
        "tried",
        "try",
        "after",
        "before",
        "when",
        "while",
        "so",
        "also",
        "both",
        "each",
        "earlier",
        "later",
        "now",
        "still",
        "just",
        "only",
        "again",
        "already",
        "then",
        "thing",
        "things",
        "one",
    ]
)
"""Words that describe nothing about the value they sit next to.

A label is meant to answer "what is this the value *of*". Trailing `and` or
leading `Earlier we` answer nothing and spend budget saying so.
"""


@dataclass(frozen=True)
class Literal:
    """One exact value, with just enough words to say what it belongs to."""

    value: str
    label: str

    def render(self) -> str:
        return f"{self.label}: {self.value}" if self.label else self.value


def _label_for(text: str, start: int, end: int) -> str:
    """The nearest words before the value, and the unit right after it.

    "the write timeout is 74 seconds" has to come back as something a reader can
    act on, not a bare `74`. Words before carry what the value *is*; the word
    after carries the unit, without which `74` and `74000` look interchangeable.
    """
    # Never look past the end of the previous sentence: the words before a full
    # stop describe something else, and splicing them on produces labels like
    # "holds port Host" that read as though one fact spanned two.
    left = text[max(0, start - 90) : start]
    left = _SENTENCE_END.split(left)[-1]

    before = _LABEL.findall(left)
    while before and before[-1].lower() in _FILLER:
        before.pop()
    head_words = before[-3:]
    while head_words and head_words[0].lower() in _FILLER:
        head_words.pop(0)
    head = " ".join(head_words)

    right = _SENTENCE_END.split(text[end : end + 14])[0]
    after = _LABEL.findall(right)
    unit = after[0] if after else ""
    if unit and (unit.lower() in _FILLER or len(unit) > 12):
        unit = ""

    if head and unit:
        return f"{head} ({unit})"
    return head or unit


def extract_literals(text: str, *, limit: int = 24) -> tuple[Literal, ...]:
    """Exact values in the order they were first stated, with what they belong to.

    First statement wins: a value repeated later is the same fact, and the
    earliest mention is the one whose surrounding words defined it. A value no
    words describe is skipped rather than carried as a bare number.
    """
    found: dict[str, Literal] = {}
    for match in VALUE.finditer(text):
        value = match.group(0)
        if value in _TRIVIAL or value in found:
            continue
        label = _label_for(text, match.start(), match.end())
        # A value nothing describes cannot be acted on, and a bare number sitting
        # beside the one that matters makes the ledger harder to read, not
        # easier. The adversarial dataset is built out of exactly that: 47 and 7
        # are there to be confused with 74.
        if not label:
            continue
        found[value] = Literal(value=value, label=label)
        if len(found) >= limit:
            break
    return tuple(found.values())


def missing_from(literals: tuple[Literal, ...], *texts: str) -> tuple[Literal, ...]:
    """The values none of ``texts`` still states.

    A value the summary kept needs no ledger entry, and repeating it would spend
    budget saying something twice.
    """
    kept = "\n".join(texts)
    return tuple(
        item
        for item in literals
        if not re.search(rf"(?<![\w.]){re.escape(item.value)}(?![\w.])", kept)
    )


def render_ledger(literals: tuple[Literal, ...]) -> str:
    """The ledger as it appears in compacted context, or nothing at all."""
    if not literals:
        return ""
    lines = "\n".join(f"- {item.render()}" for item in literals)
    return f"Exact values stated earlier, preserved verbatim:\n{lines}"


__all__ = ["VALUE", "Literal", "extract_literals", "missing_from", "render_ledger"]
