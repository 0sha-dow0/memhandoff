"""Finding the record that had the number in it.

**This exists because of a measurement, not a hunch.** The Phase 8.5 adversarial
benchmark found that what compaction actually loses is *exact values* — a port
number among similar ports, a row count from a tool result. All of the loss on
that run sat in the two scenarios built around them. A summary cannot keep every
number, and it does not have to: the archive still has them, and a compiled
context can go and get the few that matter.

**Lexical only, deliberately.** The roadmap allows embeddings if a benchmark
shows material benefit, and none has been run — so there is no vector index here,
no model call, and no second store. That is also the right default for the
failure being fixed: `8082` is a token, not a concept, and exact-match retrieval
is what finds it. Semantic search is the wrong instrument for a literal.

**Nothing is loaded whole.** Records stream past a bounded heap, so peak memory
is set by how many results were asked for and not by how large the archive is.
An archive that must be read into a list to be searched has given up the property
that made it worth keeping.
"""

from __future__ import annotations

import heapq
import math
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from open_context.archive.records import ArchiveRecord

TOKEN = re.compile(r"[A-Za-z_]+|\d+(?:\.\d+)*")
"""Words and numbers, with numbers kept whole.

`8082` must survive tokenisation as `8082`, and `10.0.0.5` as one address rather
than four numbers — an IP, a version, and a port are the literals this exists to
find. The repeating group is why: a single optional decimal handles `0.5` and
splits `10.0.0.5` down the middle.
"""

_STOPWORDS = """
a an and are as at be by for from has have in is it its of on or that the
this to was were will with what which how why when do does did i you we
"""
STOPWORDS = frozenset(_STOPWORDS.split())
"""Words too common to discriminate. Short on purpose: a long stoplist mostly
removes terms that a rarity weight would have discounted anyway."""


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN.findall(text or "")]


def content_of(record: ArchiveRecord) -> str:
    """Readable text from a record, whatever shape its payload is."""
    payload = record.payload
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    for key in ("text", "content", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


@dataclass(frozen=True)
class Hit:
    """One record that matched, and how well."""

    seq: int
    record_id: str
    score: float
    excerpt: str
    matched: tuple[str, ...]

    def __lt__(self, other: Hit) -> bool:
        """Ordered by score, then by position, so ties resolve to the earlier record."""
        return (self.score, -self.seq) < (other.score, -other.seq)


@dataclass
class Filter:
    """Metadata narrowing, applied before any scoring.

    Cheap and exact, and it runs first: a query restricted to tool results should
    not pay to score the messages around them.
    """

    kinds: frozenset[str] = field(default_factory=frozenset)
    event_types: frozenset[str] = field(default_factory=frozenset)
    tool_names: frozenset[str] = field(default_factory=frozenset)
    min_seq: int = 0
    max_seq: int | None = None

    def allows(self, record: ArchiveRecord) -> bool:
        if record.seq < self.min_seq:
            return False
        if self.max_seq is not None and record.seq > self.max_seq:
            return False
        if self.kinds and record.kind not in self.kinds:
            return False

        payload = record.payload if isinstance(record.payload, dict) else {}
        if self.event_types and str(payload.get("type") or "") not in self.event_types:
            return False
        return not (self.tool_names and str(payload.get("tool_name") or "") not in self.tool_names)


EXCERPT = 240


def _excerpt_around(text: str, terms: Iterable[str]) -> str:
    """A window of the text near the first match, not the first 240 characters.

    A record matched on a value in its middle should show that value; an excerpt
    taken from the front would prove only that the record is long.
    """
    lowered = text.lower()
    position = min(
        (lowered.find(term) for term in terms if term and lowered.find(term) >= 0),
        default=-1,
    )
    if position < 0:
        return " ".join(text.split())[:EXCERPT]
    start = max(0, position - EXCERPT // 3)
    window = text[start : start + EXCERPT]
    return ("… " if start else "") + " ".join(window.split())


def search(
    records: Iterable[ArchiveRecord],
    query: str,
    *,
    limit: int = 5,
    where: Filter | None = None,
    scorer: Callable[[list[str], list[str]], float] | None = None,
) -> list[Hit]:
    """The best matches for a query, streaming.

    Memory is bounded by ``limit``: records pass through a heap of that size and
    are dropped. A million-record archive and a ten-record archive cost the same
    to search, which is the property the exit criterion is about.
    """
    terms = [term for term in tokenize(query) if term not in STOPWORDS]
    if not terms:
        return []

    predicate = where or Filter()
    score = scorer or _score
    best: list[Hit] = []

    for record in records:
        if not predicate.allows(record):
            continue
        text = content_of(record)
        if not text:
            continue

        tokens = tokenize(text)
        value = score(terms, tokens)
        if value <= 0:
            continue

        hit = Hit(
            seq=record.seq,
            record_id=record.id,
            score=value,
            excerpt=_excerpt_around(text, terms),
            matched=tuple(sorted({term for term in terms if term in tokens})),
        )
        if len(best) < limit:
            heapq.heappush(best, hit)
        elif best[0] < hit:
            heapq.heapreplace(best, hit)

    return sorted(best, reverse=True)


def _score(terms: list[str], tokens: list[str]) -> float:
    """Term frequency, damped, weighted towards rarer and more literal terms.

    **Rare terms carry the query.** A search for "the metrics port 8082" is
    really a search for `8082`; `port` narrows it a little and `metrics` less.
    Without weighting, a record repeating `port` six times outranks the one
    holding the answer.

    Frequency is damped with a logarithm because the second mention of a term
    says much less than the first, and an undamped count lets one long record
    dominate on repetition alone.

    No corpus statistics. Real IDF would need a pass over the archive to build,
    which is the thing this is avoiding; term length and shape are a decent local
    stand-in, and being a stand-in is why the function is replaceable.
    """
    if not tokens:
        return 0.0

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    total = 0.0
    for term in terms:
        seen = counts.get(term, 0)
        if not seen:
            continue
        weight = 1.0 + min(len(term), 12) / 6.0
        if any(character.isdigit() for character in term):
            # A literal value is the point of this module.
            weight *= 2.0
        total += weight * (1.0 + math.log(seen))

    coverage = sum(1 for term in terms if counts.get(term)) / len(terms)
    return total * (0.5 + coverage)


def evidence_for(
    records: Iterable[ArchiveRecord], task: str, *, limit: int = 3, where: Filter | None = None
) -> Iterator[tuple[str, str]]:
    """``(record_id, excerpt)`` for the archive records a task most needs.

    Shaped for the compiler, which wants text to show a model rather than scores
    to rank.
    """
    for hit in search(records, task, limit=limit, where=where):
        yield hit.record_id, hit.excerpt


__all__ = [
    "EXCERPT",
    "STOPWORDS",
    "TOKEN",
    "Filter",
    "Hit",
    "content_of",
    "evidence_for",
    "search",
    "tokenize",
]
