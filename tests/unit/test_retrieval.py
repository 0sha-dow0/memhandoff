"""Finding the record that had the number in it.

This exists because of a measurement: the adversarial benchmark found that what
compaction loses is *exact values*, and all of the loss on that run sat in the
two scenarios built around them. A summary cannot keep every number; the archive
still has them.
"""

import resource

from open_context.archive.records import ArchiveRecord
from open_context.retrieval import Filter, evidence_for, search, tokenize


def record(seq, text, *, kind="message", event_type="user_message", tool=None):
    payload = {"type": event_type, "text": text}
    if tool:
        payload["tool_name"] = tool
    return ArchiveRecord(seq=seq, id=f"m{seq:05d}", kind=kind, payload=payload)


def haystack(size=300):
    """A long conversation with the answer buried in it."""
    records = [
        record(n, f"turn {n}: routine work on the reporting export, nothing notable")
        for n in range(size)
    ]
    records[117] = record(
        117, "The app listens on 8080, the admin panel on 8081, and metrics on 8082."
    )
    records[204] = record(
        204,
        "ss -ltn output: LISTEN 0 4096 *:8080 *:8081 *:8082",
        event_type="tool_result",
        tool="Bash",
    )
    return records


# ----------------------------------------------------------------------
# Tokenising


def test_numbers_survive_whole():
    """`8082` must tokenise as `8082`. Splitting on non-alphanumerics turns
    `10.0.0.5` into four useless tokens — exactly the strings this is for."""
    assert "8082" in tokenize("metrics on 8082")
    assert "10.0.0.5" in tokenize("the host is 10.0.0.5")
    assert "utf" in tokenize("write utf-16le") and "16" in tokenize("write utf-16le")


def test_case_does_not_matter():
    assert tokenize("LISTEN") == tokenize("listen")


# ----------------------------------------------------------------------
# Finding the answer


def test_the_record_holding_the_answer_ranks_first():
    hits = search(haystack(), "which port do the metrics listen on", limit=3)
    assert hits
    assert hits[0].seq == 117


def test_an_exact_value_is_found_among_confusable_ones():
    """8080, 8081 and 8082 are one character apart. This is the failure mode the
    benchmark identified, and a literal match is the right instrument — `8082`
    is a token, not a concept."""
    hits = search(haystack(), "8082", limit=2)
    assert hits
    assert "8082" in hits[0].matched


def test_a_query_of_only_common_words_finds_nothing():
    """Better than returning the first three records with confidence."""
    assert search(haystack(), "the and of it") == []


def test_the_excerpt_shows_the_match_not_the_start_of_the_record():
    """A record matched on a value in its middle should show that value; an
    excerpt from the front proves only that the record is long."""
    long_record = [record(0, "preamble " * 200 + "the metrics port is 8082 " + "tail " * 200)]
    (hit,) = search(long_record, "8082", limit=1)
    assert "8082" in hit.excerpt
    assert hit.excerpt.startswith("…")


def test_rarer_terms_carry_the_query():
    """Without weighting, a record repeating a common term outranks the one
    holding the answer."""
    records = [
        record(0, "port port port port port port port port"),
        record(1, "the metrics port is 8082"),
    ]
    hits = search(records, "metrics port 8082", limit=2)
    assert hits[0].seq == 1


# ----------------------------------------------------------------------
# Metadata filtering


def test_a_filter_narrows_before_scoring():
    hits = search(haystack(), "8082 listening", limit=3, where=Filter(event_types={"tool_result"}))
    assert [hit.seq for hit in hits] == [204]


def test_a_tool_filter_selects_by_name():
    hits = search(haystack(), "8082", limit=3, where=Filter(tool_names={"Bash"}))
    assert all(hit.seq == 204 for hit in hits)


def test_a_range_filter_excludes_what_is_outside_it():
    hits = search(haystack(), "8082", limit=5, where=Filter(min_seq=150))
    assert hits and all(hit.seq >= 150 for hit in hits)


def test_a_filter_that_matches_nothing_returns_nothing():
    assert search(haystack(), "8082", where=Filter(event_types={"nonexistent"})) == []


# ----------------------------------------------------------------------
# Without loading the archive


def test_memory_is_bounded_by_the_result_count_not_the_archive_size():
    """An archive that must be read into a list to be searched has given up the
    property that made it worth keeping."""
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    hits = search(
        (record(n, f"turn {n} " + "padding " * 60) for n in range(60_000)), "turn 59999", limit=3
    )
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert len(hits) <= 3
    unit = 1024 * 1024 if __import__("os").uname().sysname == "Darwin" else 1024
    assert (after - before) / unit < 120


def test_search_accepts_a_stream_not_just_a_list():
    hits = search((r for r in haystack()), "metrics 8082", limit=1)
    assert hits


def test_evidence_is_shaped_for_the_compiler():
    """It wants text to show a model, not scores to rank."""
    pairs = list(evidence_for(haystack(), "which port for metrics", limit=2))
    assert pairs
    assert all(isinstance(record_id, str) and excerpt for record_id, excerpt in pairs)
