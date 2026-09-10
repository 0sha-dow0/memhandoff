"""The exact-value ledger.

Every deterministic failure in the `v4` adversarial matrix is an exact value a
summariser discarded — a write timeout among similar timeouts, a port among
similar ports — on every compacted arm and on no reference arm. This module is
the deterministic answer to that, so what it refuses to carry matters as much as
what it keeps: a bare number beside the one that matters makes the ledger worse.
"""

from open_context.compaction.literals import (
    extract_literals,
    missing_from,
    render_ledger,
)


def values(text: str) -> list[str]:
    return [item.value for item in extract_literals(text)]


def labelled(text: str) -> dict[str, str]:
    return {item.value: item.label for item in extract_literals(text)}


# ----------------------------------------------------------------------
# What counts as a value


def test_a_number_keeps_the_words_that_say_what_it_is():
    assert labelled("the write timeout is 74 seconds") == {"74": "write timeout (seconds)"}


def test_a_value_ending_a_sentence_is_still_a_value():
    """Conversations are sentences, and a trailing full stop is punctuation."""
    assert values("the sidecar holds port 9443.") == ["9443"]


def test_an_address_is_one_value_and_not_four_numbers():
    assert values("the host is 10.0.0.5 today") == ["10.0.0.5"]


def test_an_address_with_a_port_stays_together():
    assert values("it cannot bind 0.0.0.0:9443 now") == ["0.0.0.0:9443"]


def test_a_grouped_number_is_one_quantity():
    assert values("the batch holds 4,700 rows") == ["4,700"]


def test_digits_inside_a_name_are_not_a_value():
    """`utf8` and `v2beta` are names. Nobody chose `8` as a setting."""
    assert values("write utf8 through the v2beta endpoint") == []


def test_a_decimal_is_not_split_down_the_middle():
    assert values("running version 2.1.3 in production") == ["2.1.3"]


# ----------------------------------------------------------------------
# What is deliberately left out


def test_a_value_no_words_describe_is_skipped():
    """The adversarial dataset is built out of confusable neighbours.

    Carrying a bare `47` next to a labelled `74` spends budget making the ledger
    harder to read.
    """
    assert values("we tried 47 and 7") == []


def test_a_label_never_reaches_across_a_sentence_boundary():
    """Splicing the previous sentence on produces labels that read as one fact."""
    assert labelled("holds port 9443. Host is 10.0.0.5 now")["10.0.0.5"] == "Host"


def test_the_first_statement_of_a_value_wins():
    text = "the write timeout is 74 seconds, and later the timeout was still 74"
    assert labelled(text)["74"] == "write timeout (seconds)"


def test_extraction_is_bounded():
    text = " ".join(f"setting{i} is {i + 10} units" for i in range(60))
    assert len(extract_literals(text)) <= 24


# ----------------------------------------------------------------------
# What the summary already kept


def test_a_value_the_summary_still_states_needs_no_entry():
    literals = extract_literals("the write timeout is 74 seconds")
    assert missing_from(literals, "they set the write timeout to 74") == ()


def test_a_value_the_summary_dropped_is_carried():
    literals = extract_literals("the write timeout is 74 seconds")
    [carried] = missing_from(literals, "they discussed timeout configuration")
    assert carried.value == "74"


def test_a_different_number_does_not_count_as_keeping_this_one():
    literals = extract_literals("the write timeout is 74 seconds")
    assert [c.value for c in missing_from(literals, "the timeout is 7400")] == ["74"]


def test_the_recent_window_also_counts_as_keeping_a_value():
    literals = extract_literals("the write timeout is 74 seconds")
    assert missing_from(literals, "a summary", "verbatim: set it to 74") == ()


# ----------------------------------------------------------------------
# Rendering


def test_an_empty_ledger_renders_as_nothing_at_all():
    assert render_ledger(()) == ""


def test_the_ledger_names_every_value_it_carries():
    text = "the write timeout is 74 seconds and the sidecar holds port 9443."
    rendered = render_ledger(missing_from(extract_literals(text), "they discussed configuration"))
    assert "74" in rendered and "9443" in rendered
    assert "write timeout" in rendered
