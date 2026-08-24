"""Token counting, and the invariant that an estimate is never called exact."""

import pytest

from open_context.llm import (
    Capability,
    CharacterRatioTokenizer,
    ChatMessage,
    ModelInfo,
    TokenCount,
    TokenizationUnavailableError,
    Tokenizer,
    UnavailableTokenizer,
    exact_counting_available,
)
from open_context.llm.fakes import WordTokenizer
from open_context.llm.tokens import MIXED_METHOD

INFO = ModelInfo(provider="fake", model="m", context_window=1000, tokenizer="fake-words")


def test_a_count_carries_how_it_was_produced():
    count = TokenCount(count=12, exact=True, method="cl100k_base")
    assert (count.count, count.exact, count.method) == (12, True, "cl100k_base")


def test_a_count_must_say_how_it_was_produced():
    """A number with no provenance cannot be judged, so it is not constructible."""
    with pytest.raises(ValueError, match="how it was produced"):
        TokenCount(count=1, exact=True, method="")


def test_a_negative_count_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        TokenCount(count=-1, exact=True, method="m")


# ----------------------------------------------------------------------
# The invariant


def test_adding_an_estimate_to_an_exact_count_gives_an_estimate():
    """The enforcement mechanism: a total is only as good as its worst part."""
    exact = TokenCount(count=10, exact=True, method="cl100k_base")
    estimate = TokenCount(count=5, exact=False, method="chars/4")

    combined = exact + estimate
    assert combined.count == 15
    assert not combined.exact, "an estimate contaminates the total, by design"
    assert combined.method == MIXED_METHOD


def test_two_exact_counts_stay_exact():
    a = TokenCount(count=10, exact=True, method="cl100k_base")
    b = TokenCount(count=5, exact=True, method="cl100k_base")
    assert (a + b) == TokenCount(count=15, exact=True, method="cl100k_base")


def test_there_is_no_way_to_add_back_to_exact():
    """Whatever order you sum in, one estimate makes the whole thing estimated."""
    counts = [
        TokenCount(count=1, exact=True, method="x"),
        TokenCount(count=1, exact=False, method="x"),
        TokenCount(count=1, exact=True, method="x"),
    ]
    assert not TokenCount.total(counts).exact
    assert not TokenCount.total(reversed(counts)).exact


def test_totalling_nothing_is_exactly_zero():
    total = TokenCount.total([])
    assert total.count == 0
    assert total.exact


def test_totalling_one_count_preserves_its_method():
    only = TokenCount(count=7, exact=False, method="chars/4")
    assert TokenCount.total([only]) == only


def test_a_count_says_out_loud_whether_it_is_exact():
    assert "estimated" in str(TokenCount(count=3, exact=False, method="chars/4"))
    assert "exact" in str(TokenCount(count=3, exact=True, method="cl100k_base"))


# ----------------------------------------------------------------------
# Tokenizers


def test_the_word_tokenizer_counts_text_exactly():
    tokenizer = WordTokenizer(exact=True)
    count = tokenizer.count_text("one two three four")
    assert count.count == 4
    assert count.exact


def test_a_tokenizer_may_be_exact_for_text_and_not_for_messages():
    """Knowing a vocabulary is not knowing a chat template."""
    tokenizer = WordTokenizer(exact=True, exact_messages=False)

    assert tokenizer.count_text("one two").exact
    assert not tokenizer.count_messages([ChatMessage.user("one two")]).exact


def test_message_counts_include_framing():
    tokenizer = WordTokenizer(exact=True, message_overhead=3)
    messages = [ChatMessage.user("one two"), ChatMessage.assistant("three")]
    assert tokenizer.count_messages(messages).count == 3 + 3 * 2


def test_the_character_heuristic_is_never_exact():
    """However plausible the number, the method cannot support the claim."""
    tokenizer = CharacterRatioTokenizer(INFO)
    assert not tokenizer.count_text("x" * 400).exact
    assert not tokenizer.count_messages([ChatMessage.user("hello")]).exact


def test_the_character_heuristic_is_deterministic_and_rounds_up():
    tokenizer = CharacterRatioTokenizer(INFO, characters_per_token=4.0)
    assert tokenizer.count_text("x" * 400).count == 100
    assert tokenizer.count_text("xxxxx").count == 2, "a partial token still costs one"
    assert tokenizer.count_text("").count == 0
    assert tokenizer.count_text("abc") == tokenizer.count_text("abc")


def test_the_character_heuristic_names_its_ratio():
    assert CharacterRatioTokenizer(INFO, characters_per_token=3.5).count_text("x").method == (
        "chars/3.5"
    )


def test_the_character_heuristic_rejects_nonsense_settings():
    with pytest.raises(ValueError, match="must be positive"):
        CharacterRatioTokenizer(INFO, characters_per_token=0)
    with pytest.raises(ValueError, match="must not be negative"):
        CharacterRatioTokenizer(INFO, message_overhead=-1)


# ----------------------------------------------------------------------
# Unavailable


def test_an_unavailable_tokenizer_raises_rather_than_guessing():
    """The alternative would be an estimate the caller never asked for."""
    tokenizer = UnavailableTokenizer(INFO, reason="tiktoken is not installed")

    with pytest.raises(TokenizationUnavailableError, match="tiktoken is not installed"):
        tokenizer.count_text("anything")
    with pytest.raises(TokenizationUnavailableError):
        tokenizer.count_messages([ChatMessage.user("anything")])


def test_an_unavailable_tokenizer_still_knows_its_model():
    """So a caller can report the context window it cannot yet fill."""
    tokenizer = UnavailableTokenizer(INFO)
    assert tokenizer.model_info().context_window == 1000


# ----------------------------------------------------------------------
# Capability


def test_exact_counting_is_advertised_by_the_model():
    assert exact_counting_available(WordTokenizer(exact=True))
    assert not exact_counting_available(WordTokenizer(exact=False))


def test_tokenizers_satisfy_the_protocol():
    for tokenizer in (
        WordTokenizer(),
        CharacterRatioTokenizer(INFO),
        UnavailableTokenizer(INFO),
    ):
        assert isinstance(tokenizer, Tokenizer)


def test_a_context_window_and_a_count_are_enough_to_budget():
    """The capability later phases need. No budgeting logic exists here yet."""
    tokenizer = WordTokenizer(
        ModelInfo(
            provider="fake",
            model="m",
            context_window=128_000,
            capabilities=frozenset({Capability.EXACT_TOKEN_COUNT}),
        )
    )
    used = tokenizer.count_text("word " * 1_000)
    window = tokenizer.model_info().context_window

    assert window is not None
    assert window - used.count == 127_000
    assert used.exact, "and the caller can tell whether that number can be trusted"
