"""Dataset v4: the properties that make it able to discriminate.

v3 could not, and the tests here pin each reason why — length, filler that
resists summarization, and controlled position — so a later edit that quietly
undoes one of them fails rather than silently returning the benchmark to
measuring nothing.
"""

import pytest

from open_context.llm.fakes import WordTokenizer
from open_context_eval.adversarial import (
    ADVERSARIAL_DATASET_VERSION,
    ADVERSARIAL_SCENARIOS,
    by_id,
    noise,
)
from open_context_eval.benchmark_dataset import BENCHMARK_SCENARIOS as V2
from open_context_eval.benchmark_dataset_v3 import BENCHMARK_SCENARIOS_V3 as V3
from open_context_eval.dataset import SCENARIOS as V1

TOKENIZER = WordTokenizer(exact=True)
BUDGET = 160


def size(scenario):
    return TOKENIZER.count_text(" ".join(m.get("content", "") for m in scenario.conversation)).count


def test_v4_is_a_new_version_and_edits_nothing_earlier():
    assert ADVERSARIAL_DATASET_VERSION == "v4"
    assert all(s.dataset_version == "v4" for s in ADVERSARIAL_SCENARIOS)
    assert all(s.dataset_version == "v1" for s in V1)
    assert all(s.dataset_version == "v2" for s in V2)
    assert all(s.dataset_version == "v3" for s in V3)


# ----------------------------------------------------------------------
# Length: the reason v3 could not discriminate


def test_every_scenario_forces_real_compression():
    """v3 scenarios were 1.4x to 2.4x at this budget, which is no pressure at all.

    A benchmark whose conversations fit comfortably inside the budget measures
    whether the model can read, not whether compaction kept the right thing.
    """
    for scenario in ADVERSARIAL_SCENARIOS:
        ratio = size(scenario) / BUDGET
        assert ratio >= 8, f"{scenario.scenario_id} is only {ratio:.1f}x at budget {BUDGET}"


def test_v4_conversations_are_far_longer_than_v3():
    assert min(size(s) for s in ADVERSARIAL_SCENARIOS) > max(size(s) for s in V3) * 3


def test_every_scenario_is_a_long_session_not_a_snippet():
    for scenario in ADVERSARIAL_SCENARIOS:
        assert len(scenario.conversation) >= 60, scenario.scenario_id


# ----------------------------------------------------------------------
# Filler that cannot be collapsed for free


def test_noise_is_varied_rather_than_one_repeated_line():
    """v1's filler repeats a sentence with a changing index.

    A summarizer collapses that to a single line at no cost, so it fills the
    prompt without ever consuming summary space — it never forces a choice,
    which is the entire job of filler in an adversarial scenario.
    """
    messages = [m["content"] for m in noise(20)]
    assert len(set(messages)) == len(messages), "every filler message should differ"


def test_noise_is_deterministic():
    """A benchmark whose inputs move between runs cannot be rerun."""
    assert noise(10, seed=3) == noise(10, seed=3)


def test_different_seeds_give_different_filler():
    assert noise(10, seed=1) != noise(10, seed=2)


def test_filler_carries_nothing_a_task_needs():
    """It has to be forgettable, or dropping it would be a real loss."""
    joined = " ".join(m["content"] for m in noise(30)).lower()
    for leak in ("utf-16", "9443", "74", "postgres", "nfs", "correlated subquery"):
        assert leak not in joined


# ----------------------------------------------------------------------
# Position as a controlled variable


def test_the_positional_scenarios_differ_only_in_where_the_trap_sits():
    """Anything separating them is a position effect rather than content.

    "Recency wins" and "primacy wins" are different failures with different
    fixes, and a dataset that scattered its critical information could not tell
    them apart.
    """
    early, middle, late = (
        by_id("adv-critical-early"),
        by_id("adv-critical-middle"),
        by_id("adv-critical-late"),
    )
    for other in (middle, late):
        assert other.task.instruction == early.task.instruction
        assert [c.required_any for c in other.retention_checks] == [
            c.required_any for c in early.retention_checks
        ]
        assert len(other.conversation) == len(early.conversation)


def test_the_trap_actually_sits_where_the_name_says():
    def position(scenario):
        for index, message in enumerate(scenario.conversation):
            if "UTF-16LE" in (message.get("content") or ""):
                return index / len(scenario.conversation)
        raise AssertionError(f"no trap found in {scenario.scenario_id}")

    assert position(by_id("adv-critical-early")) < 0.15
    assert 0.35 < position(by_id("adv-critical-middle")) < 0.65
    assert position(by_id("adv-critical-late")) > 0.85


# ----------------------------------------------------------------------
# The v3 rule is kept


def test_no_check_asserts_absence_by_substring():
    """The defect that made eight of ten Phase 5.8 findings false.

    A correct answer naming what it rejected fails such a check, and does so
    more often the better informed it is.
    """
    for scenario in ADVERSARIAL_SCENARIOS:
        for check in scenario.retention_checks:
            assert not check.forbidden, f"{scenario.scenario_id}/{check.check_id}"
        for criterion in scenario.completion:
            assert not criterion.alternatives, f"{scenario.scenario_id}/{criterion.criterion_id}"


def test_absence_is_asserted_only_through_judged_questions():
    for scenario in ADVERSARIAL_SCENARIOS:
        assert scenario.judged, f"{scenario.scenario_id} asserts nothing a judge could settle"


def test_judged_questions_map_the_rejection_case_to_yes():
    """Naming a rejected option is not a violation, and the judge must be told."""
    for scenario in ADVERSARIAL_SCENARIOS:
        for question in scenario.judged:
            if "only to" in question.question.lower():
                assert "answer yes if" in question.question.lower(), question.question_id


def test_a_scenario_with_no_deterministic_evidence_still_states_what_it_wants():
    """Two scenarios are purely about avoidance and carry judged questions only.

    They state a prohibition and never say what to do instead, so there is no
    positive assertion available — the same situation v3 recorded rather than
    papering over with an invented expected answer.
    """
    bare = [s for s in ADVERSARIAL_SCENARIOS if not s.retention_checks and not s.completion]
    assert {s.scenario_id for s in bare} == {"adv-failed-approach", "adv-negative-constraint"}
    assert all(s.judged for s in bare)


# ----------------------------------------------------------------------


def test_coverage_of_the_adversarial_patterns():
    modes = {mode for scenario in ADVERSARIAL_SCENARIOS for mode in scenario.failure_modes}
    for required in (
        "early_critical_fact",
        "late_critical_fact",
        "long_irrelevant_section",
        "negative_constraint",
        "reversed_decision",
        "failed_approach",
        "exact_value",
        "similar_entities",
        "temporal_state",
        "tool_result",
    ):
        assert required in {mode.value for mode in modes}, required


def test_by_id_rejects_an_unknown_scenario():
    with pytest.raises(KeyError):
        by_id("no-such-scenario")
