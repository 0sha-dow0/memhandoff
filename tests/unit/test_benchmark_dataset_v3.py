"""Dataset v3: deterministic checks assert presence, judged questions assert absence.

The rule exists because the Phase 5.8 run showed that asserting absence by
substring fails on correct answers, and fails them in a biased direction. These
tests pin the rule, pin what it cost, and pin that the earlier versions were not
edited to get it.
"""

from open_context_eval.benchmark_dataset import BENCHMARK_DATASET_VERSION
from open_context_eval.benchmark_dataset import BENCHMARK_SCENARIOS as V2
from open_context_eval.benchmark_dataset_v3 import (
    BENCHMARK_DATASET_V3_VERSION,
    STRANDED,
    by_id,
)
from open_context_eval.benchmark_dataset_v3 import (
    BENCHMARK_SCENARIOS_V3 as V3,
)
from open_context_eval.dataset import SCENARIOS as V1


def test_v3_is_a_new_version_not_an_edit():
    assert BENCHMARK_DATASET_V3_VERSION == "v3" != BENCHMARK_DATASET_VERSION
    assert all(s.dataset_version == "v3" for s in V3)


def test_v2_is_not_modified():
    """The whole reason v3 exists as a separate version.

    Editing a dataset after seeing results is how a benchmark stops meaning
    anything, so the earlier one has to survive intact and be checked.
    """
    assert all(s.dataset_version == "v2" for s in V2)
    assert any(c.forbidden for s in V2 for c in s.retention_checks), (
        "v2 still has its negative checks; v3 did not reach back and edit them"
    )
    assert any(c.alternatives for s in V2 for c in s.completion)


def test_v1_is_still_untouched():
    assert all(s.dataset_version == "v1" for s in V1)
    assert all(s.completion == () for s in V1)


def test_v3_carries_v2s_conversations_verbatim():
    """Only the scoring changed. A changed conversation would be a new benchmark."""
    for old, new in zip(V2, V3, strict=True):
        assert new.scenario_id == old.scenario_id
        assert new.conversation == old.conversation
        assert new.task == old.task
        assert new.failure_modes == old.failure_modes


# ----------------------------------------------------------------------
# The rule


def test_no_retention_check_asserts_absence():
    offenders = [(s.scenario_id, c.check_id) for s in V3 for c in s.retention_checks if c.forbidden]
    assert offenders == [], f"v3 must not assert absence by substring: {offenders}"


def test_no_completion_criterion_asserts_absence():
    offenders = [
        (s.scenario_id, c.criterion_id) for s in V3 for c in s.completion if c.alternatives
    ]
    assert offenders == [], f"v3 must not assert absence by substring: {offenders}"


def test_every_surviving_check_still_asserts_something():
    for scenario in V3:
        for check in scenario.retention_checks:
            assert check.required or check.required_any, (
                f"{scenario.scenario_id}/{check.check_id} asserts nothing"
            )


def test_a_check_with_positive_evidence_keeps_it():
    """Dropping `forbidden` must not throw away the half that worked."""
    check = next(c for c in by_id("reversed-decision").retention_checks)
    assert check.check_id == "latest-decision"
    assert check.required == ("sqlite",)
    assert check.forbidden == ()


def test_a_purely_negative_check_is_dropped_not_emptied():
    """An empty check would be a check that always passes, which is worse than none."""
    assert by_id("negative-constraint").retention_checks == ()
    assert by_id("failed-approach").retention_checks == ()


def test_a_criterion_with_expected_keeps_it():
    criterion = next(c for c in by_id("exact-value").completion)
    assert criterion.expected == ("17",)
    assert criterion.alternatives == ()


# ----------------------------------------------------------------------
# What replaced them


def test_every_removed_assertion_became_a_judged_question():
    """Nothing the dataset cared about was silently dropped.

    Every scenario that lost a negative assertion has a question stating what
    that assertion was for — whether the assertion was the whole check or only
    the discriminating half of a completion criterion.
    """
    lost = {
        s.scenario_id
        for s in V2
        if any(c.forbidden for c in s.retention_checks) or any(c.alternatives for c in s.completion)
    }
    assert lost, "the fixture would be vacuous otherwise"
    for scenario_id in lost:
        assert by_id(scenario_id).judged, f"{scenario_id} lost an assertion and gained no question"


def test_a_hedging_question_names_the_options_it_replaced():
    """Derived from the criterion, so it cannot drift from what it replaced."""
    question = next(
        q for q in by_id("reversed-decision").judged if q.question_id.endswith("-not-hedged")
    )
    assert "postgresql" in question.question
    assert "postgres" in question.question
    assert "only to" in question.question, "naming a rejected option is not a violation"


def test_every_criterion_that_lost_alternatives_gained_a_hedging_question():
    for old, new in zip(V2, V3, strict=True):
        for criterion in old.completion:
            if not (criterion.alternatives and criterion.expected):
                continue
            assert any(
                q.question_id == f"{criterion.criterion_id}-not-hedged" for q in new.judged
            ), f"{old.scenario_id}/{criterion.criterion_id}"


def test_judged_questions_say_that_naming_a_rejected_option_is_allowed():
    """The exact distinction the substring checks could not make.

    A judge given a question that does not say so would plausibly repeat the
    mistake the question exists to fix.
    """
    for question in STRANDED["negative-constraint"] + STRANDED["failed-approach"]:
        text = question.question.lower()
        assert "only to" in text, (
            f"{question.question_id} must tell the judge that naming a rejected "
            "option is not a violation"
        )


def test_a_judged_question_without_a_judge_is_not_a_failure():
    """The property that makes this repair honest rather than cosmetic.

    ``JudgedQuestion`` reports not-evaluated when no judge is configured. If it
    defaulted to passed or failed, moving the assertions here would have traded
    one wrong answer for another.
    """
    from open_context_eval.runner import EvaluationConfig

    assert EvaluationConfig(provider=None, tokenizer=None).judge is None


# ----------------------------------------------------------------------
# What it cost


def test_two_scenarios_now_have_no_deterministic_evidence():
    """Recorded rather than hidden.

    ``negative-constraint`` and ``failed-approach`` state a prohibition and never
    say what to do instead, so there is no positive assertion to make: the space
    of acceptable answers is open. They were only ever measurable by the
    mechanism that turned out to be broken. Without a judge they now measure
    nothing, which is the honest reading and not a regression to fix by
    inventing an expected answer.
    """
    bare = [s.scenario_id for s in V3 if not s.retention_checks and not s.completion]
    assert sorted(bare) == ["failed-approach", "negative-constraint"]
    for scenario_id in bare:
        assert by_id(scenario_id).judged, "they still state what they care about"


def test_the_dataset_still_covers_every_scenario():
    assert len(V3) == len(V2) == 15


def test_by_id_rejects_an_unknown_scenario():
    import pytest

    with pytest.raises(KeyError):
        by_id("no-such-scenario")


# ----------------------------------------------------------------------
# Polarity


def test_yes_is_always_the_desirable_answer():
    """The harness records `passed` straight from the verdict, so polarity matters.

    A question asking "does the response recommend Redis?" scores a correct
    answer — one recommending nothing of the sort — as a failure, because the
    judge correctly answers NO. An earlier draft did exactly that, and a real
    judge caught it: `passed=False`, reasoning "the response does not mention
    Redis at all, let alone recommend using it."

    Enforceable form of the rule: any question that acknowledges the
    naming-in-order-to-reject case must map that case to YES.
    """
    for scenario in V3:
        for question in scenario.judged:
            if "only to" in question.question.lower():
                assert "answer yes if" in question.question.lower(), (
                    f"{question.question_id} maps the rejection case to NO, so a correct "
                    "answer scores as a failure"
                )


def test_the_stranded_questions_ask_whether_the_thing_was_avoided():
    """Phrased as avoidance, not as commission."""
    for questions in STRANDED.values():
        for question in questions:
            assert "avoid" in question.question.lower(), question.question_id
