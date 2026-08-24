"""Classifying observed failures into the fifteen roadmap classes.

The tests that matter here are the ones about what the taxonomy refuses to say:
that a class nothing can detect reports as unmeasured rather than zero, that a
run which never produced a response contributes no retention failure, and that
evidence the mapping cannot label is counted rather than dropped.
"""

from open_context_eval.results import (
    CompletionOutcome,
    EvaluationResult,
    RetentionOutcome,
    RunMetadata,
    RunStatus,
)
from open_context_eval.scenario import FailureMode, MetricCategory
from open_context_eval.taxonomy import (
    UNMEASURED,
    FailureClass,
    classify,
    classify_result,
    counts_by_class,
    format_taxonomy,
)


def metadata(*, target_tokens: int = 160) -> RunMetadata:
    return RunMetadata(
        run_id="run-test",
        dataset_version="v2",
        provider="fake",
        model="fake-model",
        continuation_prompt_id="continuation_v1",
        continuation_prompt_hash="abc123",
        target_tokens=target_tokens,
    )


def result(
    *,
    checks=(),
    completion=(),
    modes=(),
    status=RunStatus.SUCCESS,
    strategy="phase_5_baseline",
    context_tokens=100,
    target_tokens=160,
) -> EvaluationResult:
    return EvaluationResult(
        run=metadata(target_tokens=target_tokens),
        scenario_id="scenario-1",
        failure_modes=tuple(mode.value for mode in modes),
        task_id="task-1",
        task_kind="answer_question",
        strategy=strategy,
        strategy_version=1,
        status=status,
        response="a response",
        original_tokens=400,
        context_tokens=context_tokens,
        evaluation_mode="deterministic_test",
        retention_checks=tuple(checks),
        completion=tuple(completion),
    )


def failed_check(category: MetricCategory, check_id: str = "check-1") -> RetentionOutcome:
    return RetentionOutcome(check_id=check_id, category=category.value, passed=False)


def passed_check(category: MetricCategory, check_id: str = "check-1") -> RetentionOutcome:
    return RetentionOutcome(check_id=check_id, category=category.value, passed=True)


# ----------------------------------------------------------------------
# The mapping


def test_a_failed_constraint_check_is_a_lost_constraint():
    observations = classify_result(result(checks=[failed_check(MetricCategory.CONSTRAINT)]))
    assert [item.failure_class for item in observations] == [FailureClass.LOST_CONSTRAINT]


def test_a_negative_constraint_scenario_refines_the_class():
    """The distinction the roadmap draws and the category cannot make.

    Both are written as constraints, so only the scenario's trap says which one
    was lost.
    """
    observations = classify_result(
        result(
            checks=[failed_check(MetricCategory.CONSTRAINT)],
            modes=[FailureMode.NEGATIVE_CONSTRAINT],
        )
    )
    assert [item.failure_class for item in observations] == [FailureClass.LOST_NEGATIVE_CONSTRAINT]


def test_an_exact_value_scenario_refines_a_critical_fact():
    observations = classify_result(
        result(checks=[failed_check(MetricCategory.CRITICAL_FACT)], modes=[FailureMode.EXACT_VALUE])
    )
    assert [item.failure_class for item in observations] == [FailureClass.EXACT_VALUE_LOSS]


def test_categories_that_map_directly():
    cases = {
        MetricCategory.DECISION: FailureClass.LOST_DECISION,
        MetricCategory.RATIONALE: FailureClass.LOST_RATIONALE,
        MetricCategory.TEMPORAL_STATE: FailureClass.TEMPORAL_CONFUSION,
        MetricCategory.OPEN_TASK: FailureClass.LOST_INCOMPLETE_WORK,
        MetricCategory.FAILED_APPROACH: FailureClass.LOST_FAILED_APPROACH,
        MetricCategory.ENTITY: FailureClass.ENTITY_CONFUSION,
    }
    for category, expected in cases.items():
        observations = classify_result(result(checks=[failed_check(category)]))
        assert [item.failure_class for item in observations] == [expected], category


def test_a_passing_check_contributes_nothing():
    assert classify_result(result(checks=[passed_check(MetricCategory.CONSTRAINT)])) == []


def test_an_unmappable_category_is_counted_as_unclassified_not_dropped():
    """A gap in the mapping must be visible in the totals.

    Dropping it would make the per-class counts disagree with the number of
    failures, and the disagreement is the only signal that the mapping is
    incomplete.
    """
    observations = classify_result(
        result(checks=[RetentionOutcome(check_id="x", category="something_new", passed=False)])
    )
    assert [item.failure_class for item in observations] == [FailureClass.UNCLASSIFIED]


def test_classification_does_not_depend_on_mode_ordering():
    """Two scenarios with the same traps in a different order classify alike."""
    modes = [FailureMode.EXACT_VALUE, FailureMode.CURRENT_STATE]
    first = classify_result(
        result(checks=[failed_check(MetricCategory.CRITICAL_FACT)], modes=modes)
    )
    second = classify_result(
        result(checks=[failed_check(MetricCategory.CRITICAL_FACT)], modes=list(reversed(modes)))
    )
    assert [item.failure_class for item in first] == [item.failure_class for item in second]


# ----------------------------------------------------------------------
# Hallucination and budget


def test_a_fabricated_completion_is_a_hallucination():
    observations = classify_result(
        result(
            completion=[
                CompletionOutcome(
                    criterion_id="c1", kind="names_value", status="fabricated", detail="used Redis"
                )
            ]
        )
    )
    assert [item.failure_class for item in observations] == [FailureClass.HALLUCINATION]
    assert "used Redis" in observations[0].evidence


def test_an_omitted_completion_is_not_a_hallucination():
    """Forgetting and asserting have different causes and different classes."""
    observations = classify_result(
        result(
            completion=[CompletionOutcome(criterion_id="c1", kind="names_value", status="omitted")]
        )
    )
    assert observations == []


def test_a_context_over_its_target_is_a_budget_overflow():
    observations = classify_result(result(context_tokens=200, target_tokens=160))
    assert [item.failure_class for item in observations] == [FailureClass.BUDGET_OVERFLOW]
    assert "200 tokens against a target of 160" in observations[0].evidence


def test_a_context_within_its_target_is_not_an_overflow():
    assert classify_result(result(context_tokens=160, target_tokens=160)) == []


def test_a_reference_arm_never_overflows():
    """It consulted no budget, so there is no target to have exceeded.

    Reference results record ``target_tokens`` as zero. Comparing a whole
    conversation against zero would put a budget failure on every reference row
    and make the class meaningless.
    """
    observations = classify_result(
        result(strategy="full_context", context_tokens=4_000, target_tokens=0)
    )
    assert observations == []


# ----------------------------------------------------------------------
# What it refuses to say


def test_a_failed_run_contributes_no_retention_failure():
    """An outage is not a finding about a strategy.

    A rate-limited run produced no response, so it exhibits no forgetting. If it
    counted, a provider having a bad afternoon would read as a strategy that
    loses constraints.
    """
    observations = classify_result(
        result(checks=[failed_check(MetricCategory.CONSTRAINT)], status=RunStatus.MODEL_ERROR)
    )
    assert observations == []


def test_the_unmeasured_classes_are_the_ones_with_no_evidence():
    assert set(UNMEASURED) == {
        FailureClass.LOST_GOAL,
        FailureClass.LOST_COMPLETED_WORK,
        FailureClass.IRRELEVANT_CONTEXT_RETENTION,
    }


def test_every_unmeasured_class_says_what_would_measure_it():
    for failure_class, reason in UNMEASURED.items():
        assert len(reason) > 40, f"{failure_class} needs a real explanation"
        assert "closing this" in reason.lower(), f"{failure_class} must say what would fix it"


def test_the_report_never_prints_zero_for_an_unmeasured_class():
    """The single claim this module most needs not to make.

    A zero beside "lost goal" asserts that no goal was lost. Nothing in the
    harness can establish that, and a strategy that dropped every goal in the
    dataset would produce the same zero.
    """
    report = format_taxonomy([result()], strategies=["phase_5_baseline"])
    for failure_class in UNMEASURED:
        line = next(
            (row for row in report.splitlines() if row.strip().startswith(failure_class.value)),
            None,
        )
        assert line is not None, f"{failure_class} is missing from the report entirely"
        assert ": " in line, f"{failure_class} is reported as a count rather than a reason"


def test_the_report_lists_every_unmeasured_class_with_its_reason():
    report = format_taxonomy([result()], strategies=["phase_5_baseline"])
    assert "Declared and not measured" in report
    for failure_class in UNMEASURED:
        assert failure_class.value in report


def test_the_taxonomy_covers_the_fifteen_roadmap_classes():
    """The roadmap names fifteen; ``UNCLASSIFIED`` is ours, for accounting."""
    named = [item for item in FailureClass if item is not FailureClass.UNCLASSIFIED]
    assert len(named) == 15


# ----------------------------------------------------------------------
# Aggregation


def test_counts_are_grouped_by_class_and_strategy():
    results = [
        result(checks=[failed_check(MetricCategory.DECISION)], strategy="phase_5_baseline"),
        result(checks=[failed_check(MetricCategory.DECISION)], strategy="phase_5_baseline"),
        result(checks=[failed_check(MetricCategory.DECISION)], strategy="simple_summary_v1"),
    ]
    table = counts_by_class(classify(results))
    assert table[FailureClass.LOST_DECISION]["phase_5_baseline"] == 2
    assert table[FailureClass.LOST_DECISION]["simple_summary_v1"] == 1


def test_a_scenario_with_several_failed_checks_contributes_several():
    observations = classify_result(
        result(
            checks=[
                failed_check(MetricCategory.DECISION, "c1"),
                failed_check(MetricCategory.RATIONALE, "c2"),
            ]
        )
    )
    assert len(observations) == 2


def test_every_observation_names_the_evidence_it_came_from():
    observations = classify_result(result(checks=[failed_check(MetricCategory.DECISION, "c7")]))
    assert "c7" in observations[0].evidence
    assert observations[0].scenario_id == "scenario-1"
    assert observations[0].strategy == "phase_5_baseline"
