"""Two checks the benchmark grew because a real run needed them.

Both correspond to a Phase 8.5 run that produced a readable, plausible, wrong
answer. Neither failure was visible in any score — one was a warning nobody
aggregated, the other a control nobody checked — so both are now enforced where
the numbers are produced rather than left to whoever reads them.
"""

from open_context_eval.benchmark import REFERENCE_BUDGET
from open_context_eval.benchmark_report import format_benchmark, summarize_benchmark
from open_context_eval.results import (
    EvaluationResult,
    JudgedOutcome,
    RetentionOutcome,
    RunMetadata,
    RunStatus,
)

DISOWNED = (
    "extraction produced no usable state; this result is the Phase 5 baseline with an "
    "empty state section, and should not be read as hybrid compaction"
)


def run(budget: int) -> RunMetadata:
    return RunMetadata(
        run_id=f"run-{budget}",
        dataset_version="v4",
        provider="fake",
        model="fake-model",
        continuation_prompt_id="continuation_v1",
        continuation_prompt_hash="0" * 16,
        target_tokens=budget,
    )


def cell(
    *,
    strategy: str = "hybrid_v1",
    budget: int = 160,
    scenario: str = "adv-exact-values",
    retention: bool = True,
    judged: bool | None = None,
    warnings: tuple[str, ...] = (),
) -> EvaluationResult:
    return EvaluationResult(
        run=run(budget),
        scenario_id=scenario,
        task_id=f"{scenario}-task",
        task_kind="code_continuation",
        strategy=strategy,
        strategy_version=1,
        status=RunStatus.SUCCESS,
        retention_checks=(
            RetentionOutcome(check_id="c", category="critical_fact_retention", passed=retention),
        ),
        judged=(
            ()
            if judged is None
            else (
                JudgedOutcome(
                    question_id="q",
                    category="critical_fact_retention",
                    evaluated=True,
                    passed=judged,
                ),
            )
        ),
        strategy_detail={"warnings": list(warnings)},
    )


# ----------------------------------------------------------------------
# A cell the strategy disowned is not a measurement of that strategy


def test_a_disowned_cell_is_excluded_from_its_arm_score():
    """Phase 8.5 ran 24 hybrid cells against an 8B model and 21 were not hybrid.

    Extraction produced no state, the compactor fell back to the Phase 5
    baseline and said so, and the arm's headline score was the fallback wearing
    hybrid's name. The warning was in the results file the whole time.
    """
    results = [
        cell(retention=True),
        cell(retention=False, scenario="adv-tool-heavy", warnings=(DISOWNED,)),
    ]
    (row,) = summarize_benchmark(results)
    assert row.succeeded == 1
    assert row.disowned == 1
    assert row.retention_score == 1.0, "the disowned cell must not drag the score down"


def test_a_disowned_cell_is_reported_rather_than_dropped():
    """A strategy disowning most of its own cells has said something important.

    A row that simply showed fewer runs would hide it.
    """
    text = format_benchmark([cell(warnings=(DISOWNED,)), cell()])
    assert "disowned" in text.lower()
    assert "1 of 2 cells" in text


def test_an_ordinary_warning_does_not_disown_a_cell():
    """Only a strategy saying the result is not what its name claims counts.

    Estimated token counts and an unknown context window are warnings about the
    measurement, not disavowals of it.
    """
    ordinary = ("token counts are estimated, not exact",)
    (row,) = summarize_benchmark([cell(warnings=ordinary)])
    assert row.disowned == 0
    assert row.succeeded == 1


def test_a_cell_with_no_warnings_at_all_is_kept():
    (row,) = summarize_benchmark([cell()])
    assert row.disowned == 0
    assert row.succeeded == 1


# ----------------------------------------------------------------------
# The reference arm is the judge's control


def test_a_judge_that_fails_the_reference_arm_is_reported_as_unreliable():
    """The reference arm gets the whole conversation, uncompacted.

    When its deterministic checks all pass and its judged questions do not, the
    judge is failing answers that demonstrably contain what was asked for. On
    identical questions, an 8B judge passed 9 of 24 here where a 70B judge
    passed 8 of 8 — and without this check that arrives as a compaction result.
    """
    results = [
        cell(
            strategy="full_context",
            budget=REFERENCE_BUDGET,
            scenario=f"s{i}",
            retention=True,
            judged=False,
        )
        for i in range(4)
    ]
    text = format_benchmark(results)
    assert "JUDGE UNRELIABLE" in text
    assert "0/4" in text
    assert "deterministic" in text


def test_a_judge_that_agrees_with_the_reference_arm_is_not_flagged():
    results = [
        cell(
            strategy="full_context",
            budget=REFERENCE_BUDGET,
            scenario=f"s{i}",
            retention=True,
            judged=True,
        )
        for i in range(4)
    ]
    assert "JUDGE UNRELIABLE" not in format_benchmark(results)


def test_the_check_stays_quiet_when_the_reference_arm_lost_information_itself():
    """A failing deterministic check means the response really is missing something.

    The judge and the answer then disagree for a reason the judge may be right
    about, and calling it unreliable would be the same unearned conclusion in
    the other direction.
    """
    results = [
        cell(
            strategy="full_context",
            budget=REFERENCE_BUDGET,
            scenario=f"s{i}",
            retention=False,
            judged=False,
        )
        for i in range(4)
    ]
    assert "JUDGE UNRELIABLE" not in format_benchmark(results)


def test_the_check_stays_quiet_when_nothing_was_judged():
    """A run without `--judge` has no judge to calibrate."""
    results = [
        cell(strategy="full_context", budget=REFERENCE_BUDGET, scenario=f"s{i}", retention=True)
        for i in range(4)
    ]
    assert "JUDGE UNRELIABLE" not in format_benchmark(results)


def test_the_hybrid_compactor_still_uses_the_wording_the_check_matches():
    """Two files agree on a phrase, so a test holds them together.

    If the compactor rewords its warning, the exclusion silently stops working
    and hybrid's fallback cells rejoin its score.
    """
    import inspect

    from open_context.hybrid.compactor import HybridCompaction
    from open_context_eval.results import _DISOWNED

    source = inspect.getsource(HybridCompaction)
    assert _DISOWNED in source, "the compactor no longer disowns cells in the expected wording"
