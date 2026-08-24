"""Reading a benchmark.

Plain text. No plotting, no HTML, no composite score.

**Retention and completion are never collapsed into one number.** They answer
different questions — what survived, and whether the agent acted correctly — and
a strategy can be strong on one and weak on the other. That difference is the
most useful thing the benchmark produces, and averaging it away would destroy
it. A composite score can be defined later, once there is evidence about what it
should weigh.

**Every number here comes from a run.** Nothing is hard-coded, and a report of
deterministic-mode results says at the top that it measures the harness rather
than any strategy.

**Failures are listed, not averaged.** "What did this strategy lose, on which
scenario" is what a benchmark is for; a single score cannot answer it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from open_context_eval.benchmark import DETERMINISTIC_TEST, REFERENCE_BUDGET
from open_context_eval.results import EvaluationResult, RunStatus

REFERENCE_LABEL = "ref"
"""How a budget-independent arm's budget column reads.

A reference condition is not held to a budget. Truncating it to match the
compacted arms would make the reference a different, unnamed strategy, and every
comparison against it would flatter the others.

The benchmark runs such an arm once per scenario and repetition, recording
``target_tokens = REFERENCE_BUDGET``, so grouping here is ordinary grouping by
budget and needs no special case for any strategy by name.
"""


@dataclass(frozen=True)
class Cell:
    """One strategy at one budget, aggregated over scenarios and repetitions."""

    strategy: str
    budget: int
    runs: int
    succeeded: int
    disowned: int
    """Cells that ran, returned a context, and were disowned by their strategy.

    Excluded from every score above. Reported rather than dropped: a strategy
    that disowns most of its own cells has said something important about the
    model it ran against, and a row that simply showed fewer runs would hide it.
    """

    failures: dict[str, int]

    retention_passed: int
    retention_total: int

    completed: int
    completion_evaluated: int
    not_evaluable: int

    fabrications: int
    mean_context_tokens: float | None
    mean_compression: float | None
    mean_total_tokens: float | None
    mean_compaction_calls: float | None
    mean_latency: float | None

    @property
    def retention_score(self) -> float | None:
        return self.retention_passed / self.retention_total if self.retention_total else None

    @property
    def completion_score(self) -> float | None:
        """Fraction of scenarios whose task was completed.

        Over scenarios that could be evaluated. ``None`` when none could, which
        is different from zero.
        """
        return self.completed / self.completion_evaluated if self.completion_evaluated else None


def summarize_benchmark(results: Iterable[EvaluationResult]) -> list[Cell]:
    """One row per strategy and budget, in a stable order."""
    grouped: dict[tuple[str, int], list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault((result.strategy, result.run.target_tokens), []).append(result)

    cells = []
    for (strategy, budget), runs in sorted(grouped.items()):
        # A cell the strategy disowned is not a measurement of that strategy.
        # It is excluded from every score and counted separately, because
        # averaging it in reports the fallback under the arm's name — which is
        # exactly what happened to hybrid compaction in Phase 8.5, where 21 of
        # 24 cells were the Phase 5 baseline wearing hybrid's label.
        ok = [run for run in runs if run.measures_its_strategy]
        disowned = sum(1 for run in runs if run.succeeded and run.strategy_disowned)
        failures: dict[str, int] = {}
        for run in runs:
            if run.status is not RunStatus.SUCCESS:
                failures[run.status.value] = failures.get(run.status.value, 0) + 1

        judged = [run.task_completed for run in ok]
        evaluated = [verdict for verdict in judged if verdict is not None]
        compressions = [r.compression_ratio for r in ok if r.compression_ratio is not None]
        totals = [r.total_tokens for r in ok if r.total_tokens is not None]
        latencies = [r.latency_seconds for r in ok if r.latency_seconds is not None]

        cells.append(
            Cell(
                strategy=strategy,
                budget=budget,
                runs=len(runs),
                succeeded=len(ok),
                disowned=disowned,
                failures=failures,
                retention_passed=sum(
                    1 for run in ok for check in run.retention_checks if check.passed
                ),
                retention_total=sum(len(run.retention_checks) for run in ok),
                completed=sum(1 for verdict in evaluated if verdict),
                completion_evaluated=len(evaluated),
                not_evaluable=sum(1 for verdict in judged if verdict is None),
                fabrications=sum(run.deterministic_hallucination_checks for run in ok),
                mean_context_tokens=_mean([r.context_tokens for r in ok]),
                mean_compression=_mean(compressions),
                mean_total_tokens=_mean(totals),
                mean_compaction_calls=_mean([r.compaction_llm_calls for r in ok]),
                mean_latency=_mean(latencies),
            )
        )
    return cells


def format_benchmark(results: Sequence[EvaluationResult]) -> str:
    """The headline table."""
    cells = summarize_benchmark(results)
    if not cells:
        return "no results"

    lines: list[str] = []
    modes = {result.evaluation_mode for result in results}
    if modes == {DETERMINISTIC_TEST}:
        lines += [
            "EVALUATION MODE: deterministic_test",
            "",
            "These numbers came from a fake provider that returns a fixed string. They",
            "validate the benchmark machinery and say nothing about any strategy's",
            "quality. Real-model numbers require a configured provider.",
            "",
        ]
    else:
        lines += [f"EVALUATION MODE: {', '.join(sorted(modes))}", ""]

    header = (
        f"{'Strategy':<22}{'Budget':>8}{'Runs':>6}{'Retention':>11}"
        f"{'Completion':>12}{'Context':>9}{'Compr':>8}{'Total tok':>11}{'Calls':>7}"
    )
    lines += [header, "-" * len(header)]
    for cell in cells:
        budget = REFERENCE_LABEL if cell.budget == REFERENCE_BUDGET else str(cell.budget)
        lines.append(
            f"{cell.strategy:<22}{budget:>8}{cell.succeeded:>6}"
            f"{_ratio(cell.retention_score):>11}{_ratio(cell.completion_score):>12}"
            f"{_number(cell.mean_context_tokens, '{:.0f}'):>9}"
            f"{_number(cell.mean_compression, '{:.1f}x'):>8}"
            f"{_number(cell.mean_total_tokens, '{:.0f}'):>11}"
            f"{_number(cell.mean_compaction_calls, '{:.1f}'):>7}"
        )

    lines += [
        "",
        f"{REFERENCE_LABEL}: a budget-independent reference condition, run once per scenario",
        "    and repetition. Every budget row is compared against the same reference.",
        "Retention: expected information that survived into the response.",
    ]
    lines.append("Completion: scenarios where every task criterion was met. Not the same measure.")

    fabrications = sum(cell.fabrications for cell in cells)
    lines.append(
        f"Deterministic hallucination checks: {fabrications} responses used something the "
        "conversation had ruled out."
    )

    unevaluable = sum(cell.not_evaluable for cell in cells)
    if unevaluable:
        lines.append(
            f"{unevaluable} runs could not be scored for completion and are excluded from it, "
            "not counted as failures."
        )

    disowned = [
        f"  {cell.strategy} @ {cell.budget}: {cell.disowned} of "
        f"{cell.disowned + cell.succeeded} cells"
        for cell in cells
        if cell.disowned
    ]
    if disowned:
        lines += [
            "",
            "Cells the strategy disowned, excluded from every score above:",
            *disowned,
            "    The strategy ran, returned a context, and said it was not what its name",
            "    claims — a composite whose inner step failed and fell back. Counting these",
            "    would report the fallback under the arm's name.",
        ]

    warning = _judge_calibration(results)
    if warning:
        lines += ["", warning]

    problems = [
        f"  {cell.strategy} @ {cell.budget}: {status} x{count}"
        for cell in cells
        for status, count in sorted(cell.failures.items())
    ]
    if problems:
        lines += ["", "Failures (not counted as zero scores):", *problems]
    return "\n".join(lines)


JUDGE_CALIBRATION_FLOOR = 0.7
"""Below this, on the reference arm, the judge is the thing being measured."""


def _judge_calibration(results: Sequence[EvaluationResult]) -> str:
    """Check the judge against the arm that was given everything.

    The reference arm runs at ``REFERENCE_BUDGET`` — the whole conversation,
    uncompacted. When its deterministic checks pass and its judged questions do
    not, the judge is failing answers that demonstrably contain what was asked
    for, and a judge that cannot grade the control cannot grade anything below
    it.

    Phase 8.5 measured exactly that. On identical questions and the same
    reference arm, an 8B judge passed **9 of 24** where a 70B judge passed
    **8 of 8**, with deterministic retention at 100% in both. Without this
    check, that arrives as a compaction result.
    """
    reference = [r for r in results if r.run.target_tokens == REFERENCE_BUDGET and r.succeeded]
    verdicts = [v for r in reference for v in r.judged if v.evaluated]
    if not verdicts:
        return ""
    deterministic = [c.passed for r in reference for c in r.retention_checks]
    if not deterministic or sum(deterministic) / len(deterministic) < 1.0:
        return ""

    passed = sum(1 for v in verdicts if v.passed)
    score = passed / len(verdicts)
    if score >= JUDGE_CALIBRATION_FLOOR:
        return ""
    return (
        f"JUDGE UNRELIABLE: the reference arm passed every deterministic check but only "
        f"{passed}/{len(verdicts)} judged questions ({score:.0%}).\n"
        f"    That arm was given the whole conversation uncompacted, so the information "
        f"is\n    demonstrably there. The judge is what is being measured. Treat every "
        f"judged\n    score in this run as void and compare on deterministic checks, or "
        f"rejudge with\n    a stronger model."
    )


def format_failures(results: Sequence[EvaluationResult], *, limit: int = 40) -> str:
    """What each strategy actually lost, scenario by scenario.

    The point of the benchmark. A score says a strategy was worse; this says
    which check it failed on which scenario, which is what a next phase can act
    on.
    """
    lines: list[str] = []
    shown = 0
    for result in results:
        if not result.succeeded:
            lines.append(
                f"{result.scenario_id:<28}{result.strategy:<22}status={result.status.value}"
            )
            shown += 1
            continue

        failed_checks = [check for check in result.retention_checks if not check.passed]
        failed_criteria = [
            outcome for outcome in result.completion if outcome.status not in {"passed"}
        ]
        if not failed_checks and not failed_criteria:
            continue

        lines.append(
            f"{result.scenario_id:<28}{result.strategy:<22}budget={result.run.target_tokens}"
        )
        for check in failed_checks:
            lines.append(f"    retention  {check.check_id}: {'; '.join(check.problems)}")
        for outcome in failed_criteria:
            lines.append(
                f"    completion {outcome.criterion_id}: {outcome.status} — {outcome.detail}"
            )
        shown += 1
        if shown >= limit:
            lines.append(f"... more failures not shown (limit {limit})")
            break

    return "\n".join(lines) if lines else "no failures"


def failure_modes_table(results: Sequence[EvaluationResult]) -> str:
    """Which kinds of forgetting cost the most, per strategy.

    Groups completion failures by the scenario's declared failure mode, which is
    the closest this benchmark comes to saying *why* a strategy lost.
    """
    counts: dict[tuple[str, str], list[int]] = {}
    for result in results:
        if not result.succeeded:
            continue
        verdict = result.task_completed
        if verdict is None:
            continue
        for mode in result.failure_modes:
            counts.setdefault((result.strategy, mode), []).append(1 if verdict else 0)

    if not counts:
        return "no completion data"

    lines = [f"{'Strategy':<22}{'Failure mode':<28}{'Completed':>10}"]
    lines.append("-" * len(lines[0]))
    for (strategy, mode), verdicts in sorted(counts.items()):
        lines.append(f"{strategy:<22}{mode:<28}{sum(verdicts)}/{len(verdicts):<10}".rstrip())
    return "\n".join(lines)


def _mean(values: Sequence[float] | list[int] | list[float]) -> float | None:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else None


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _number(value: float | None, spec: str) -> str:
    return "-" if value is None else spec.format(value)


__all__ = [
    "REFERENCE_LABEL",
    "Cell",
    "failure_modes_table",
    "format_benchmark",
    "format_failures",
    "summarize_benchmark",
]
