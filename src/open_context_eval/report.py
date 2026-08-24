"""Turning results into something readable.

Plain text tables and a structured summary. No plotting, no dashboard, no HTML.

**A column is shown only when it was measured.** Quality columns appear only for
runs that actually evaluated checks, and a failed run shows its status rather
than a zero. A report that printed 0.00 for a run that never reached the model
would read as a strategy that failed the task.

The aggregate view is deliberately two numbers side by side — token cost and
check score — because that is the trade the project is about, and a single
combined score would hide it. Anything that wants to plot quality against cost
can read ``pareto_points`` and do so; nothing here draws it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from open_context_eval.results import EvaluationResult, RunStatus


@dataclass(frozen=True)
class StrategySummary:
    """Aggregate for one strategy across the scenarios it was run on."""

    strategy: str
    runs: int
    succeeded: int
    failures: dict[str, int]
    mean_context_tokens: float | None
    mean_compression: float | None
    retention_checks_passed: int
    retention_checks_total: int
    mean_latency: float | None

    @property
    def retention_score(self) -> float | None:
        """Fraction of deterministic checks passed, or None if none were run."""
        return (
            self.retention_checks_passed / self.retention_checks_total
            if self.retention_checks_total
            else None
        )


def summarize(results: Iterable[EvaluationResult]) -> list[StrategySummary]:
    """One row per strategy, in name order."""
    grouped: dict[str, list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault(result.strategy, []).append(result)

    summaries = []
    for strategy, runs in sorted(grouped.items()):
        ok = [run for run in runs if run.succeeded]
        failures: dict[str, int] = {}
        for run in runs:
            if run.status is not RunStatus.SUCCESS:
                failures[run.status.value] = failures.get(run.status.value, 0) + 1

        compressions = [run.compression_ratio for run in ok if run.compression_ratio is not None]
        latencies = [run.latency_seconds for run in ok if run.latency_seconds is not None]
        summaries.append(
            StrategySummary(
                strategy=strategy,
                runs=len(runs),
                succeeded=len(ok),
                failures=failures,
                mean_context_tokens=_mean([run.context_tokens for run in ok]),
                mean_compression=_mean(compressions),
                retention_checks_passed=sum(
                    1 for run in ok for check in run.retention_checks if check.passed
                ),
                retention_checks_total=sum(len(run.retention_checks) for run in ok),
                mean_latency=_mean(latencies),
            )
        )
    return summaries


def pareto_points(results: Iterable[EvaluationResult]) -> list[dict[str, float | str | None]]:
    """Cost against quality, as data rather than a picture.

    One point per strategy. Whatever wants to plot it can; this refuses to
    decide that fewer tokens at a lower score is better or worse, because that
    is the question the benchmark exists to inform.
    """
    return [
        {
            "strategy": summary.strategy,
            "mean_context_tokens": summary.mean_context_tokens,
            "retention_score": summary.retention_score,
            "runs": summary.succeeded,
        }
        for summary in summarize(results)
    ]


def format_table(results: Sequence[EvaluationResult]) -> str:
    """A comparison table. Only measured columns carry numbers."""
    summaries = summarize(results)
    if not summaries:
        return "no results"

    header = f"{'Strategy':<22}{'Runs':>6}{'Tokens':>10}{'Compression':>13}{'Checks':>12}"
    lines = [header, "-" * len(header)]
    for summary in summaries:
        tokens = _number(summary.mean_context_tokens, "{:.0f}")
        compression = _number(summary.mean_compression, "{:.1f}x")
        checks = (
            f"{summary.retention_checks_passed}/{summary.retention_checks_total}"
            if summary.retention_checks_total
            else "-"
        )
        lines.append(
            f"{summary.strategy:<22}{summary.succeeded:>6}{tokens:>10}{compression:>13}{checks:>12}"
        )

    problems = [
        f"  {summary.strategy}: {status} x{count}"
        for summary in summaries
        for status, count in sorted(summary.failures.items())
    ]
    if problems:
        lines.extend(["", "Failures (not counted as zero scores):", *problems])
    return "\n".join(lines)


def format_scenarios(results: Sequence[EvaluationResult]) -> str:
    """Per scenario and strategy, for reading what actually happened."""
    if not results:
        return "no results"
    lines = []
    for result in results:
        if result.succeeded:
            score = result.retention_score
            detail = "no checks" if score is None else f"{score:.0%} of checks"
            failed = [check.check_id for check in result.retention_checks if not check.passed]
            suffix = f", failed: {', '.join(failed)}" if failed else ""
            lines.append(f"{result.scenario_id:<28}{result.strategy:<22}{detail}{suffix}")
        else:
            lines.append(
                f"{result.scenario_id:<28}{result.strategy or '-':<22}{result.status.value}"
            )
    return "\n".join(lines)


def _mean(values: Sequence[float] | list[int] | list[float]) -> float | None:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else None


def _number(value: float | None, spec: str) -> str:
    return "-" if value is None else spec.format(value)


__all__ = [
    "StrategySummary",
    "format_scenarios",
    "format_table",
    "pareto_points",
    "summarize",
]
