"""What went wrong, classified.

```
result -> failed checks + fabrications + budget -> failure classes
```

A benchmark that reports "the baseline scored 0.62" says nothing a next phase
can act on. The number that matters is *which kind of thing* each strategy drops,
because that is what an algorithm gets designed against. So every observed
failure is classified into one of the classes below, and the report groups by
class rather than by scenario.

**A class is a property of an observed failure, not of a scenario.**
``scenario.FailureMode`` says which trap a scenario sets — it is fixed when the
dataset is written and is the same whether every arm passes or every arm fails.
A class here says what actually happened in one run. The two are related: a
scenario's mode is used to disambiguate a failed check whose category alone is
not specific enough, which is why they appear together in the mapping below and
never as substitutes for each other.

**Three classes are declared and cannot be measured, and they report as
unmeasured rather than as zero.** Nothing in the current instrumentation
detects a lost goal, lost completed work, or irrelevant context that survived
compaction. A taxonomy that printed ``0`` beside them would be asserting those
failures did not happen, which is a claim about the strategies rather than
about the harness — and the wrong one, since a strategy that dropped every goal
in the dataset would produce exactly the same zero. They are listed with a
reason, and closing each is a dataset or judge change rather than a reporting
one.

**Every failed check lands somewhere.** A check whose category and scenario do
not determine a class is counted as ``unclassified`` rather than dropped, so the
per-class counts and the total number of failures agree. A taxonomy that
silently discards the evidence it cannot label is how a gap in the mapping stays
invisible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from open_context_eval.results import EvaluationResult
from open_context_eval.scenario import FailureMode, MetricCategory


class FailureClass(StrEnum):
    """One kind of thing a context representation can lose or invent.

    The fifteen classes the roadmap names for the first real benchmark, plus
    ``UNCLASSIFIED`` so the accounting closes.
    """

    LOST_GOAL = "lost_goal"
    LOST_CONSTRAINT = "lost_constraint"
    LOST_NEGATIVE_CONSTRAINT = "lost_negative_constraint"
    LOST_DECISION = "lost_decision"
    LOST_RATIONALE = "lost_rationale"
    LOST_CURRENT_STATE = "lost_current_state"
    LOST_COMPLETED_WORK = "lost_completed_work"
    LOST_INCOMPLETE_WORK = "lost_incomplete_work"
    LOST_FAILED_APPROACH = "lost_failed_approach"
    ENTITY_CONFUSION = "entity_confusion"
    TEMPORAL_CONFUSION = "temporal_confusion"
    EXACT_VALUE_LOSS = "exact_value_loss"
    HALLUCINATION = "hallucination"
    IRRELEVANT_CONTEXT_RETENTION = "irrelevant_context_retention"
    BUDGET_OVERFLOW = "budget_overflow"

    UNCLASSIFIED = "unclassified"
    """A real failure the mapping could not name. Never silently dropped."""


UNMEASURED: dict[FailureClass, str] = {
    FailureClass.LOST_GOAL: (
        "no retention check carries a goal category, so a dropped goal is invisible to the "
        "deterministic checks. Closing this is a dataset change: scenarios need an explicit "
        "goal statement and a check for it"
    ),
    FailureClass.LOST_COMPLETED_WORK: (
        "the dataset checks open tasks but not finished ones, so a representation that forgot "
        "what was already done produces no failed check. Closing this needs scenarios whose "
        "task depends on not redoing completed work"
    ),
    FailureClass.IRRELEVANT_CONTEXT_RETENTION: (
        "every check asks whether something survived; none asks whether something that should "
        "have been dropped survived instead. A context full of irrelevant material scores "
        "identically to a focused one of the same size. Closing this needs either a judge or "
        "checks that name what must be absent"
    ),
}
"""Classes nothing currently detects, and what it would take to detect them.

Reported explicitly. A class absent from a report reads as a class that did not
occur, and neither of those is what "we cannot see this" means.
"""

_BY_CATEGORY: dict[MetricCategory, FailureClass] = {
    MetricCategory.CONSTRAINT: FailureClass.LOST_CONSTRAINT,
    MetricCategory.DECISION: FailureClass.LOST_DECISION,
    MetricCategory.RATIONALE: FailureClass.LOST_RATIONALE,
    MetricCategory.TEMPORAL_STATE: FailureClass.TEMPORAL_CONFUSION,
    MetricCategory.OPEN_TASK: FailureClass.LOST_INCOMPLETE_WORK,
    MetricCategory.FAILED_APPROACH: FailureClass.LOST_FAILED_APPROACH,
    MetricCategory.ENTITY: FailureClass.ENTITY_CONFUSION,
    MetricCategory.HALLUCINATION: FailureClass.HALLUCINATION,
}
"""What a failed check means, when its category alone is enough to say."""

_REFINED: dict[tuple[MetricCategory, FailureMode], FailureClass] = {
    (MetricCategory.CONSTRAINT, FailureMode.NEGATIVE_CONSTRAINT): (
        FailureClass.LOST_NEGATIVE_CONSTRAINT
    ),
    (MetricCategory.CRITICAL_FACT, FailureMode.EXACT_VALUE): FailureClass.EXACT_VALUE_LOSS,
    (MetricCategory.CRITICAL_FACT, FailureMode.CURRENT_STATE): FailureClass.LOST_CURRENT_STATE,
    (MetricCategory.CRITICAL_FACT, FailureMode.SIMILAR_ENTITIES): FailureClass.ENTITY_CONFUSION,
    (MetricCategory.CRITICAL_FACT, FailureMode.TEMPORAL_STATE): FailureClass.TEMPORAL_CONFUSION,
    (MetricCategory.DECISION, FailureMode.REVERSED_DECISION): FailureClass.LOST_DECISION,
}
"""Where the category is not specific enough and the scenario's trap decides.

A failed constraint check is a lost constraint in general and a lost *negative*
constraint when the scenario was built around one — the distinction the roadmap
draws, and one the category alone cannot make because both are written as
constraints.
"""


@dataclass(frozen=True)
class Observation:
    """One classified failure, with what it was derived from.

    ``evidence`` names the specific check, criterion, or measurement, so a count
    in a report can be traced back to the row that produced it rather than
    taken on trust.
    """

    failure_class: FailureClass
    scenario_id: str
    strategy: str
    evidence: str


def classify_result(result: EvaluationResult) -> list[Observation]:
    """Every failure one run exhibited.

    **Only successful runs are classified.** A run that failed with a model
    error or a rate limit produced no response, so it exhibits no retention
    failure — counting it as one would turn an outage into a finding about a
    strategy. Those are reported separately as run failures.
    """
    if not result.succeeded:
        return []

    modes = {FailureMode(mode) for mode in result.failure_modes if mode in set(FailureMode)}
    observations: list[Observation] = []

    for check in result.retention_checks:
        if check.passed:
            continue
        observations.append(
            Observation(
                failure_class=_classify_check(check.category, modes),
                scenario_id=result.scenario_id,
                strategy=result.strategy,
                evidence=f"retention check {check.check_id} ({check.category}) failed",
            )
        )

    for outcome in result.completion:
        if outcome.fabricated:
            observations.append(
                Observation(
                    failure_class=FailureClass.HALLUCINATION,
                    scenario_id=result.scenario_id,
                    strategy=result.strategy,
                    evidence=(
                        f"completion criterion {outcome.criterion_id} used something the "
                        f"conversation ruled out: {outcome.detail or 'no detail'}"
                    ),
                )
            )

    overflow = _budget_overflow(result)
    if overflow is not None:
        observations.append(
            Observation(
                failure_class=FailureClass.BUDGET_OVERFLOW,
                scenario_id=result.scenario_id,
                strategy=result.strategy,
                evidence=overflow,
            )
        )

    return observations


def _classify_check(category: str, modes: set[FailureMode]) -> FailureClass:
    """The class a failed check belongs to, given the scenario's traps.

    A refinement wins over the category default when the scenario sets exactly
    the trap that distinguishes them. Where a scenario sets several traps that
    each refine the same category, the first in a fixed order wins, so the
    classification does not depend on set iteration order.
    """
    try:
        metric = MetricCategory(category)
    except ValueError:
        return FailureClass.UNCLASSIFIED

    for mode in sorted(modes, key=lambda item: item.value):
        refined = _REFINED.get((metric, mode))
        if refined is not None:
            return refined
    return _BY_CATEGORY.get(metric, FailureClass.UNCLASSIFIED)


def _budget_overflow(result: EvaluationResult) -> str | None:
    """Whether a budgeted arm produced a context larger than its target.

    Reference arms are exempt: they consult no budget, record ``target_tokens``
    as zero, and a conversation longer than zero is not an overflow. Counting
    them would put a budget failure on every reference row and make the class
    meaningless.

    Measured against the run's own target and the context it actually produced.
    A count that is an estimate is still compared — the alternative is not
    checking at all — but the estimate is what the strategies themselves
    enforced against, so the comparison is at least self-consistent.
    """
    target = result.run.target_tokens
    if not target:
        return None
    if result.context_tokens <= target:
        return None
    return (
        f"context is {result.context_tokens} tokens against a target of {target}"
        f"{' (estimated count)' if not result.tokens_exact else ''}"
    )


def classify(results: Iterable[EvaluationResult]) -> list[Observation]:
    """Every failure across a results set."""
    observations: list[Observation] = []
    for result in results:
        observations.extend(classify_result(result))
    return observations


def counts_by_class(
    observations: Iterable[Observation],
) -> dict[FailureClass, Counter[str]]:
    """Failures per class, broken down by strategy.

    The shape the report needs: a class is only interesting next to which arm
    exhibited it, since a failure every arm shares is a property of the dataset
    and one only compaction shows is a property of compaction.
    """
    table: dict[FailureClass, Counter[str]] = {}
    for observation in observations:
        table.setdefault(observation.failure_class, Counter())[observation.strategy] += 1
    return table


def format_taxonomy(results: Sequence[EvaluationResult], *, strategies: Sequence[str]) -> str:
    """The failure taxonomy as a table, including what could not be measured."""
    observations = classify(results)
    table = counts_by_class(observations)
    arms = list(strategies)

    width = max([len(item.value) for item in FailureClass] + [20])
    header = f"{'failure class':<{width}}  " + "  ".join(f"{arm:>18}" for arm in arms)
    lines = [header, "-" * len(header)]

    for failure_class in FailureClass:
        if failure_class in UNMEASURED:
            continue
        counts = table.get(failure_class, Counter())
        if not counts and failure_class is FailureClass.UNCLASSIFIED:
            continue
        cells = "  ".join(f"{counts.get(arm, 0):>18}" for arm in arms)
        lines.append(f"{failure_class.value:<{width}}  {cells}")

    lines.append("")
    lines.append("Declared and not measured — reported as unmeasured, never as zero:")
    for failure_class, reason in UNMEASURED.items():
        lines.append(f"  {failure_class.value}: {reason}")

    total = len(observations)
    lines.append("")
    lines.append(
        f"{total} classified failures across {len(results)} results. "
        "Counts are failures, not rates: a scenario with four checks can contribute four."
    )
    return "\n".join(lines)


__all__ = [
    "UNMEASURED",
    "FailureClass",
    "Observation",
    "classify",
    "classify_result",
    "counts_by_class",
    "format_taxonomy",
]
