"""The benchmark dataset, version 2.

**v1 is not modified.** Phase 5.6 needed task-completion criteria, which v1
scenarios do not carry, and editing them in place would have meant two different
benchmarks sharing one version number. So v2 is built by taking each v1 scenario
and attaching criteria to it, leaving v1 importable and unchanged.

The conversations, tasks, and retention checks are v1's, verbatim. What is new
is the completion criteria and a few additional failure-mode tags — nothing that
changes what a v1 run would have measured.

Criteria are written to *discriminate*, not to match. Each names the answer the
conversation supports and the answers it ruled out, so a response hedging across
every option it can remember fails rather than passes. See ``completion.py``.
"""

from __future__ import annotations

from open_context_eval.completion import CompletionCriterion, CompletionKind
from open_context_eval.dataset import SCENARIOS as V1_SCENARIOS
from open_context_eval.scenario import EvaluationScenario, FailureMode

BENCHMARK_DATASET_VERSION = "v2"

MINIMUM_SUBSTANCE_WORDS = 12
"""A continuation shorter than this is not a continuation of anything.

A floor, not a quality bar. It exists so a one-line answer cannot satisfy a task
that asked for an approach to be described.
"""


def _criterion(**kwargs: object) -> CompletionCriterion:
    return CompletionCriterion(**kwargs)


CRITERIA: dict[str, tuple[CompletionCriterion, ...]] = {
    "early-critical-fact": (
        _criterion(
            criterion_id="respects-offline",
            kind=CompletionKind.EXPECTED_OUTPUT_PROPERTY,
            description="Proposes a fetch strategy that honours the offline requirement.",
            expected=("offline", "no network", "without network", "local"),
            alternatives=("download from", "call the api", "fetch from the internet"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "late-critical-fact": (
        _criterion(
            criterion_id="works-to-45-minutes",
            kind=CompletionKind.EXACT_VALUE,
            description="Works to the 45 minute ceiling and not another figure.",
            expected=("45",),
            alternatives=("30 minute", "60 minute", "an hour", "two hours"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "negative-constraint": (
        _criterion(
            criterion_id="no-redis",
            kind=CompletionKind.FORBIDDEN_ACTION,
            description="Does not reintroduce Redis after it was ruled out.",
            alternatives=("redis",),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "reversed-decision": (
        _criterion(
            criterion_id="uses-sqlite",
            kind=CompletionKind.EXPECTED_DECISION,
            description="Writes SQLite setup, not the superseded PostgreSQL.",
            expected=("sqlite",),
            alternatives=("postgresql", "postgres"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "failed-approach": (
        _criterion(
            criterion_id="does-not-retry-in-memory",
            kind=CompletionKind.FORBIDDEN_ACTION,
            description="Does not propose the in-memory sort that already failed.",
            alternatives=(
                "load the whole table into memory",
                "in-memory sort",
                "sort it in memory",
                "load it all into memory",
            ),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "exact-value": (
        _criterion(
            criterion_id="timeout-17",
            kind=CompletionKind.EXACT_VALUE,
            description="Configures the timeout at 17 seconds, not a rounder number.",
            expected=("17",),
            alternatives=("15 second", "30 second", "60 second", "timeout=30", "timeout=60"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "similar-entities": (
        _criterion(
            criterion_id="right-service",
            kind=CompletionKind.EXPECTED_ENTITY,
            description="Investigates billing-worker and leaves billing-api alone.",
            expected=("billing-worker",),
            alternatives=("investigate billing-api", "billing-api leak", "look at billing-api"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "temporal-state": (
        _criterion(
            criterion_id="migration-complete",
            kind=CompletionKind.EXPECTED_STATE,
            description="Reports the migration as finished, not stalled at an earlier state.",
            expected=("complete", "finished", "applied", "done"),
            alternatives=("step 3", "not started", "half applied", "halfway"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "tool-result": (
        _criterion(
            criterion_id="row-count",
            kind=CompletionKind.EXACT_VALUE,
            description="Uses the row count the tool returned.",
            expected=("4823991", "4,823,991", "4.8 million", "4.8m"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "open-task": (
        _criterion(
            criterion_id="continues-csv-export",
            kind=CompletionKind.EXPECTED_OUTPUT_PROPERTY,
            description="Continues the CSV export, not work already finished.",
            expected=("csv",),
            alternatives=("start with rate limiting", "begin the audit log", "build rate limiting"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "rationale": (
        _criterion(
            criterion_id="gives-the-reason",
            kind=CompletionKind.EXPECTED_OUTPUT_PROPERTY,
            description="Explains why string timestamps are required, not merely that they are.",
            expected=("mainframe", "downstream reader", "cannot parse"),
            alternatives=("go ahead and switch", "native date types are better"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "multi-step": (
        _criterion(
            criterion_id="next-step-is-index",
            kind=CompletionKind.EXPECTED_STATE,
            description="Continues at the index step rather than repeating finished work.",
            expected=("index",),
            alternatives=("add the column", "run the backfill", "start the backfill"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "repeated-information": (
        _criterion(
            criterion_id="corrected-sla",
            kind=CompletionKind.EXACT_VALUE,
            description="Uses the corrected 99.99 figure, not the repeated 99.9.",
            expected=("99.99",),
            alternatives=("99.9 percent", "99.9%"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "confusable-numbers": (
        _criterion(
            criterion_id="metrics-port",
            kind=CompletionKind.EXPECTED_ENTITY,
            description="Opens the metrics port and not a neighbouring one.",
            expected=("8082",),
            alternatives=("8080", "8081"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    "decision-with-constraint": (
        _criterion(
            criterion_id="in-process-queue",
            kind=CompletionKind.EXPECTED_DECISION,
            description="Builds the in-process queue and does not reintroduce Kafka.",
            expected=("sqlite", "in-process", "in process"),
            alternatives=("kafka",),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
}

EXTRA_FAILURE_MODES: dict[str, tuple[FailureMode, ...]] = {
    "early-critical-fact": (FailureMode.POSITIVE_CONSTRAINT,),
    "late-critical-fact": (FailureMode.POSITIVE_CONSTRAINT,),
    "temporal-state": (FailureMode.CURRENT_STATE,),
    "multi-step": (FailureMode.CURRENT_STATE,),
    "reversed-decision": (FailureMode.DECISION,),
    "decision-with-constraint": (FailureMode.DECISION,),
}


def _upgrade(scenario: EvaluationScenario) -> EvaluationScenario:
    """One v1 scenario, restamped as v2 with completion criteria attached."""
    extra = EXTRA_FAILURE_MODES.get(scenario.scenario_id, ())
    return scenario.model_copy(
        update={
            "dataset_version": BENCHMARK_DATASET_VERSION,
            "completion": CRITERIA.get(scenario.scenario_id, ()),
            "failure_modes": (*scenario.failure_modes, *extra),
        }
    )


BENCHMARK_SCENARIOS: tuple[EvaluationScenario, ...] = tuple(
    _upgrade(scenario) for scenario in V1_SCENARIOS
)


def by_id(scenario_id: str) -> EvaluationScenario:
    for scenario in BENCHMARK_SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(
        f"no scenario {scenario_id!r}; have {[s.scenario_id for s in BENCHMARK_SCENARIOS]}"
    )


__all__ = [
    "BENCHMARK_DATASET_VERSION",
    "BENCHMARK_SCENARIOS",
    "CRITERIA",
    "MINIMUM_SUBSTANCE_WORDS",
    "by_id",
]
