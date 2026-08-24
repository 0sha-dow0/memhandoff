"""The first benchmark.

```
scenarios x strategies x budgets x repetitions -> results
```

**This orchestrates the Phase 5.5 runner; it does not replace it.** Each cell of
the matrix is an ordinary `ExperimentRunner` run, so leakage prevention, the
same-prompt guarantee, failure statuses, and the run fingerprint all come from
the code that was already reviewed. A second runner would be a second place for
those guarantees to be wrong.

**Reference arms are budget-independent and run once.** Full context ignores the
target, so running it at every budget would take three independent samples of one
condition and then average them into a row claiming to be budget-independent.
For a nondeterministic model that is not a reference; it is three draws wearing
one label. So the matrix is:

```
scenario x repetition -> reference arms once
                      -> compacted arms once per budget
```

giving ``scenarios x repetitions x (references + compacted x budgets)`` runs.
Every compacted budget is compared against the same reference population, and
the reference does not move with the budget.

**A run per budget, and per repetition.** A budget change alters the fingerprint,
so each budget is a distinct experiment with its own `run_id` — the honest
reading, since results at different budgets are not interchangeable. Repetitions
go into the experiment configuration for the same reason: two runs of one
configuration would otherwise be indistinguishable in a results file.

**Nothing here decides which strategy is better.** The benchmark produces
numbers and a table; whether they mean anything depends on whether they came
from a real model, which `evaluation_mode` records on every row.

**No result in this repository is fabricated.** If a real provider is not
configured, the benchmark runs in deterministic mode against the fakes, says so
in every row and at the top of every report, and produces nothing that could be
mistaken for a measurement of model quality.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from open_context.compaction import Prompt
from open_context.llm import LLMProvider, Tokenizer
from open_context_eval.benchmark_dataset import BENCHMARK_SCENARIOS
from open_context_eval.judge import EvaluationJudge
from open_context_eval.prompts import CONTINUATION_PROMPT_V1
from open_context_eval.results import EvaluationResult, RunStatus
from open_context_eval.runner import EvaluationConfig, ExperimentRunner, SkipPredicate
from open_context_eval.scenario import EvaluationScenario
from open_context_eval.strategies import (
    FULL_CONTEXT,
    HYBRID_V1,
    PHASE_5_BASELINE,
    SIMPLE_SUMMARY_V1,
    EvaluationStrategy,
    default_strategies,
)

DETERMINISTIC_TEST = "deterministic_test"
REAL_MODEL = "real_model"

BENCHMARK_STRATEGIES: tuple[str, ...] = (
    FULL_CONTEXT,
    SIMPLE_SUMMARY_V1,
    PHASE_5_BASELINE,
    HYBRID_V1,
)
"""The arms of the first benchmark.

Retrieval is absent because it does not exist. A named arm with no
implementation would appear in the report as a strategy that scored nothing,
which is not the same as one that was never run.

`hybrid_v1` joined in Phase 8. It is the most expensive arm — two model calls
per context instead of one — and was added on a specific falsifiable bet rather
than on the general principle that more structure is better: Phase 5.8 measured
exact-value loss as the thing compaction actually costs, and structured state is
the mechanism that can carry an exact value through intact.
"""

REFERENCE_BUDGET = 0
"""The ``target_tokens`` a budget-independent arm records.

Zero rather than one of the real budgets, because a reference result carrying
``target_tokens=80`` would claim to have been produced under a budget it never
consulted. Zero reads as "no budget applied", and the report groups on it
directly rather than special-casing a strategy by name.
"""

DEFAULT_BUDGETS: tuple[int, ...] = (80, 160, 240)
"""Small, medium, and large, chosen against the dataset rather than in the abstract.

The v2 scenarios are 242 to 417 tokens, so these three sit below every one of
them and compaction genuinely happens at each. A budget above the largest
scenario would measure the passthrough path and call it compression; one below
the harness minimum would make every arm fail and call that a finding.
"""


class InvalidBenchmarkError(ValueError):
    """A benchmark configuration that could not produce a fair comparison."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """One benchmark: what to run, at what sizes, how many times."""

    provider: LLMProvider
    tokenizer: Tokenizer
    evaluation_mode: str = DETERMINISTIC_TEST
    budgets: Sequence[int] = DEFAULT_BUDGETS
    strategies: Sequence[str] = BENCHMARK_STRATEGIES
    repetitions: int = 1
    """How many times each cell runs.

    One by default. A real model is nondeterministic and would need more, but
    nothing here averages repetitions away: each is recorded separately so the
    reporting layer can show spread instead of hiding it.
    """

    continuation_prompt: Prompt = CONTINUATION_PROMPT_V1
    judge: EvaluationJudge | None = None
    registry: dict[str, EvaluationStrategy] = field(default_factory=default_strategies)
    scenarios: Sequence[EvaluationScenario] = BENCHMARK_SCENARIOS

    def __post_init__(self) -> None:
        if not self.budgets:
            raise InvalidBenchmarkError("a benchmark needs at least one target budget")
        if any(budget <= 0 for budget in self.budgets):
            raise InvalidBenchmarkError(f"budgets must be positive, got {list(self.budgets)}")
        if self.repetitions < 1:
            raise InvalidBenchmarkError("repetitions must be at least one")
        if not self.strategies:
            raise InvalidBenchmarkError("a benchmark needs at least one strategy")
        unknown = [name for name in self.strategies if name not in self.registry]
        if unknown:
            raise InvalidBenchmarkError(f"no strategy registered as {unknown}")
        if not self.scenarios:
            raise InvalidBenchmarkError("a benchmark needs at least one scenario")
        if self.evaluation_mode not in {DETERMINISTIC_TEST, REAL_MODEL}:
            raise InvalidBenchmarkError(
                f"evaluation_mode must be {DETERMINISTIC_TEST!r} or {REAL_MODEL!r}, "
                f"got {self.evaluation_mode!r}"
            )
        versions = {scenario.dataset_version for scenario in self.scenarios}
        if len(versions) > 1:
            raise InvalidBenchmarkError(
                f"scenarios span dataset versions {sorted(versions)}; results would be incomparable"
            )

    @property
    def dataset_version(self) -> str:
        return next(iter({scenario.dataset_version for scenario in self.scenarios}))

    @property
    def reference_strategies(self) -> list[str]:
        """Arms the target budget does not affect. Run once per scenario and repetition."""
        return [name for name in self.strategies if self.registry[name].budget_independent]

    @property
    def compacted_strategies(self) -> list[str]:
        """Arms held to a budget. Run once per budget."""
        return [name for name in self.strategies if not self.registry[name].budget_independent]

    @property
    def cells(self) -> int:
        """How many runs the matrix produces.

        ``scenarios x repetitions x (references + compacted x budgets)``. The
        reference term has no budget factor, which is the whole point.
        """
        per_repetition = len(self.reference_strategies) + len(self.compacted_strategies) * len(
            self.budgets
        )
        return len(self.scenarios) * self.repetitions * per_repetition

    def describe(self) -> str:
        return (
            f"{len(self.scenarios)} scenarios x {self.repetitions} repetitions x "
            f"({len(self.reference_strategies)} reference + "
            f"{len(self.compacted_strategies)} compacted x {len(self.budgets)} budgets) "
            f"= {self.cells} runs"
        )


def experiment_configs(config: BenchmarkConfig) -> Iterator[tuple[int, int, EvaluationConfig]]:
    """Every evaluation configuration the matrix needs.

    Yields ``(budget, repetition, config)``. Reference arms appear once per
    repetition at ``REFERENCE_BUDGET``; compacted arms appear once per budget per
    repetition. Nothing runs a reference arm three times and calls the average a
    reference.

    The repetition index travels in the experiment's extra configuration so it
    reaches the fingerprint: two runs of an otherwise identical configuration
    must not share a ``run_id``, or a results file could not tell them apart.
    """

    def build(budget: int, repetition: int, strategies: list[str]) -> EvaluationConfig:
        return EvaluationConfig(
            provider=config.provider,
            tokenizer=config.tokenizer,
            target_tokens=budget,
            strategies=strategies,
            continuation_prompt=config.continuation_prompt,
            judge=config.judge,
            registry=config.registry,
            evaluation_mode=config.evaluation_mode,
            extra={"benchmark_repetition": repetition},
        )

    for repetition in range(1, config.repetitions + 1):
        if config.reference_strategies:
            yield (
                REFERENCE_BUDGET,
                repetition,
                build(REFERENCE_BUDGET, repetition, config.reference_strategies),
            )
        for budget in config.budgets:
            if config.compacted_strategies:
                yield budget, repetition, build(budget, repetition, config.compacted_strategies)


def run_benchmark(config: BenchmarkConfig, *, root: str | Path) -> list[EvaluationResult]:
    """Execute the matrix and return every result.

    A thin wrapper over ``stream_benchmark``. A run that is spending a metered
    quota should consume the stream and persist each result instead, so that
    stopping early keeps what it has already paid for.
    """
    return list(stream_benchmark(config, root=root))


def stream_benchmark(
    config: BenchmarkConfig,
    *,
    root: str | Path,
    skip: SkipPredicate | None = None,
) -> Iterator[EvaluationResult]:
    """Yield each cell's result as it is produced.

    Scenario archives are materialised once and reused across budgets and
    repetitions: materialisation is content-addressed and idempotent, so running
    the same scenario thirty times costs one import and cannot double the
    conversation.

    ``skip`` is passed straight through to the runner and is how a resumed run
    skips cells a previous one recorded. Cells are visited in a fixed order —
    repetition, then reference arms, then each budget — so an interrupted run
    resumes at a predictable place rather than wherever a set iteration happened
    to land.

    **Exceptions are not swallowed.** ``RequestBudgetExhausted`` in particular
    propagates to the caller, whose job it is to stop; everything yielded before
    it is already the caller's and, if the caller was persisting, already on
    disk.
    """
    workspace = Path(root)
    for _budget, _repetition, experiment in experiment_configs(config):
        runner = ExperimentRunner(experiment, root=workspace / "archive")
        yield from runner.stream(config.scenarios, skip=skip)


def estimated_requests(config: BenchmarkConfig) -> int:
    """How many model requests the matrix will send, at minimum.

    One continuation per cell, plus one compaction call for each cell whose
    strategy calls a model to build its context. **A lower bound, not a
    promise**: the Phase 5 compactor chunks history that exceeds the model's
    context window and retries a summary that overshot its budget, and either
    adds calls this cannot see in advance.

    It exists so a run against a metered quota can be sized before it starts
    rather than discovered at request fifty-one.
    """
    per_repetition = 0
    for name in config.reference_strategies:
        per_repetition += 1 + config.registry[name].requests_per_run
    for name in config.compacted_strategies:
        per_repetition += (1 + config.registry[name].requests_per_run) * len(config.budgets)
    return len(config.scenarios) * config.repetitions * per_repetition


def check_comparable(results: Sequence[EvaluationResult]) -> list[str]:
    """Reasons a set of results is not a fair algorithm comparison.

    Empty when the comparison is sound. A benchmark that quietly compared full
    context on one model against compaction on another would be measuring
    models, and would look exactly like a result about strategies.
    """
    ok = [result for result in results if result.succeeded or result.status.value != "success"]
    problems = []

    models = {(r.run.provider, r.run.model) for r in ok}
    if len(models) > 1:
        problems.append(
            f"results span more than one downstream model: {sorted(models)}; this cannot be "
            "read as a comparison of context representations"
        )

    prompts = {(r.run.continuation_prompt_id, r.run.continuation_prompt_hash) for r in ok}
    if len(prompts) > 1:
        problems.append(f"results span more than one continuation prompt: {sorted(prompts)}")

    datasets = {r.run.dataset_version for r in ok}
    if len(datasets) > 1:
        problems.append(f"results span dataset versions {sorted(datasets)}")

    modes = {r.evaluation_mode for r in ok}
    if len(modes) > 1:
        problems.append(
            f"results mix evaluation modes {sorted(modes)}; deterministic and real-model "
            "numbers are not comparable"
        )
    problems.extend(_coverage_problems(results))
    return problems


def _coverage_problems(results: Sequence[EvaluationResult]) -> list[str]:
    """Whether the arms were measured on the same scenarios.

    **Found by the Phase 5.8 run, which this check would have caught.** A
    benchmark stopped by a request budget leaves the reference arm complete and
    the compacted arms partway through the scenario list. The report then prints
    one row per arm, each averaging a different subset, and the rows line up in a
    table that invites exactly the comparison they cannot support.

    Unequal ``Runs`` counts are visible in that table, but only to a reader who
    thinks to distrust it. A benchmark whose output has to be distrusted to be
    read correctly is not reporting; it is leaving the work to the reader.

    **The reference arm is included in every budget's comparison**, because that
    is what it is for: each budget row is read against the same reference
    population. Grouping strictly by budget would put the reference in a group of
    its own and never check it against anything, which would miss the largest
    coverage gap a stopped run produces.
    """
    succeeded = [r for r in results if r.status is RunStatus.SUCCESS]
    if not succeeded:
        return []

    by_budget: dict[int, dict[str, set[str]]] = {}
    for result in succeeded:
        arms = by_budget.setdefault(result.run.target_tokens, {})
        arms.setdefault(result.strategy, set()).add(result.scenario_id)

    every_scenario = {result.scenario_id for result in succeeded}
    reference = by_budget.get(REFERENCE_BUDGET, {})

    problems = []
    for budget, arms in sorted(by_budget.items()):
        if budget == REFERENCE_BUDGET:
            continue
        coverage = {**reference, **arms}
        if len(coverage) < 2:
            continue
        shared = set.intersection(*coverage.values())
        if any(scenarios != shared for scenarios in coverage.values()):
            detail = ", ".join(
                f"{name} {len(scenarios)}" for name, scenarios in sorted(coverage.items())
            )
            problems.append(
                f"at budget {budget}, the arms were measured on different scenarios "
                f"({detail}, of {len(every_scenario)} attempted); only {len(shared)} "
                "scenario(s) have every arm, so the per-arm rows average different subsets "
                "and cannot be compared to each other"
            )
    return problems


__all__ = [
    "BENCHMARK_STRATEGIES",
    "DEFAULT_BUDGETS",
    "DETERMINISTIC_TEST",
    "REAL_MODEL",
    "REFERENCE_BUDGET",
    "BenchmarkConfig",
    "InvalidBenchmarkError",
    "check_comparable",
    "estimated_requests",
    "experiment_configs",
    "run_benchmark",
    "stream_benchmark",
]
