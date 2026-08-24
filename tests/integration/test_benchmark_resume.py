"""Surviving a run that stops early.

A benchmark against a metered free tier cannot finish in one sitting, so the
properties tested here are the ones that make a partial run worth something:
results reach disk as they are produced, a stop is not a failure, and resuming
does not pay twice for a cell it already has.

Offline and deterministic throughout. No credential, no network.
"""

import pytest

from open_context.llm.fakes import FakeProvider, WordTokenizer
from open_context_eval.benchmark import (
    BenchmarkConfig,
    check_comparable,
    estimated_requests,
    stream_benchmark,
)
from open_context_eval.benchmark_dataset import by_id
from open_context_eval.budget import BudgetedProvider, RequestBudget, RequestBudgetExhausted
from open_context_eval.results import (
    ResultWriter,
    RunStatus,
    completed_cells,
    read_results,
)
from open_context_eval.runner import EvaluationConfig, ExperimentRunner

pytestmark = pytest.mark.integration

SMALL = [by_id("reversed-decision"), by_id("negative-constraint")]


def provider(**kwargs):
    return FakeProvider(
        reply=kwargs.pop("reply", "continuing the work described in the context above"),
        context_window=kwargs.pop("context_window", 100_000),
        **kwargs,
    )


def config(**kwargs):
    defaults = {
        "provider": provider(),
        "tokenizer": WordTokenizer(exact=True),
        "budgets": (160,),
        "scenarios": SMALL,
    }
    return BenchmarkConfig(**{**defaults, **kwargs})


# ----------------------------------------------------------------------
# Streaming


def test_the_runner_yields_results_one_at_a_time(tmp_path):
    """Not a list built and returned at the end.

    Against a metered quota a request is not repeatable on demand, so a run that
    accumulated privately and raised before returning would discard every
    measurement it had already bought.
    """
    experiment = EvaluationConfig(
        provider=provider(), tokenizer=WordTokenizer(exact=True), target_tokens=160
    )
    runner = ExperimentRunner(experiment, root=tmp_path / "archive")
    stream = runner.stream(SMALL)

    first = next(stream)
    assert first.scenario_id
    assert list(stream), "the rest of the run is still available"


def test_run_still_returns_everything(tmp_path):
    """The old interface is unchanged; ``run`` is now a wrapper over ``stream``."""
    experiment = EvaluationConfig(
        provider=provider(), tokenizer=WordTokenizer(exact=True), target_tokens=160
    )
    runner = ExperimentRunner(experiment, root=tmp_path / "archive")
    results = runner.run(SMALL)
    assert len(results) == len(SMALL) * len(experiment.selected())


def test_stream_benchmark_covers_the_whole_matrix(tmp_path):
    settings = config()
    results = list(stream_benchmark(settings, root=tmp_path))
    assert len(results) == settings.cells


# ----------------------------------------------------------------------
# Skipping


def test_a_skipped_cell_is_neither_run_nor_recorded(tmp_path):
    """Skipping means absent, not a second row saying nothing happened."""
    settings = config()
    everything = list(stream_benchmark(settings, root=tmp_path / "first"))
    skip_all = {(r.run.run_id, r.scenario_id, r.strategy) for r in everything}

    again = list(
        stream_benchmark(
            config(),
            root=tmp_path / "second",
            skip=lambda run, scenario, strategy: (run, scenario, strategy) in skip_all,
        )
    )
    assert again == []


def test_skipping_one_arm_still_runs_the_others(tmp_path):
    settings = config()
    everything = list(stream_benchmark(settings, root=tmp_path / "first"))
    one = everything[0]

    remaining = list(
        stream_benchmark(
            config(),
            root=tmp_path / "second",
            skip=lambda run, scenario, strategy: (
                (run, scenario, strategy) == (one.run.run_id, one.scenario_id, one.strategy)
            ),
        )
    )
    assert len(remaining) == len(everything) - 1


def test_a_skipped_cell_costs_no_request(tmp_path):
    """The point of resuming: a cell already recorded is never paid for again."""
    counted = provider()
    settings = config(provider=counted)
    everything = list(stream_benchmark(settings, root=tmp_path / "first"))
    spent_first = len(counted.requests)
    assert spent_first > 0

    skip_all = {(r.run.run_id, r.scenario_id, r.strategy) for r in everything}
    second = provider()
    list(
        stream_benchmark(
            config(provider=second),
            root=tmp_path / "second",
            skip=lambda run, scenario, strategy: (run, scenario, strategy) in skip_all,
        )
    )
    assert len(second.requests) == 0


# ----------------------------------------------------------------------
# Persistence


def test_the_writer_flushes_each_result(tmp_path):
    """A killed process must leave what it had, not an empty file."""
    destination = tmp_path / "results.jsonl"
    settings = config()
    writer = ResultWriter(destination)
    for index, result in enumerate(stream_benchmark(settings, root=tmp_path)):
        writer.write(result)
        assert len(list(read_results(destination))) == index + 1, "readable before close"
    writer.close()


def test_writing_appends_rather_than_replacing(tmp_path):
    destination = tmp_path / "results.jsonl"
    first = list(stream_benchmark(config(), root=tmp_path / "a"))

    with ResultWriter(destination) as writer:
        for result in first:
            writer.write(result)
    with ResultWriter(destination) as writer:
        for result in first:
            writer.write(result)

    assert len(list(read_results(destination))) == len(first) * 2


def test_completed_cells_reads_back_what_ran(tmp_path):
    destination = tmp_path / "results.jsonl"
    settings = config()
    results = list(stream_benchmark(settings, root=tmp_path))
    with ResultWriter(destination) as writer:
        for result in results:
            writer.write(result)

    done = completed_cells(destination)
    assert done == {(r.run.run_id, r.scenario_id, r.strategy) for r in results}


def test_completed_cells_of_a_missing_file_is_empty(tmp_path):
    assert completed_cells(tmp_path / "nothing.jsonl") == set()


def test_a_failed_cell_does_not_count_as_done(tmp_path):
    """A rate limit says nothing about a strategy, so the cell is worth retrying.

    Treating it as complete would freeze a transient outage into the results
    permanently.
    """
    destination = tmp_path / "results.jsonl"
    settings = config()
    results = list(stream_benchmark(settings, root=tmp_path))
    broken = results[0].model_copy(update={"status": RunStatus.MODEL_ERROR})

    with ResultWriter(destination) as writer:
        writer.write(broken)
        for result in results[1:]:
            writer.write(result)

    done = completed_cells(destination)
    assert (broken.run.run_id, broken.scenario_id, broken.strategy) not in done
    assert len(done) == len(results) - 1


# ----------------------------------------------------------------------
# Stopping on the budget


def test_a_run_stops_when_the_budget_runs_out(tmp_path):
    budget = RequestBudget(3)
    settings = config(provider=BudgetedProvider(provider(), budget))

    produced = []
    with pytest.raises(RequestBudgetExhausted):
        for result in stream_benchmark(settings, root=tmp_path):
            produced.append(result)

    assert budget.spent == 3
    assert len(produced) < settings.cells, "it stopped before the matrix finished"


def test_what_was_produced_before_the_stop_is_kept(tmp_path):
    """The whole point. A stop must not cost the requests already spent."""
    destination = tmp_path / "results.jsonl"
    settings = config(provider=BudgetedProvider(provider(), RequestBudget(3)))

    writer = ResultWriter(destination)
    try:
        for result in stream_benchmark(settings, root=tmp_path):
            writer.write(result)
    except RequestBudgetExhausted:
        pass
    finally:
        writer.close()

    persisted = list(read_results(destination))
    assert persisted, "results bought before the stop survived"
    assert all(r.status is RunStatus.SUCCESS for r in persisted)


def test_cells_past_the_budget_are_absent_not_failed(tmp_path):
    """Exhaustion is not an ``LLMError``, so no cell is recorded as a model failure.

    If it were, the results file would assert that a strategy was tried and did
    not work on every cell nobody asked about.
    """
    destination = tmp_path / "results.jsonl"
    settings = config(provider=BudgetedProvider(provider(), RequestBudget(2)))

    writer = ResultWriter(destination)
    try:
        for result in stream_benchmark(settings, root=tmp_path):
            writer.write(result)
    except RequestBudgetExhausted:
        pass
    finally:
        writer.close()

    persisted = list(read_results(destination))
    assert not [r for r in persisted if r.status is RunStatus.MODEL_ERROR]
    assert len(persisted) < settings.cells


def test_a_stopped_run_resumes_where_it_left_off(tmp_path):
    """Two capped runs plus a resume equal one uncapped run."""
    destination = tmp_path / "results.jsonl"

    def attempt(limit):
        settings = config(provider=BudgetedProvider(provider(), RequestBudget(limit)))
        done = completed_cells(destination)
        writer = ResultWriter(destination)
        try:
            for result in stream_benchmark(
                settings,
                root=tmp_path / "archive",
                skip=lambda run, scenario, strategy: (run, scenario, strategy) in done,
            ):
                writer.write(result)
        except RequestBudgetExhausted:
            pass
        finally:
            writer.close()

    attempt(2)
    partial = len(list(read_results(destination)))
    assert 0 < partial < config().cells

    attempt(100)
    finished = list(read_results(destination))
    assert len(finished) == config().cells
    cells = {(r.run.run_id, r.scenario_id, r.strategy) for r in finished}
    assert len(cells) == config().cells, "no cell was measured twice"


# ----------------------------------------------------------------------
# Sizing


def test_estimated_requests_counts_a_continuation_for_every_cell():
    """One reference arm sending no compaction call still costs one request."""
    settings = config(strategies=("full_context",), budgets=(160,))
    assert estimated_requests(settings) == len(SMALL)


def test_estimated_requests_adds_a_compaction_call_for_compacted_arms():
    settings = config(strategies=("simple_summary_v1",), budgets=(160,))
    assert estimated_requests(settings) == len(SMALL) * 2


def test_estimated_requests_scales_with_budgets_and_repetitions():
    one = config(strategies=("simple_summary_v1",), budgets=(80,))
    three = config(strategies=("simple_summary_v1",), budgets=(80, 160, 240))
    assert estimated_requests(three) == estimated_requests(one) * 3

    twice = config(strategies=("simple_summary_v1",), budgets=(80,), repetitions=2)
    assert estimated_requests(twice) == estimated_requests(one) * 2


def test_the_estimate_is_a_lower_bound_on_a_real_run(tmp_path):
    """It may undercount, and must never overcount.

    The compactor chunks and retries, so a run can cost more than predicted. A
    prediction above the truth would make a run stop short of the quota it
    actually had.
    """
    counted = provider()
    settings = config(provider=counted)
    list(stream_benchmark(settings, root=tmp_path))
    assert estimated_requests(settings) <= len(counted.requests)


# ----------------------------------------------------------------------
# Coverage: did the arms actually run on the same scenarios?


def test_equal_coverage_is_not_a_problem(tmp_path):
    results = list(stream_benchmark(config(), root=tmp_path))
    assert check_comparable(results) == []


def test_arms_on_different_scenarios_are_flagged(tmp_path):
    """The confound a budget-stopped run produces, and the one a table hides.

    Unequal ``Runs`` counts are visible to a reader who thinks to distrust the
    table. A benchmark that has to be distrusted to be read correctly is leaving
    the work to the reader.
    """
    results = list(stream_benchmark(config(), root=tmp_path))
    truncated = [
        r
        for r in results
        if not (r.strategy == "phase_5_baseline" and r.scenario_id == SMALL[1].scenario_id)
    ]
    problems = check_comparable(truncated)
    assert problems, "an arm missing a scenario the others have must be reported"
    assert "different scenarios" in problems[0]
    assert "cannot be compared" in problems[0]


def test_the_reference_arm_is_included_in_the_coverage_check(tmp_path):
    """It sits at its own budget, so grouping strictly by budget would skip it.

    That would miss the largest gap a stopped run leaves, since the reference arm
    runs first and finishes while the compacted arms are partway through.
    """
    results = list(stream_benchmark(config(), root=tmp_path))
    truncated = [
        r
        for r in results
        if not (r.strategy == "full_context" and r.scenario_id == SMALL[1].scenario_id)
    ]
    problems = check_comparable(truncated)
    assert problems
    assert "full_context" in problems[0]


def test_a_failed_run_does_not_count_as_coverage(tmp_path):
    """A cell that errored measured nothing, so the arm did not cover that scenario."""
    results = list(stream_benchmark(config(), root=tmp_path))
    broken = [
        r.model_copy(update={"status": RunStatus.MODEL_ERROR})
        if (r.strategy == "phase_5_baseline" and r.scenario_id == SMALL[1].scenario_id)
        else r
        for r in results
    ]
    assert check_comparable(broken), "an errored cell is not coverage"


def test_no_results_is_not_a_coverage_problem():
    assert check_comparable([]) == []


# ----------------------------------------------------------------------
# The judge shares the run's budget


def test_judge_requests_are_charged_to_the_same_budget(tmp_path):
    """A judge built from the raw provider would send requests around the ceiling.

    The budget would then silently stop meaning what it says, which is worse than
    having no budget: a run would report having spent 46 of 46 while having sent
    considerably more.
    """
    from open_context_eval.benchmark_dataset_v3 import by_id as v3_by_id
    from open_context_eval.judge import LLMJudge

    counted = provider(reply="YES\nbecause")
    budget = RequestBudget(100)
    wrapped = BudgetedProvider(counted, budget)

    settings = config(
        provider=wrapped,
        scenarios=[v3_by_id("reversed-decision")],
        judge=LLMJudge(wrapped),
    )
    list(stream_benchmark(settings, root=tmp_path))

    assert budget.spent == len(counted.requests), "every request went through the budget"
    assert budget.spent > settings.cells, "judging added requests beyond one per cell"


def test_v3_runs_end_to_end_without_a_judge(tmp_path):
    """Judged questions go unevaluated rather than failing the run."""
    from open_context_eval.benchmark_dataset_v3 import by_id as v3_by_id

    settings = config(scenarios=[v3_by_id("reversed-decision")])
    results = list(stream_benchmark(settings, root=tmp_path))

    assert results and all(r.status is RunStatus.SUCCESS for r in results)
    judged = [outcome for r in results for outcome in r.judged]
    assert judged, "v3 carries judged questions"
    assert all(not outcome.evaluated for outcome in judged)
    assert all(outcome.passed is None for outcome in judged)
