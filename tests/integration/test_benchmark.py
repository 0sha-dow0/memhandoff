"""The first benchmark.

Offline and deterministic throughout. These tests validate the benchmark
machinery; they do not measure any strategy, and neither does a deterministic
benchmark run.
"""

import pytest

from open_context.archive import Archive
from open_context.llm import RateLimitError
from open_context.llm.fakes import FailingProvider, FakeProvider, WordTokenizer
from open_context_eval import (
    BENCHMARK_DATASET_VERSION,
    BENCHMARK_SCENARIOS,
    BENCHMARK_STRATEGIES,
    DEFAULT_BUDGETS,
    DETERMINISTIC_TEST,
    FULL_CONTEXT,
    HYBRID_V1,
    PHASE_5_BASELINE,
    REAL_MODEL,
    REFERENCE_BUDGET,
    SIMPLE_SUMMARY_V1,
    BenchmarkConfig,
    CompletionCriterion,
    CompletionKind,
    CompletionStatus,
    InvalidBenchmarkError,
    RunStatus,
    build_session,
    check_comparable,
    experiment_configs,
    failure_modes_table,
    format_benchmark,
    format_failures,
    read_results,
    run_benchmark,
    run_id_from_fingerprint,
    scenario_completed,
    summarize_benchmark,
    write_results,
)
from open_context_eval.benchmark_dataset import CRITERIA, by_id
from open_context_eval.dataset import DATASET_VERSION
from open_context_eval.dataset import SCENARIOS as V1_SCENARIOS
from open_context_eval.scenario import FailureMode

pytestmark = pytest.mark.integration

SMALL = [by_id("reversed-decision"), by_id("negative-constraint")]


def provider(reply="continuing the work as described in the context provided above", **kwargs):
    return FakeProvider(reply=reply, context_window=kwargs.pop("context_window", 100_000), **kwargs)


def config(**kwargs):
    defaults = {
        "provider": provider(),
        "tokenizer": WordTokenizer(exact=True),
        "budgets": (80, 160),
        "scenarios": SMALL,
    }
    return BenchmarkConfig(**{**defaults, **kwargs})


# ----------------------------------------------------------------------
# Dataset


def test_the_benchmark_dataset_is_a_new_version_not_an_edit_of_v1():
    """v1 stays exactly as Phase 5.5 closed it."""
    assert BENCHMARK_DATASET_VERSION == "v2" != DATASET_VERSION
    assert all(s.dataset_version == DATASET_VERSION for s in V1_SCENARIOS)
    assert all(s.completion == () for s in V1_SCENARIOS), "v1 gained no criteria"


def test_v2_carries_v1s_conversations_verbatim():
    for original, upgraded in zip(V1_SCENARIOS, BENCHMARK_SCENARIOS, strict=True):
        assert upgraded.scenario_id == original.scenario_id
        assert upgraded.conversation == original.conversation
        assert upgraded.task == original.task
        assert upgraded.retention_checks == original.retention_checks


def test_every_benchmark_scenario_has_completion_criteria():
    for scenario in BENCHMARK_SCENARIOS:
        assert scenario.completion, scenario.scenario_id
        assert scenario.dataset_version == BENCHMARK_DATASET_VERSION


def test_the_benchmark_covers_every_named_failure_mode():
    """All seventeen conditions the benchmark is supposed to exercise."""
    covered = {mode for scenario in BENCHMARK_SCENARIOS for mode in scenario.failure_modes}
    assert covered == set(FailureMode), f"missing {set(FailureMode) - covered}"


def test_criteria_discriminate_rather_than_merely_match():
    """Most criteria name what the conversation ruled out, not just the answer."""
    with_alternatives = [c for group in CRITERIA.values() for c in group if c.alternatives]
    assert len(with_alternatives) >= 10


# ----------------------------------------------------------------------
# Matrix


def test_the_matrix_separates_reference_arms_from_budgeted_ones():
    """scenarios x repetitions x (references + compacted x budgets)."""
    plan = config(budgets=(80, 160, 240), repetitions=2)

    assert plan.reference_strategies == [FULL_CONTEXT]
    assert plan.compacted_strategies == [SIMPLE_SUMMARY_V1, PHASE_5_BASELINE, HYBRID_V1]
    assert plan.cells == 2 * 2 * (1 + len(plan.compacted_strategies) * 3)
    assert "1 reference + 3 compacted x 3 budgets" in plan.describe()


def test_the_default_matrix_size_follows_from_the_arms():
    """15 scenarios, one repetition, one reference plus every compacted arm at three budgets.

    Written against the arm list rather than a constant, because Phase 8 added a
    fourth arm and a hardcoded 105 says nothing about why it was 105.
    """
    plan = config(budgets=DEFAULT_BUDGETS, scenarios=BENCHMARK_SCENARIOS)
    compacted = len(plan.compacted_strategies)
    assert plan.cells == 15 * 1 * (1 + compacted * 3)


def test_a_reference_arm_runs_once_per_repetition_not_once_per_budget():
    plans = list(experiment_configs(config(budgets=(80, 160, 240))))
    reference = [p for p in plans if list(p[2].strategies) == [FULL_CONTEXT]]

    assert len(reference) == 1, "one reference experiment, whatever the budgets"
    assert reference[0][0] == REFERENCE_BUDGET


def test_compacted_arms_run_once_per_budget_and_repetition():
    plans = list(experiment_configs(config(budgets=(80, 160), repetitions=2)))
    compacted = [(budget, rep) for budget, rep, e in plans if FULL_CONTEXT not in e.strategies]

    assert sorted(compacted) == [(80, 1), (80, 2), (160, 1), (160, 2)]


def test_adding_budgets_adds_no_reference_runs(tmp_path):
    """The correction, stated directly."""
    few = run_benchmark(config(budgets=(80,)), root=tmp_path / "a")
    many = run_benchmark(config(budgets=(80, 160, 240)), root=tmp_path / "b")

    def references(results):
        return [r for r in results if r.strategy == FULL_CONTEXT]

    assert len(references(few)) == len(references(many)) == len(SMALL)
    assert len(many) > len(few), "only the compacted arms multiplied"


def test_a_reference_result_records_no_budget(tmp_path):
    """It never consulted one, so claiming a budget would be false."""
    results = run_benchmark(config(budgets=(80, 160)), root=tmp_path)
    for result in results:
        if result.strategy == FULL_CONTEXT:
            assert result.run.target_tokens == REFERENCE_BUDGET
        else:
            assert result.run.target_tokens in {80, 160}


def test_repetitions_create_separate_reference_samples(tmp_path):
    """Not one reference averaged across budgets: one draw per repetition."""
    results = run_benchmark(config(budgets=(80, 160), repetitions=3), root=tmp_path)
    reference = [r for r in results if r.strategy == FULL_CONTEXT]

    assert len(reference) == len(SMALL) * 3
    assert len({r.run.fingerprint["configuration"]["benchmark_repetition"] for r in reference}) == 3
    assert len({r.run.run_id for r in reference}) == 3, "one run per repetition"


def test_repetitions_are_distinguishable_in_the_results(tmp_path):
    """Two runs of one configuration must not share an identity."""
    results = run_benchmark(config(budgets=(80,), repetitions=2), root=tmp_path)
    compacted = [r for r in results if r.strategy != FULL_CONTEXT]

    assert len({r.run.run_id for r in compacted}) == 2
    assert {r.run.fingerprint["configuration"]["benchmark_repetition"] for r in results} == {1, 2}


def test_every_budget_produces_its_own_run_id(tmp_path):
    results = run_benchmark(config(budgets=(80, 160, 240)), root=tmp_path)
    compacted = {r.run.target_tokens: r.run.run_id for r in results if r.strategy != FULL_CONTEXT}

    assert set(compacted) == {80, 160, 240}
    assert len(set(compacted.values())) == 3


def test_the_whole_matrix_runs(tmp_path):
    plan = config(budgets=DEFAULT_BUDGETS, scenarios=BENCHMARK_SCENARIOS)
    results = run_benchmark(plan, root=tmp_path)

    assert plan.cells == 15 * (1 + len(plan.compacted_strategies) * 3)
    assert all(result.succeeded for result in results)
    assert {r.strategy for r in results} == set(BENCHMARK_STRATEGIES)

    counts = {name: sum(1 for r in results if r.strategy == name) for name in BENCHMARK_STRATEGIES}
    assert counts == {
        FULL_CONTEXT: 15,
        SIMPLE_SUMMARY_V1: 45,
        PHASE_5_BASELINE: 45,
        HYBRID_V1: 45,
    }


def test_a_benchmark_of_only_the_reference_needs_no_budgets(tmp_path):
    results = run_benchmark(config(strategies=(FULL_CONTEXT,), budgets=(80, 160)), root=tmp_path)
    assert len(results) == len(SMALL), "budgets do not multiply a budget-independent arm"


# ----------------------------------------------------------------------
# Fairness


def test_every_strategy_sees_the_same_scenario_and_task(tmp_path):
    results = run_benchmark(config(budgets=(80,)), root=tmp_path)
    for scenario_id in {r.scenario_id for r in results}:
        arms = [r for r in results if r.scenario_id == scenario_id]
        assert len(arms) == len(BENCHMARK_STRATEGIES), "the reference plus every compacted arm"
        assert len({r.task_id for r in arms}) == 1
        assert len({r.original_tokens for r in arms}) == 1, "the same conversation went in"


def test_each_budget_is_compared_against_the_same_reference(tmp_path):
    """The reference population does not move with the budget."""
    results = run_benchmark(config(budgets=(80, 160, 240)), root=tmp_path)
    reference = [r for r in results if r.strategy == FULL_CONTEXT]

    assert len(reference) == len(SMALL)
    for budget in (80, 160, 240):
        compacted = [r for r in results if r.run.target_tokens == budget]
        assert {r.scenario_id for r in compacted} == {r.scenario_id for r in reference}


def test_every_arm_uses_the_same_continuation_prompt(tmp_path):
    results = run_benchmark(config(budgets=(80, 160)), root=tmp_path)
    assert len({r.run.continuation_prompt_id for r in results}) == 1
    assert len({r.run.continuation_prompt_hash for r in results}) == 1


def test_every_arm_uses_the_same_downstream_model(tmp_path):
    results = run_benchmark(config(budgets=(80, 160)), root=tmp_path)
    assert len({(r.run.provider, r.run.model) for r in results}) == 1
    assert check_comparable(results) == []


def test_a_mixed_model_comparison_is_called_invalid(tmp_path):
    """Not silently presented as fair."""
    one = run_benchmark(config(budgets=(80,)), root=tmp_path / "a")
    other = run_benchmark(
        config(budgets=(80,), provider=provider(model="a-different-model")), root=tmp_path / "b"
    )

    problems = check_comparable([*one, *other])
    assert problems
    assert "more than one downstream model" in problems[0]


def test_mixed_evaluation_modes_are_called_invalid(tmp_path):
    fake = run_benchmark(config(budgets=(80,)), root=tmp_path / "a")
    labelled_real = run_benchmark(
        config(budgets=(80,), evaluation_mode=REAL_MODEL), root=tmp_path / "b"
    )
    problems = check_comparable([*fake, *labelled_real])
    assert any("evaluation modes" in problem for problem in problems)


def test_compacted_arms_share_the_budget_and_full_context_does_not(tmp_path):
    results = run_benchmark(config(budgets=(80,)), root=tmp_path)
    compacted = [r for r in results if r.strategy != FULL_CONTEXT]
    reference = [r for r in results if r.strategy == FULL_CONTEXT]

    assert all(r.context_tokens <= 80 for r in compacted), "held to the same budget"
    assert all(r.context_tokens > 80 for r in reference), "the reference is not truncated to fit"


# ----------------------------------------------------------------------
# Leakage


def test_full_context_receives_the_conversation_and_the_others_do_not(tmp_path):
    model = provider()
    run_benchmark(
        config(budgets=(80,), provider=model, scenarios=[by_id("reversed-decision")]),
        root=tmp_path,
    )

    continuations = [
        request.messages[1].content
        for request in model.requests
        if "YOUR TASK" in request.messages[1].content
    ]
    assert len(continuations) == len(BENCHMARK_STRATEGIES)

    distinctive = "It matches the existing stack"
    full = [body for body in continuations if distinctive in body]
    assert len(full) == 1, "only the reference condition carries original wording"


def test_no_arm_receives_another_arms_representation(tmp_path):
    model = provider(reply="THE-SUMMARY-MARKER continuing the work described above")
    run_benchmark(
        config(budgets=(80,), provider=model, scenarios=[by_id("negative-constraint")]),
        root=tmp_path,
    )

    continuations = [
        request.messages[1].content
        for request in model.requests
        if "YOUR TASK" in request.messages[1].content
    ]
    carrying_summary = [body for body in continuations if "THE-SUMMARY-MARKER" in body]
    assert len(carrying_summary) == 3, "only the arms that summarise carry a summary"


def test_no_expected_answer_reaches_the_model(tmp_path):
    model = provider()
    run_benchmark(
        config(budgets=(80,), provider=model, scenarios=BENCHMARK_SCENARIOS), root=tmp_path
    )

    bodies = " ".join(m.content for request in model.requests for m in request.messages)
    for scenario in BENCHMARK_SCENARIOS:
        for criterion in scenario.completion:
            assert criterion.description not in bodies, scenario.scenario_id
            assert criterion.criterion_id not in bodies


def test_the_model_is_never_told_which_strategy_produced_its_context(tmp_path):
    model = provider()
    run_benchmark(config(budgets=(80,), provider=model), root=tmp_path)

    bodies = " ".join(m.content for request in model.requests for m in request.messages).lower()
    for name in BENCHMARK_STRATEGIES:
        assert name not in bodies


# ----------------------------------------------------------------------
# Task completion


def criterion(**kwargs):
    defaults = {
        "criterion_id": "c",
        "kind": CompletionKind.EXPECTED_DECISION,
        "description": "d",
        "expected": ("sqlite",),
    }
    return CompletionCriterion(**{**defaults, **kwargs})


def test_completion_needs_the_right_answer():
    assert criterion().evaluate("use sqlite here")[0] is CompletionStatus.PASSED
    assert criterion().evaluate("use something else")[0] is CompletionStatus.OMITTED


def test_completion_is_a_choice_not_a_mention():
    """Naming the answer alongside the option that was ruled out is not success."""
    discriminating = criterion(alternatives=("postgres",))
    assert discriminating.evaluate("use sqlite")[0] is CompletionStatus.PASSED
    assert (
        discriminating.evaluate("use sqlite, or postgres if you prefer")[0]
        is CompletionStatus.FABRICATED
    )


def test_a_fabrication_is_distinguished_from_an_omission():
    """Different failures with different causes: asserting versus forgetting."""
    check = criterion(alternatives=("kafka",))
    assert check.evaluate("use kafka")[0] is CompletionStatus.FABRICATED
    assert check.evaluate("no idea")[0] is CompletionStatus.OMITTED


def test_a_substance_floor_rejects_a_one_word_answer():
    check = criterion(min_words=10)
    assert check.evaluate("sqlite")[0] is CompletionStatus.INSUBSTANTIAL
    assert check.evaluate("use sqlite " + "and connect to it locally right here now")[0] is (
        CompletionStatus.PASSED
    )


def test_a_criterion_needing_execution_is_not_evaluable():
    """A code task is marked as unmeasurable rather than guessed either way."""
    check = criterion(kind=CompletionKind.EXPECTED_CODE_CHANGE, requires_execution=True)
    status, detail = check.evaluate("anything at all")
    assert status is CompletionStatus.NOT_EVALUABLE
    assert "code execution" in detail


def test_completion_is_conjunctive():
    assert scenario_completed((CompletionStatus.PASSED, CompletionStatus.PASSED)) is True
    assert scenario_completed((CompletionStatus.PASSED, CompletionStatus.OMITTED)) is False
    assert scenario_completed(()) is None


def test_an_unevaluable_criterion_makes_the_scenario_unscored():
    assert scenario_completed((CompletionStatus.PASSED, CompletionStatus.NOT_EVALUABLE)) is None


def test_completion_and_retention_are_reported_separately(tmp_path):
    """A response can retain the fact and still not do the task."""
    model = provider(reply="I recall a mention of sqlite somewhere in the earlier discussion")
    results = run_benchmark(
        config(budgets=(80,), provider=model, scenarios=[by_id("reversed-decision")]),
        root=tmp_path,
    )
    for result in results:
        assert result.retention_score is not None
        assert result.task_completed is not None
        assert result.retention_checks and result.completion


def test_a_correct_continuation_completes_the_task(tmp_path):
    answer = "Set up the sqlite connection with a local file path and a short busy timeout here."
    results = run_benchmark(
        config(
            budgets=(80,), provider=provider(reply=answer), scenarios=[by_id("reversed-decision")]
        ),
        root=tmp_path,
    )
    assert all(result.task_completed for result in results)


def test_a_wrong_continuation_does_not(tmp_path):
    answer = "Set up the postgresql connection with a pool and a statement timeout as before."
    results = run_benchmark(
        config(
            budgets=(80,), provider=provider(reply=answer), scenarios=[by_id("reversed-decision")]
        ),
        root=tmp_path,
    )
    assert all(result.task_completed is False for result in results)
    assert all(result.deterministic_hallucination_checks == 1 for result in results)


# ----------------------------------------------------------------------
# Cost


def test_compaction_and_continuation_cost_are_recorded_separately(tmp_path):
    from open_context.llm import TokenUsage

    model = provider(usage=TokenUsage(input_tokens=500, output_tokens=40))
    results = run_benchmark(config(budgets=(80,), provider=model), root=tmp_path)

    baseline = next(r for r in results if r.strategy == PHASE_5_BASELINE)
    assert baseline.compaction_llm_calls >= 1
    assert baseline.continuation_llm_calls == 1
    assert baseline.compaction_input_tokens == 500
    assert baseline.continuation_input_tokens == 500
    assert baseline.total_tokens == 1080, "compaction is not free and is not hidden"


def test_the_reference_condition_costs_no_compaction(tmp_path):
    results = run_benchmark(config(budgets=(80,)), root=tmp_path)
    reference = next(r for r in results if r.strategy == FULL_CONTEXT)

    assert reference.compaction_llm_calls == 0
    assert reference.compression_ratio == pytest.approx(1.0, abs=0.01)


def test_unreported_usage_is_absent_rather_than_zero(tmp_path):
    results = run_benchmark(config(budgets=(80,)), root=tmp_path)
    assert all(r.compaction_input_tokens is None for r in results)
    assert all(r.total_tokens is None for r in results)


def test_compression_rises_as_the_budget_falls(tmp_path):
    results = run_benchmark(config(budgets=(80, 240), scenarios=SMALL), root=tmp_path)
    tight = [r for r in results if r.strategy == PHASE_5_BASELINE and r.run.target_tokens == 80]
    loose = [r for r in results if r.strategy == PHASE_5_BASELINE and r.run.target_tokens == 240]

    tight_ratios = [r.compression_ratio for r in tight if r.compression_ratio is not None]
    loose_ratios = [r.compression_ratio for r in loose if r.compression_ratio is not None]
    assert tight_ratios and loose_ratios
    assert min(tight_ratios) > max(loose_ratios)


def test_latency_is_split_between_compaction_and_continuation(tmp_path):
    results = run_benchmark(config(budgets=(80,)), root=tmp_path)
    for result in results:
        assert result.compaction_latency_seconds is not None
        assert result.continuation_latency_seconds is not None
        assert result.latency_seconds is not None
        assert result.latency_seconds >= result.continuation_latency_seconds


# ----------------------------------------------------------------------
# Failures and modes


def test_a_model_failure_is_recorded_across_the_matrix(tmp_path):
    results = run_benchmark(
        config(budgets=(80,), provider=FailingProvider(RateLimitError("slow down"))), root=tmp_path
    )
    assert results
    assert all(r.status is RunStatus.MODEL_ERROR for r in results)
    assert all(r.task_completed is None for r in results), "a failure is not an incomplete task"


def test_an_unavailable_reference_condition_is_not_a_zero(tmp_path):
    results = run_benchmark(
        config(budgets=(80,), provider=provider(context_window=5)), root=tmp_path
    )
    reference = [r for r in results if r.strategy == FULL_CONTEXT]

    assert all(r.status is RunStatus.CONTEXT_ERROR for r in reference)
    assert all(r.retention_score is None for r in reference)


def test_failures_are_reported_per_scenario_and_check(tmp_path):
    results = run_benchmark(
        config(
            budgets=(80,), provider=provider(reply="I have no idea what to do next here at all")
        ),
        root=tmp_path,
    )
    report = format_failures(results)

    assert "retention" in report
    assert "completion" in report
    assert "reversed-decision" in report


def test_failure_modes_are_grouped_for_analysis(tmp_path):
    results = run_benchmark(config(budgets=(80,), scenarios=BENCHMARK_SCENARIOS), root=tmp_path)
    table = failure_modes_table(results)

    assert "reversed_decision" in table
    assert PHASE_5_BASELINE in table


# ----------------------------------------------------------------------
# Reporting


def test_the_report_keeps_retention_and_completion_apart(tmp_path):
    results = run_benchmark(config(budgets=(80, 160)), root=tmp_path)
    report = format_benchmark(results)

    assert "Retention" in report and "Completion" in report
    assert "Not the same measure" in report


def test_a_deterministic_report_says_it_measures_nothing(tmp_path):
    results = run_benchmark(config(budgets=(80,)), root=tmp_path)
    report = format_benchmark(results)

    assert "deterministic_test" in report
    assert "say nothing about any strategy's" in report


def test_the_reference_condition_is_labelled_not_budgeted(tmp_path):
    results = run_benchmark(config(budgets=(80, 160, 240)), root=tmp_path)
    cells = summarize_benchmark(results)
    reference = [cell for cell in cells if cell.strategy == FULL_CONTEXT]

    assert len(reference) == 1, "the reference does not vary with the budget"
    assert "ref" in format_benchmark(results)


def test_a_report_of_nothing_says_so():
    assert format_benchmark([]) == "no results"


# ----------------------------------------------------------------------
# Configuration and persistence


@pytest.mark.parametrize(
    ("label", "changed"),
    [
        ("no budgets", {"budgets": ()}),
        ("negative budget", {"budgets": (-1,)}),
        ("no repetitions", {"repetitions": 0}),
        ("no strategies", {"strategies": ()}),
        ("unknown strategy", {"strategies": ("memhandoff",)}),
        ("no scenarios", {"scenarios": []}),
        ("unknown mode", {"evaluation_mode": "vibes"}),
    ],
)
def test_invalid_configurations_are_refused(label, changed):
    with pytest.raises(InvalidBenchmarkError):
        config(**changed)


def test_mixing_dataset_versions_is_refused():
    with pytest.raises(InvalidBenchmarkError, match="dataset versions"):
        config(scenarios=[BENCHMARK_SCENARIOS[0], V1_SCENARIOS[0]])


def test_results_persist_and_reload_with_their_fingerprints(tmp_path):
    results = run_benchmark(config(budgets=(80, 160)), root=tmp_path)
    restored = list(read_results(write_results(tmp_path / "bench.jsonl", results)))

    assert restored == results
    for result in restored:
        assert run_id_from_fingerprint(result.run.fingerprint) == result.run.run_id
        assert result.run.dataset_version == BENCHMARK_DATASET_VERSION
        assert result.evaluation_mode == DETERMINISTIC_TEST


def test_the_evaluation_mode_is_part_of_the_experiment_identity(tmp_path):
    fake = run_benchmark(config(budgets=(80,)), root=tmp_path / "a")[0]
    real = run_benchmark(config(budgets=(80,), evaluation_mode=REAL_MODEL), root=tmp_path / "b")[0]

    assert fake.run.run_id != real.run.run_id
    assert fake.run.fingerprint["evaluation_mode"] == DETERMINISTIC_TEST


# ----------------------------------------------------------------------
# Isolation


def test_repeating_the_benchmark_does_not_duplicate_scenario_events(tmp_path):
    plan = config(budgets=(80, 160), repetitions=2)
    run_benchmark(plan, root=tmp_path)

    archive = Archive(tmp_path / "archive")
    for session in archive.sessions():
        log = archive.open(session)
        scenario = next(s for s in plan.scenarios if len(s.conversation) == log.count)
        assert log.count == len(scenario.conversation)
        assert log.verify().ok


def test_the_benchmark_does_not_touch_a_production_archive(tmp_path):
    import hashlib

    production = Archive(tmp_path / "production")
    log = build_session(production, by_id("reversed-decision"))
    before = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(production.root.rglob("*")) if path.is_file())
    ).hexdigest()

    run_benchmark(config(budgets=(80,)), root=tmp_path / "benchmark")

    after = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(production.root.rglob("*")) if path.is_file())
    ).hexdigest()
    assert after == before
    assert log.verify().ok


def test_no_benchmark_result_is_committed_to_the_repository():
    """The one thing this phase must never do.

    Running a benchmark writes into ``benchmarks/results``, which is ignored, so
    the check is that artifacts land only in ignored directories — not that no
    artifact exists. A committed number nobody can reproduce is worse than none.
    """
    from pathlib import Path

    import open_context_eval

    root = Path(open_context_eval.__file__).parents[2]
    ignored = (root / "benchmarks" / ".gitignore").read_text()
    allowed_roots = {"data", "reports", "results"}
    assert all(f"{directory}/" in ignored for directory in allowed_roots)

    stray = [
        path
        for path in (root / "benchmarks").rglob("*.jsonl")
        if path.relative_to(root / "benchmarks").parts[0] not in allowed_roots
    ]
    assert stray == [], f"benchmark results outside the ignored directories: {stray}"


def test_no_unsupported_superiority_claim_appears_in_documentation():
    """A measured result may be stated. A superiority claim may not.

    Phase 5.6 forbade any number, because none existed. Phase 5.8 produced a
    real one, so the rule tightens rather than lifts: documentation may report
    what was measured, and may say a strategy did *not* beat another — that was
    the actual finding — but must never assert that one beats another.

    The distinction is the whole point. "The baseline does not beat a plain
    summary" is a result; "the baseline beats a plain summary" would be a claim
    this project has no evidence for and, on the run it did execute, evidence
    against.
    """
    import re
    from pathlib import Path

    import open_context_eval

    root = Path(open_context_eval.__file__).parents[2]
    claim = re.compile(
        r"(memhandoff|phase.5|simple.summary|baseline)[^.\n]{0,40}?"
        r"(achieves|scores|reaches|outperforms|beats)",
        re.IGNORECASE,
    )
    negated = ("not ", "never ", "no evidence", "cannot", "n't ", "without ")
    for document in [root / "README.md", *(root / "docs").glob("*.md")]:
        text = document.read_text()
        for match in claim.finditer(text):
            window = text[max(0, match.start() - 60) : match.end()].lower()
            assert any(mark in window for mark in negated), (
                f"{document.name} asserts superiority: {match.group(0)!r}"
            )
