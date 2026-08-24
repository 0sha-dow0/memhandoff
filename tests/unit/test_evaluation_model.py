"""The evaluation data model, the dataset, and the result format.

No archive and no model.
"""

import ast
import inspect
from pathlib import Path

import pytest

import open_context
import open_context_eval
from open_context_eval import (
    DATASET_VERSION,
    SCENARIOS,
    DownstreamTask,
    EvaluationResult,
    EvaluationScenario,
    FailureMode,
    JudgedQuestion,
    KeywordJudge,
    MetricCategory,
    RetentionCheck,
    RetentionOutcome,
    RunMetadata,
    RunStatus,
    TaskKind,
    build_continuation_request,
    by_id,
    format_table,
    pareto_points,
    read_results,
    summarize,
    write_results,
)
from open_context_eval.prompts import CONTINUATION_PROMPT_V1
from open_context_eval.scenario import EvaluationScenario as ScenarioType

# ----------------------------------------------------------------------
# The dependency rule


def modules_of(package) -> list[Path]:
    return sorted(Path(package.__file__).parent.rglob("*.py"))


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_the_runtime_does_not_depend_on_the_evaluation_harness():
    """The direction that must hold: evaluation depends on the runtime, never back.

    A runtime that imported its own benchmark would ship the benchmark, and
    could not be released without it.
    """
    for path in modules_of(open_context):
        offending = {name for name in imports_of(path) if name.startswith("open_context_eval")}
        assert not offending, f"{path.name} imports {offending}"


def test_the_harness_does_depend_on_the_runtime():
    """A guard that the test above is not passing over an empty relationship."""
    imported = {name for path in modules_of(open_context_eval) for name in imports_of(path)}
    assert any(name.startswith("open_context.") for name in imported)


def test_the_harness_is_not_shipped_in_the_runtime_wheel():
    """It lives beside the runtime in the repository, not inside its distribution."""
    pyproject = (Path(open_context.__file__).parents[2] / "pyproject.toml").read_text()
    assert 'packages = ["src/open_context"]' in pyproject
    assert "open_context_eval" not in pyproject


# ----------------------------------------------------------------------
# Scenarios


def a_check(**kwargs):
    defaults = {
        "check_id": "c1",
        "category": MetricCategory.CRITICAL_FACT,
        "description": "d",
        "required": ("x",),
    }
    return RetentionCheck(**{**defaults, **kwargs})


def a_task():
    return DownstreamTask(task_id="t", kind=TaskKind.QUESTION_ANSWERING, instruction="do it")


def test_a_scenario_pairs_a_conversation_with_a_task():
    scenario = EvaluationScenario(
        scenario_id="s",
        dataset_version="v1",
        description="d",
        conversation=[{"role": "user", "content": "hello"}],
        task=a_task(),
        retention_checks=(a_check(),),
    )
    assert scenario.task.instruction == "do it"
    assert scenario.conversation[0]["content"] == "hello"


def test_a_scenario_that_measures_nothing_is_refused():
    """Running it could not tell anyone anything."""
    with pytest.raises(ValueError, match="no checks and no judged questions"):
        EvaluationScenario(
            scenario_id="s",
            dataset_version="v1",
            description="d",
            conversation=[{"role": "user", "content": "hi"}],
            task=a_task(),
        )


def test_a_scenario_needs_a_conversation():
    with pytest.raises(ValueError):
        EvaluationScenario(
            scenario_id="s",
            dataset_version="v1",
            description="d",
            conversation=[],
            task=a_task(),
            retention_checks=(a_check(),),
        )


def test_a_check_must_assert_something():
    with pytest.raises(ValueError, match="asserts nothing"):
        RetentionCheck(check_id="c", category=MetricCategory.DECISION, description="d")


def test_scenarios_are_frozen():
    with pytest.raises(ValueError):
        SCENARIOS[0].scenario_id = "renamed"


# ----------------------------------------------------------------------
# Checks


def test_required_strings_must_all_appear():
    check = a_check(required=("alpha", "beta"))
    assert check.evaluate("alpha and beta")[0]
    passed, problems = check.evaluate("alpha only")
    assert not passed
    assert "missing 'beta'" in problems


def test_required_any_needs_one_spelling():
    check = a_check(required=(), required_any=("45 minutes", "45-minute"))
    assert check.evaluate("within 45-minute budget")[0]
    assert not check.evaluate("within an hour")[0]


def test_forbidden_strings_must_be_absent():
    check = a_check(required=(), forbidden=("redis",))
    assert check.evaluate("use an in-process cache")[0]
    passed, problems = check.evaluate("just use Redis")
    assert not passed
    assert "forbidden 'redis' present" in problems


def test_matching_is_case_insensitive_by_default():
    assert a_check(required=("SQLite",)).evaluate("sqlite works")[0]
    assert not a_check(required=("SQLite",), case_sensitive=True).evaluate("sqlite works")[0]


def test_a_check_combines_required_and_forbidden():
    """The shape a reversed decision needs: take the new one, not the old."""
    check = a_check(required=("sqlite",), forbidden=("postgres",))
    assert check.evaluate("connect with sqlite")[0]
    assert not check.evaluate("connect with sqlite, migrated from postgres")[0]


def test_check_evaluation_is_deterministic():
    check = a_check(required=("a",), forbidden=("b",))
    assert check.evaluate("a and b") == check.evaluate("a and b")


# ----------------------------------------------------------------------
# The dataset


def test_the_dataset_is_versioned_and_consistent():
    assert DATASET_VERSION == "v1"
    assert all(scenario.dataset_version == DATASET_VERSION for scenario in SCENARIOS)


def test_the_dataset_covers_the_failure_modes_it_claims_to():
    """Every trap named in the design has at least one scenario built around it."""
    covered = {mode for scenario in SCENARIOS for mode in scenario.failure_modes}
    for required in (
        FailureMode.EARLY_CRITICAL_FACT,
        FailureMode.LATE_CRITICAL_FACT,
        FailureMode.LONG_IRRELEVANT,
        FailureMode.NEGATIVE_CONSTRAINT,
        FailureMode.REVERSED_DECISION,
        FailureMode.FAILED_APPROACH,
        FailureMode.EXACT_VALUE,
        FailureMode.SIMILAR_ENTITIES,
        FailureMode.TEMPORAL_STATE,
        FailureMode.TOOL_RESULT,
        FailureMode.OPEN_TASK,
        FailureMode.RATIONALE,
        FailureMode.MULTI_STEP,
        FailureMode.REPEATED_INFORMATION,
    ):
        assert required in covered, f"nothing exercises {required.value}"


def test_the_dataset_is_large_enough_to_be_worth_running():
    assert len(SCENARIOS) >= 15
    assert len({scenario.scenario_id for scenario in SCENARIOS}) == len(SCENARIOS)


def test_every_scenario_is_measurable():
    for scenario in SCENARIOS:
        assert scenario.retention_checks or scenario.judged
        assert scenario.task.instruction


def test_scenarios_stay_small_enough_for_ci():
    for scenario in SCENARIOS:
        assert len(scenario.conversation) < 60, scenario.scenario_id


def test_lookup_by_id():
    assert by_id("reversed-decision").scenario_id == "reversed-decision"
    with pytest.raises(KeyError, match="no scenario"):
        by_id("does-not-exist")


def test_the_dataset_contains_no_measured_results():
    """The harness ships machinery, never numbers."""
    source = Path(open_context_eval.__file__).parent
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for claim in ("achieves 9", "% retention", "outperforms", "beats the baseline"):
            assert claim not in text, f"{path.name} appears to state a result"


# ----------------------------------------------------------------------
# Leakage prevention, structurally


def test_the_prompt_builder_cannot_see_a_scenario():
    """The structural guarantee: what is not in scope cannot leak.

    ``build_continuation_request`` takes two strings. It has no access to the
    expected answers, so leaking one is not a mistake anybody can make here.
    """
    signature = inspect.signature(build_continuation_request)
    annotations = [str(parameter.annotation) for parameter in signature.parameters.values()]
    assert not any("Scenario" in annotation for annotation in annotations)
    assert not any("RetentionCheck" in annotation for annotation in annotations)
    assert "context" in signature.parameters
    assert "task_instruction" in signature.parameters


def test_a_request_contains_only_the_context_and_the_task():
    request = build_continuation_request("THE CONTEXT", "THE TASK")
    body = request.messages[1].content

    assert "THE CONTEXT" in body
    assert "THE TASK" in body
    assert body.count("THE CONTEXT") == 1


def test_the_downstream_agent_is_not_told_which_strategy_it_is_running():
    """Blind evaluation: telling a model it holds a summary changes how it answers."""
    text = CONTINUATION_PROMPT_V1.text.lower()
    for tell in ("summary", "compact", "baseline", "strategy", "evaluat"):
        assert tell not in text, f"the continuation prompt mentions {tell!r}"


def test_the_prompt_is_versioned():
    assert CONTINUATION_PROMPT_V1.identifier == "continuation_v1"
    assert len(CONTINUATION_PROMPT_V1.content_hash) == 16


# ----------------------------------------------------------------------
# Results


def metadata(**kwargs):
    defaults = {
        "run_id": "run-1",
        "dataset_version": "v1",
        "provider": "fake",
        "model": "fake-model",
        "continuation_prompt_id": "continuation_v1",
        "continuation_prompt_hash": "abc",
        "target_tokens": 100,
    }
    return RunMetadata(**{**defaults, **kwargs})


def result(**kwargs):
    defaults = {
        "run": metadata(),
        "scenario_id": "s",
        "task_id": "t",
        "task_kind": "question_answering",
        "strategy": "full_context",
        "strategy_version": 1,
    }
    return EvaluationResult(**{**defaults, **kwargs})


def test_compression_ratio_is_original_over_context():
    assert result(original_tokens=50_000, context_tokens=5_000).compression_ratio == 10.0


def test_compression_is_absent_for_a_failed_run():
    """A run that never produced a context has no ratio, not a ratio of zero."""
    failed = result(status=RunStatus.MODEL_ERROR, original_tokens=100, context_tokens=10)
    assert failed.compression_ratio is None
    assert failed.retention_score is None
    assert failed.retention_checks_passed is None


def test_a_failure_is_distinguishable_from_a_score_of_zero():
    scored_zero = result(
        retention_checks=(
            RetentionOutcome(check_id="c", category="decision_retention", passed=False),
        )
    )
    failed = result(status=RunStatus.MODEL_ERROR, error="provider exploded")

    assert scored_zero.retention_score == 0.0
    assert failed.retention_score is None
    assert scored_zero.succeeded and not failed.succeeded


def test_scores_group_by_category():
    outcomes = (
        RetentionOutcome(check_id="a", category="decision_retention", passed=True),
        RetentionOutcome(check_id="b", category="decision_retention", passed=False),
        RetentionOutcome(check_id="c", category="constraint_retention", passed=True),
    )
    scores = result(retention_checks=outcomes).scores_by_category()
    assert scores == {"constraint_retention": 1.0, "decision_retention": 0.5}


def test_results_round_trip_through_jsonl(tmp_path):
    original = [
        result(scenario_id="one", response="a"),
        result(scenario_id="two", strategy="phase_5_baseline", status=RunStatus.CONTEXT_ERROR),
    ]
    path = write_results(tmp_path / "run.jsonl", original)
    restored = list(read_results(path))

    assert restored == original
    assert path.read_text().count("\n") == 2, "one JSON object per line"


def test_each_result_line_is_independently_interpretable(tmp_path):
    """So a results file can be concatenated, filtered, and grepped."""
    import json

    path = write_results(tmp_path / "run.jsonl", [result(), result(scenario_id="other")])
    for line in path.read_text().splitlines():
        parsed = json.loads(line)
        assert parsed["run"]["run_id"] == "run-1"
        assert parsed["run"]["dataset_version"] == "v1"


def test_run_metadata_records_what_is_needed_to_reproduce():
    run = metadata(
        tokenizer="fake-words", context_window=8192, git_commit="abc123", software_version="x"
    )
    for field in (
        "run_id",
        "dataset_version",
        "provider",
        "model",
        "tokenizer",
        "continuation_prompt_id",
        "continuation_prompt_hash",
        "target_tokens",
        "git_commit",
        "software_version",
    ):
        assert getattr(run, field) is not None


# ----------------------------------------------------------------------
# Reporting


def test_a_report_shows_a_dash_where_nothing_was_measured():
    table = format_table([result(status=RunStatus.MODEL_ERROR)])
    assert "0.0" not in table
    assert "model_error" in table


def test_a_report_separates_failures_from_scores():
    table = format_table(
        [
            result(original_tokens=100, context_tokens=50),
            result(scenario_id="b", status=RunStatus.CONTEXT_ERROR),
        ]
    )
    assert "Failures (not counted as zero scores)" in table


def test_summaries_aggregate_per_strategy():
    (summary,) = summarize([result(original_tokens=100, context_tokens=25)])
    assert summary.strategy == "full_context"
    assert summary.mean_compression == 4.0


def test_pareto_points_are_data_not_a_picture():
    points = pareto_points([result(original_tokens=100, context_tokens=50)])
    assert points[0]["strategy"] == "full_context"
    assert "mean_context_tokens" in points[0]
    assert "retention_score" in points[0]


# ----------------------------------------------------------------------
# Judge


def test_a_judged_question_without_a_judge_is_not_scored():
    question = JudgedQuestion(
        question_id="q", category=MetricCategory.RATIONALE, question="does it explain why?"
    )
    assert question.question_id == "q"


def test_the_keyword_judge_is_deterministic():
    judge = KeywordJudge(answers={"why?": ("mainframe",)})
    assert judge.judge("because the mainframe cannot parse it", "why?").passed
    assert not judge.judge("because it is nicer", "why?").passed


def test_the_keyword_judge_refuses_questions_it_has_no_rule_for():
    verdict = KeywordJudge().judge("anything", "unseen question")
    assert not verdict.passed
    assert "no rule" in verdict.reasoning


def test_scenario_type_is_exported():
    assert ScenarioType is EvaluationScenario
