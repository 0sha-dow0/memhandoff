"""The evaluation harness end to end.

Offline and deterministic throughout: the Phase 4.5 fakes are the whole model
surface. No key, no network, no model server, no database.
"""

import hashlib

import pytest

from open_context.archive import Archive
from open_context.compaction import Prompt
from open_context.llm import ModelInfo, RateLimitError, UnavailableTokenizer
from open_context.llm.fakes import FailingProvider, FakeProvider, WordTokenizer
from open_context_eval import (
    DATASET_VERSION,
    FULL_CONTEXT,
    PHASE_5_BASELINE,
    SCENARIOS,
    SIMPLE_SUMMARY_V1,
    DownstreamTask,
    EvaluationConfig,
    EvaluationScenario,
    ExperimentRunner,
    FullContextStrategy,
    KeywordJudge,
    MetricCategory,
    Phase5BaselineStrategy,
    RetentionCheck,
    RunStatus,
    StrategyContext,
    TaskKind,
    build_session,
    by_id,
    default_strategies,
    read_results,
    run_fingerprint,
    run_id_for,
    run_id_from_fingerprint,
    write_results,
)
from open_context_eval.dataset import filler
from open_context_eval.prompts import CONTEXT_HEADER, TASK_HEADER
from open_context_eval.runner import ScenarioMaterializationError, session_id_for

pytestmark = pytest.mark.integration


@pytest.fixture
def tokenizer():
    return WordTokenizer(exact=True)


def provider(reply="a continuation that mentions nothing in particular", **kwargs):
    return FakeProvider(reply=reply, context_window=kwargs.pop("context_window", 100_000), **kwargs)


def config(**kwargs):
    defaults = {
        "provider": provider(),
        "tokenizer": WordTokenizer(exact=True),
        "target_tokens": 300,
    }
    return EvaluationConfig(**{**defaults, **kwargs})


def runner(tmp_path, **kwargs):
    return ExperimentRunner(config(**kwargs), root=tmp_path / "eval-archive")


def scenario(**kwargs):
    defaults = {
        "scenario_id": "test-scenario",
        "dataset_version": DATASET_VERSION,
        "description": "a test scenario",
        "conversation": [
            {"role": "user", "content": "The secret marker is ZEBRAFISH and it matters."},
            *filler(6),
            {"role": "user", "content": "Carry on."},
        ],
        "task": DownstreamTask(
            task_id="t", kind=TaskKind.QUESTION_ANSWERING, instruction="What is the marker?"
        ),
        "retention_checks": (
            RetentionCheck(
                check_id="marker",
                category=MetricCategory.CRITICAL_FACT,
                description="the marker survives",
                required=("zebrafish",),
            ),
        ),
    }
    return EvaluationScenario(**{**defaults, **kwargs})


# ----------------------------------------------------------------------
# Running


def test_a_scenario_becomes_an_archived_session(tmp_path):
    archive = Archive(tmp_path / "a")
    log = build_session(archive, scenario())

    assert log.count == len(scenario().conversation)
    assert log.verify().ok


def test_every_strategy_runs_on_every_scenario(tmp_path):
    results = runner(tmp_path).run([scenario(), scenario(scenario_id="second")])

    assert len(results) == 2 * len(default_strategies())
    assert {r.strategy for r in results} == set(default_strategies())
    assert {r.scenario_id for r in results} == {"test-scenario", "second"}
    assert all(r.succeeded for r in results)


def test_a_selected_subset_of_strategies_runs(tmp_path):
    results = runner(tmp_path, strategies=[FULL_CONTEXT]).run([scenario()])
    assert [r.strategy for r in results] == [FULL_CONTEXT]


def test_an_unregistered_strategy_is_refused(tmp_path):
    with pytest.raises(KeyError, match="no strategy registered"):
        runner(tmp_path, strategies=["memhandoff"]).run([scenario()])


def test_the_dataset_runs_offline_end_to_end(tmp_path):
    results = runner(tmp_path).run(SCENARIOS)

    assert len(results) == len(SCENARIOS) * len(default_strategies())
    assert all(r.succeeded for r in results)
    assert all(r.run.dataset_version == DATASET_VERSION for r in results)


# ----------------------------------------------------------------------
# Strategies


def test_full_context_gives_the_whole_conversation(tmp_path, tokenizer):
    archive = Archive(tmp_path / "a")
    log = build_session(archive, scenario())
    prepared = FullContextStrategy().prepare(
        StrategyContext(log=log, provider=provider(), tokenizer=tokenizer, target_tokens=50)
    )

    assert prepared.available
    assert "ZEBRAFISH" in prepared.text
    assert prepared.llm_calls == 0, "the reference condition calls no model"


def test_full_context_refuses_rather_than_truncating(tmp_path, tokenizer):
    """A truncated reference condition is a different strategy wearing its name."""
    archive = Archive(tmp_path / "a")
    log = build_session(archive, scenario())
    prepared = FullContextStrategy().prepare(
        StrategyContext(
            log=log,
            provider=provider(context_window=5),
            tokenizer=tokenizer,
            target_tokens=50,
        )
    )

    assert not prepared.available
    assert prepared.text == "", "nothing was silently cut down to fit"
    assert "cannot be run" in prepared.unavailable_reason


def test_an_unavailable_full_context_is_a_context_error_not_a_zero(tmp_path):
    results = runner(tmp_path, provider=provider(context_window=5)).run([scenario()])
    full = next(r for r in results if r.strategy == FULL_CONTEXT)

    assert full.status is RunStatus.CONTEXT_ERROR
    assert full.retention_score is None
    assert full.response == ""


def test_the_phase_5_baseline_is_used_through_its_public_interface(tmp_path, tokenizer):
    archive = Archive(tmp_path / "a")
    log = build_session(archive, scenario())
    strategy = Phase5BaselineStrategy()
    prepared = strategy.prepare(
        StrategyContext(log=log, provider=provider(), tokenizer=tokenizer, target_tokens=60)
    )

    assert strategy.name == PHASE_5_BASELINE
    assert prepared.available
    assert "compaction_needed" in prepared.detail
    assert prepared.detail["prompt_id"] == "baseline_summary_v1"


def test_the_baseline_strategy_reports_its_compaction_detail(tmp_path):
    results = runner(tmp_path, target_tokens=60).run([scenario()])
    baseline = next(r for r in results if r.strategy == PHASE_5_BASELINE)

    assert "fallbacks" in baseline.strategy_detail
    assert "dropped_recent_event_count" in baseline.strategy_detail
    assert baseline.compaction_llm_calls >= 1


def test_simple_summary_is_one_call_and_keeps_nothing_verbatim(tmp_path):
    results = runner(tmp_path).run([scenario()])
    summary = next(r for r in results if r.strategy == SIMPLE_SUMMARY_V1)

    assert summary.compaction_llm_calls == 1
    assert summary.succeeded


def test_strategy_metadata_is_recorded(tmp_path):
    results = runner(tmp_path).run([scenario()])
    for result in results:
        assert result.strategy in default_strategies()
        assert result.strategy_version >= 1
        assert result.task_id == "t"
        assert result.task_kind == "question_answering"


# ----------------------------------------------------------------------
# Leakage
#
# The harness must not hand the downstream agent anything the strategy was
# supposed to remove, nor anything only the evaluator is meant to know.


def test_the_downstream_prompt_is_exactly_context_plus_task(tmp_path, tokenizer):
    """Nothing else can be in there, which is what makes leakage structural."""
    model = provider()
    ExperimentRunner(
        config(provider=model, tokenizer=tokenizer, strategies=[FULL_CONTEXT]),
        root=tmp_path / "a",
    ).run([scenario()])

    archive = Archive(tmp_path / "b")
    log = build_session(archive, scenario())
    prepared = FullContextStrategy().prepare(
        StrategyContext(log=log, provider=model, tokenizer=tokenizer, target_tokens=300)
    )

    downstream = model.requests[-1]
    expected = f"{CONTEXT_HEADER}\n\n{prepared.text}\n\n{TASK_HEADER}\n\nWhat is the marker?"
    assert downstream.messages[1].content == expected


def test_the_original_conversation_does_not_reach_a_compacted_arm(tmp_path):
    """The summary arm condenses everything, so no original turn may survive."""
    model = provider(reply="A condensed handover with no original wording.")
    ExperimentRunner(
        config(provider=model, strategies=[SIMPLE_SUMMARY_V1]), root=tmp_path / "a"
    ).run([scenario()])

    downstream = model.requests[-1].messages[1].content
    assert "ZEBRAFISH" not in downstream
    assert "Small question 0 about logging formatting." not in downstream


def test_expected_answers_never_reach_the_model(tmp_path):
    """A check description is evaluator-only and must not appear in any request."""
    model = provider()
    subject = scenario()
    ExperimentRunner(config(provider=model), root=tmp_path / "a").run([subject])

    for request in model.requests:
        body = " ".join(message.content for message in request.messages)
        for check in subject.retention_checks:
            assert check.description not in body
        for word in ("required_any", "check_id", "forbidden", "MetricCategory"):
            assert word not in body


def test_a_value_a_check_looks_for_may_appear_because_the_conversation_said_it(tmp_path):
    """The distinction leakage tests have to get right.

    "ZEBRAFISH" is both what a check requires and what the conversation says, so
    finding it in a full-context prompt is the conversation arriving, not an
    expectation escaping. Only the evaluator's own words are forbidden.
    """
    model = provider()
    ExperimentRunner(config(provider=model, strategies=[FULL_CONTEXT]), root=tmp_path / "a").run(
        [scenario()]
    )

    body = model.requests[-1].messages[1].content
    assert "ZEBRAFISH" in body, "the conversation reached the reference condition"
    assert "the marker survives" not in body, "the check's description did not"


def test_every_dataset_scenario_keeps_its_expectations_out_of_the_model(tmp_path):
    model = provider()
    ExperimentRunner(config(provider=model), root=tmp_path / "a").run(SCENARIOS)

    bodies = " ".join(message.content for request in model.requests for message in request.messages)
    for subject in SCENARIOS:
        for check in subject.retention_checks:
            assert check.description not in bodies, subject.scenario_id


def test_the_same_prompt_and_model_are_used_across_strategies(tmp_path):
    """Otherwise the comparison measures the harness rather than the strategies."""
    results = runner(tmp_path).run([scenario()])

    assert len({r.run.continuation_prompt_hash for r in results}) == 1
    assert len({(r.run.provider, r.run.model) for r in results}) == 1
    assert len({r.run.target_tokens for r in results}) == 1


# ----------------------------------------------------------------------
# Accounting


def test_token_accounting_is_recorded(tmp_path):
    results = runner(tmp_path).run([scenario()])
    for result in results:
        assert result.original_tokens > 0
        assert result.context_tokens > 0
        assert result.tokens_exact, "an exact tokenizer was used"


def test_compression_is_original_over_context(tmp_path):
    results = runner(tmp_path, target_tokens=60).run([scenario()])
    full = next(r for r in results if r.strategy == FULL_CONTEXT)
    baseline = next(r for r in results if r.strategy == PHASE_5_BASELINE)

    assert full.compression_ratio == pytest.approx(1.0, abs=0.01)
    assert baseline.compression_ratio > full.compression_ratio


def test_an_estimated_tokenizer_is_reported_as_estimated(tmp_path):
    results = runner(tmp_path, tokenizer=WordTokenizer(exact=False)).run([scenario()])
    assert all(not r.tokens_exact for r in results)
    assert all(not r.run.tokenizer_exact for r in results)


def test_latency_is_recorded(tmp_path):
    results = runner(tmp_path).run([scenario()])
    assert all(r.latency_seconds is not None and r.latency_seconds >= 0 for r in results)


def test_llm_calls_are_counted_separately(tmp_path):
    results = runner(tmp_path).run([scenario()])
    full = next(r for r in results if r.strategy == FULL_CONTEXT)

    assert full.compaction_llm_calls == 0
    assert full.continuation_llm_calls == 1


# ----------------------------------------------------------------------
# Metrics


def test_checks_are_evaluated_against_the_response(tmp_path):
    results = runner(tmp_path, provider=provider(reply="The marker is ZEBRAFISH.")).run(
        [scenario()]
    )
    assert all(r.retention_score == 1.0 for r in results)


def test_a_response_missing_the_fact_scores_zero_but_still_succeeds(tmp_path):
    results = runner(tmp_path, provider=provider(reply="I do not recall a marker.")).run(
        [scenario()]
    )
    for result in results:
        assert result.succeeded, "the run worked; the continuation was simply wrong"
        assert result.retention_score == 0.0


def test_judged_questions_are_not_scored_without_a_judge(tmp_path):
    results = runner(tmp_path).run([by_id("rationale")])
    (judged,) = [outcome for r in results for outcome in r.judged][:1]

    assert not judged.evaluated
    assert judged.passed is None
    assert "no judge" in judged.reasoning


def test_a_judge_answers_judged_questions(tmp_path):
    subject = by_id("rationale")
    question = subject.judged[0].question
    results = runner(
        tmp_path,
        provider=provider(reply="Because the mainframe reader cannot parse dates."),
        judge=KeywordJudge(answers={question: ("mainframe",)}),
    ).run([subject])

    outcomes = [outcome for r in results for outcome in r.judged]
    assert outcomes and all(outcome.evaluated for outcome in outcomes)
    assert all(outcome.passed for outcome in outcomes)


# ----------------------------------------------------------------------
# Failures


def test_a_model_failure_is_recorded_not_swallowed(tmp_path):
    results = runner(tmp_path, provider=FailingProvider(RateLimitError("slow down"))).run(
        [scenario()]
    )

    assert results
    assert all(r.status is RunStatus.MODEL_ERROR for r in results)
    assert all("slow down" in r.error for r in results)
    assert all(r.retention_score is None for r in results)


def test_a_tokenizer_failure_is_its_own_status(tmp_path):
    unavailable = UnavailableTokenizer(
        ModelInfo(provider="fake", model="m"), reason="no tokenizer installed"
    )
    results = runner(tmp_path, tokenizer=unavailable).run([scenario()])

    assert [r.status for r in results] == [RunStatus.TOKENIZATION_ERROR]
    assert "no tokenizer installed" in results[0].error


def test_mixing_dataset_versions_in_one_run_is_refused(tmp_path):
    results = runner(tmp_path).run([scenario(), scenario(scenario_id="old", dataset_version="v0")])
    invalid = [r for r in results if r.status is RunStatus.INVALID_SCENARIO]

    assert len(invalid) == 1
    assert "incomparable" in invalid[0].error


def test_an_empty_run_produces_nothing(tmp_path):
    assert runner(tmp_path).run([]) == []


# ----------------------------------------------------------------------
# Reproducibility


def test_the_run_id_does_not_depend_on_the_clock(tmp_path):
    """Two runs of the same experiment carry the same identity."""
    first = runner(tmp_path / "one").run([scenario()])
    second = runner(tmp_path / "two").run([scenario()])

    assert first[0].run.run_id == second[0].run.run_id
    assert first[0].run.started_at != "" and second[0].run.started_at != ""


def test_the_run_id_changes_when_the_experiment_does(tmp_path):
    base = config()
    changed = config(target_tokens=999)
    assert run_id_for(base, "v1", ["s"]) != run_id_for(changed, "v1", ["s"])
    assert run_id_for(base, "v1", ["s"]) != run_id_for(base, "v2", ["s"])


def test_a_changed_prompt_changes_the_run_id(tmp_path):
    other = Prompt(name="continuation", version=2, text="different instructions")
    assert run_id_for(config(), "v1", ["s"]) != run_id_for(
        config(continuation_prompt=other), "v1", ["s"]
    )


def test_the_whole_experiment_reproduces(tmp_path):
    """Same inputs, same results — apart from the two things that measure time."""
    first = runner(tmp_path / "one").run([scenario()])
    second = runner(tmp_path / "two").run([scenario()])

    ignore = {
        "latency_seconds": True,
        "compaction_latency_seconds": True,
        "continuation_latency_seconds": True,
        "run": {"started_at"},
    }
    for left, right in zip(first, second, strict=True):
        assert left.model_dump(exclude=ignore) == right.model_dump(exclude=ignore)
        assert left.run.run_id == right.run.run_id


def test_results_survive_a_round_trip_to_disk(tmp_path):
    results = runner(tmp_path).run([scenario()])
    path = write_results(tmp_path / "out" / "run.jsonl", results)

    assert list(read_results(path)) == results


def test_prompt_and_dataset_versions_travel_with_the_results(tmp_path):
    results = runner(tmp_path).run([scenario()])
    for result in results:
        assert result.run.dataset_version == DATASET_VERSION
        assert result.run.continuation_prompt_id == "continuation_v1"
        assert result.run.continuation_prompt_hash


# ----------------------------------------------------------------------
# The archive is not disturbed


def fingerprint(archive: Archive) -> dict[str, str]:
    return {
        str(path.relative_to(archive.root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(archive.root.rglob("*"))
        if path.is_file()
    }


def test_evaluation_does_not_modify_the_archive_it_reads(tmp_path):
    experiment = runner(tmp_path)
    experiment.run([scenario()])
    before = fingerprint(experiment.archive)

    ExperimentRunner(config(), root=tmp_path / "eval-archive").run([])
    for session in experiment.archive.sessions():
        log = experiment.archive.open(session)
        FullContextStrategy().prepare(
            StrategyContext(
                log=log,
                provider=provider(),
                tokenizer=WordTokenizer(),
                target_tokens=300,
            )
        )

    assert fingerprint(experiment.archive) == before
    for session in experiment.archive.sessions():
        assert experiment.archive.open(session).verify().ok


def test_the_harness_needs_no_network_key_or_database(tmp_path):
    """Everything above ran with the fakes; this states it as a test."""
    results = runner(tmp_path).run([scenario()])
    assert all(r.run.provider == "fake" for r in results)
    assert all(r.succeeded for r in results)


# ----------------------------------------------------------------------
# Repeated materialization
#
# The archive is append-only, so importing a scenario twice into one session
# would silently double the conversation and every later experiment would
# evaluate a session that says everything twice.


def test_materializing_a_scenario_twice_does_not_duplicate_it(tmp_path):
    archive = Archive(tmp_path / "a")
    subject = scenario()

    first = build_session(archive, subject)
    second = build_session(archive, subject)

    assert first.count == len(subject.conversation)
    assert second.count == len(subject.conversation), "the second call appended nothing"
    assert list(archive.sessions()) == [session_id_for(subject)], "and made no second session"


def test_repeated_materialization_preserves_event_order(tmp_path):
    archive = Archive(tmp_path / "a")
    subject = scenario()

    before = [(r.seq, r.id, r.hash) for r in build_session(archive, subject)]
    after = [(r.seq, r.id, r.hash) for r in build_session(archive, subject)]

    assert after == before
    assert [r.seq for r in archive.open(session_id_for(subject))] == list(range(len(before)))


def test_repeated_experiments_against_one_root_do_not_grow_the_archive(tmp_path):
    """The workflow this protects: running the same experiment again."""
    experiment = runner(tmp_path)
    first = experiment.run([scenario()])
    counts = [experiment.archive.open(s).count for s in experiment.archive.sessions()]

    second = experiment.run([scenario()])
    assert [experiment.archive.open(s).count for s in experiment.archive.sessions()] == counts
    assert [r.original_tokens for r in first] == [r.original_tokens for r in second]
    assert first[0].run.run_id == second[0].run.run_id


def test_an_edited_scenario_gets_its_own_session(tmp_path):
    """Content is in the session id, so an edit cannot reuse a stale materialisation."""
    archive = Archive(tmp_path / "a")
    original = scenario()
    edited = scenario(
        conversation=[*original.conversation, {"role": "user", "content": "one more turn"}]
    )

    build_session(archive, original)
    build_session(archive, edited)

    assert session_id_for(original) != session_id_for(edited)
    assert archive.open(session_id_for(original)).count == len(original.conversation)
    assert archive.open(session_id_for(edited)).count == len(edited.conversation)


def test_a_session_holding_something_else_is_refused(tmp_path):
    """Reuse is only safe while the session can only have come from this scenario."""
    archive = Archive(tmp_path / "a")
    subject = scenario()
    log = archive.create(session_id_for(subject))
    log.append("stray", {"content": "not from this scenario"})

    with pytest.raises(ScenarioMaterializationError, match="written by something other than"):
        build_session(archive, subject)


# ----------------------------------------------------------------------
# Experiment identity
#
# The run id must move whenever anything that can change a result changes, and
# must not move for anything that cannot.


def run_id(**kwargs):
    return run_id_for(config(**kwargs), "v1", ["s"])


def test_the_run_id_is_stable_for_an_unchanged_experiment():
    assert run_id() == run_id()


@pytest.mark.parametrize(
    ("label", "changed"),
    [
        ("target budget", {"target_tokens": 999}),
        ("max output tokens", {"max_output_tokens": 64}),
        ("model", {"provider": FakeProvider(model="other-model", context_window=100_000)}),
        ("context window", {"provider": FakeProvider(context_window=4_096)}),
        ("tokenizer exactness", {"tokenizer": WordTokenizer(exact=False)}),
        ("continuation prompt", {"continuation_prompt": Prompt("continuation", 2, "other")}),
        ("selected strategies", {"strategies": [FULL_CONTEXT]}),
        ("judge", {"judge": KeywordJudge(answers={})}),
        ("extra configuration", {"extra": {"repetitions": 3}}),
    ],
)
def test_every_result_affecting_setting_changes_the_run_id(label, changed):
    assert run_id() != run_id(**changed), f"{label} did not change the run id"


def test_the_dataset_version_and_scenarios_change_the_run_id():
    base = config()
    assert run_id_for(base, "v1", ["s"]) != run_id_for(base, "v2", ["s"])
    assert run_id_for(base, "v1", ["s"]) != run_id_for(base, "v1", ["s", "t"])


def test_scenario_order_does_not_change_the_run_id():
    """The same set of scenarios is the same experiment however it was listed."""
    base = config()
    assert run_id_for(base, "v1", ["b", "a"]) == run_id_for(base, "v1", ["a", "b"])


def test_a_strategy_configuration_change_moves_the_run_id():
    from open_context.compaction import BaselineConfig
    from open_context_eval import Phase5BaselineStrategy

    registry = default_strategies()
    tuned = dict(registry)
    tuned[PHASE_5_BASELINE] = Phase5BaselineStrategy(BaselineConfig(recent_fraction=0.9))

    assert run_id(registry=registry) != run_id(registry=tuned)


def test_a_strategy_version_change_moves_the_run_id():
    class Rewritten(FullContextStrategy):
        version = 99

    tuned = dict(default_strategies())
    tuned[FULL_CONTEXT] = Rewritten()
    assert run_id() != run_id(registry=tuned)


def test_the_clock_does_not_change_the_run_id(tmp_path):
    """Two runs minutes apart are the same experiment."""
    import time as clock

    first = runner(tmp_path / "one").run([scenario()])
    clock.sleep(0.01)
    second = runner(tmp_path / "two").run([scenario()])

    assert first[0].run.started_at != second[0].run.started_at
    assert first[0].run.run_id == second[0].run.run_id


def test_the_archive_location_does_not_change_the_run_id(tmp_path):
    """A filesystem path is machine-specific and says nothing about the experiment."""
    here = ExperimentRunner(config(), root=tmp_path / "somewhere").run([scenario()])
    there = ExperimentRunner(config(), root=tmp_path / "elsewhere-entirely").run([scenario()])
    assert here[0].run.run_id == there[0].run.run_id


def test_the_fingerprint_explains_a_changed_run_id():
    """Exposed so two runs that disagree can be diffed rather than guessed at."""
    base = run_fingerprint(config(), "v1", ["s"])
    other = run_fingerprint(config(target_tokens=999), "v1", ["s"])

    differing = {key for key in base if base[key] != other.get(key)}
    assert differing == {"target_tokens"}
    assert "started_at" not in base
    assert "root" not in base


# ----------------------------------------------------------------------
# Retention is not task completion


def test_the_result_calls_them_retention_checks(tmp_path):
    results = runner(tmp_path, provider=provider(reply="The marker is ZEBRAFISH.")).run(
        [scenario()]
    )
    for result in results:
        assert result.retention_score == 1.0
        assert result.retention_checks
        assert not hasattr(result, "task_completion_score")


def test_there_is_no_task_completion_category():
    """A substring check cannot establish that a task was completed."""
    assert not hasattr(MetricCategory, "TASK_COMPLETION")
    assert "task_completion" not in {category.value for category in MetricCategory}


# ----------------------------------------------------------------------
# The stored fingerprint
#
# A result carrying only "run-a1b2c3" names an experiment nobody can
# reconstruct. The fingerprint that produced the id travels with it.


FINGERPRINT_FIELDS = {
    "dataset_version",
    "scenario_ids",
    "provider",
    "model",
    "context_window",
    "max_output_tokens",
    "tokenizer",
    "tokenizer_provider",
    "tokenizer_model",
    "tokenizer_exact",
    "target_tokens",
    "continuation_prompt_id",
    "continuation_prompt_hash",
    "evaluation_mode",
    "judge",
    "strategies",
    "configuration",
}


def test_every_result_carries_the_complete_fingerprint(tmp_path):
    results = runner(tmp_path).run([scenario(), scenario(scenario_id="second")])

    assert results
    for result in results:
        assert set(result.run.fingerprint) == FINGERPRINT_FIELDS


def test_the_fingerprint_records_every_strategy_with_its_version_and_configuration(tmp_path):
    (result, *_) = runner(tmp_path).run([scenario()])
    strategies = result.run.fingerprint["strategies"]

    assert {entry["name"] for entry in strategies} == set(default_strategies())
    assert all(entry["version"] >= 1 for entry in strategies)
    baseline = next(e for e in strategies if e["name"] == PHASE_5_BASELINE)
    assert baseline["configuration"]["recent_fraction"] == 0.4, "the compactor's own settings"


def test_the_stored_fingerprint_reproduces_the_run_id(tmp_path):
    """The id can be checked against the thing it came from, with no config in hand."""
    results = runner(tmp_path).run([scenario()])

    for result in results:
        assert run_id_from_fingerprint(result.run.fingerprint) == result.run.run_id


def test_the_scenarios_that_ran_are_named_in_the_fingerprint(tmp_path):
    results = runner(tmp_path).run([scenario(), scenario(scenario_id="second")])
    assert results[0].run.fingerprint["scenario_ids"] == ["second", "test-scenario"]


@pytest.mark.parametrize(
    ("label", "changed"),
    [
        ("target budget", {"target_tokens": 111}),
        ("model", {"provider": FakeProvider(model="other-model", context_window=100_000)}),
        ("tokenizer exactness", {"tokenizer": WordTokenizer(exact=False)}),
        ("continuation prompt", {"continuation_prompt": Prompt("continuation", 2, "other")}),
        ("selected strategies", {"strategies": [FULL_CONTEXT]}),
        ("extra configuration", {"extra": {"repetitions": 5}}),
    ],
)
def test_a_result_affecting_change_moves_both_the_fingerprint_and_the_id(tmp_path, label, changed):
    base = runner(tmp_path / "base").run([scenario()])[0].run
    other = runner(tmp_path / "other", **changed).run([scenario()])[0].run

    assert base.fingerprint != other.fingerprint, f"{label} left the fingerprint unchanged"
    assert base.run_id != other.run_id, f"{label} left the run id unchanged"
    assert run_id_from_fingerprint(other.fingerprint) == other.run_id


def test_the_clock_changes_neither_the_fingerprint_nor_the_id(tmp_path):
    import time as clock

    first = runner(tmp_path / "one").run([scenario()])[0].run
    clock.sleep(0.01)
    second = runner(tmp_path / "two").run([scenario()])[0].run

    assert first.started_at != second.started_at
    assert first.fingerprint == second.fingerprint
    assert first.run_id == second.run_id
    assert "started_at" not in first.fingerprint


def test_the_fingerprint_survives_a_round_trip_to_disk_exactly(tmp_path):
    results = runner(tmp_path).run([scenario()])
    restored = list(read_results(write_results(tmp_path / "run.jsonl", results)))

    assert [r.run.fingerprint for r in restored] == [r.run.fingerprint for r in results]
    for result in restored:
        assert run_id_from_fingerprint(result.run.fingerprint) == result.run.run_id


def test_a_results_file_is_self_describing(tmp_path):
    """Read back with nothing else in hand, a line still says what produced it."""
    import json

    runner(tmp_path).run([scenario()])
    path = write_results(tmp_path / "run.jsonl", runner(tmp_path / "b").run([scenario()]))

    line = json.loads(path.read_text().splitlines()[0])
    fingerprint = line["run"]["fingerprint"]

    assert set(fingerprint) == FINGERPRINT_FIELDS
    assert run_id_from_fingerprint(fingerprint) == line["run"]["run_id"]


def test_metadata_that_contradicts_its_fingerprint_is_refused(tmp_path):
    """The convenience fields cannot drift from the identity they contributed to."""
    original = runner(tmp_path).run([scenario()])[0].run

    with pytest.raises(ValueError, match="contradicts its own fingerprint"):
        original.model_copy(update={"target_tokens": 999}).model_validate(
            original.model_dump() | {"target_tokens": 999}
        )
