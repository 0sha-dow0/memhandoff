"""Running an experiment.

```
scenario -> archive -> strategy -> context -> same task -> same model -> result
```

**Everything except the strategy is held identical.** One model, one tokenizer,
one continuation prompt, one task, across every arm. A comparison where the
full-context condition ran on a different model than the compacted one is not a
comparison of representations, and the run metadata records enough for a reader
to notice if two result sets were produced that way.

**The archive is built from the scenario and never modified.** Each scenario
becomes a session log through the ordinary import path, and strategies read it.
Nothing here writes to an archive after it is built.

**A failure is recorded, never skipped and never scored.** Every way a run can
go wrong has a status, and a run that failed carries no metrics rather than
zeros.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from open_context.archive import Archive, SessionLog, canonical_json
from open_context.compaction import Prompt, render_events, stream_archive
from open_context.importers import EventType, RawEvent, import_events
from open_context.llm import (
    LLMError,
    LLMProvider,
    TokenCount,
    TokenizationUnavailableError,
    Tokenizer,
)
from open_context.llm.free_models import billing_class_of
from open_context_eval.judge import EvaluationJudge
from open_context_eval.prompts import CONTINUATION_PROMPT_V1, build_continuation_request
from open_context_eval.results import (
    MIRRORED_FINGERPRINT_FIELDS,
    CompletionOutcome,
    EvaluationResult,
    JudgedOutcome,
    RetentionOutcome,
    RunMetadata,
    RunStatus,
)
from open_context_eval.scenario import EvaluationScenario
from open_context_eval.strategies import (
    EvaluationStrategy,
    StrategyContext,
    default_strategies,
)

SOFTWARE_VERSION = "phase-5.6"
"""Which phase of the harness produced a result.

Recorded beside the fingerprint rather than inside it, along with
``started_at`` and ``git_commit``: it describes the occasion, not the
experiment. So bumping it identifies newer results without silently splitting
one experiment's identity across two ids.
"""

SkipPredicate = Callable[[str, str, str], bool]
"""``(run_id, scenario_id, strategy_name) -> already recorded?``

Three fields identify a cell because the run id already carries everything else
that distinguishes one experiment from another — budget, repetition, model,
tokenizer, prompt, dataset. A predicate keyed on anything less would let a
resumed run mistake a cell from a different budget for one it had done.
"""

ROLES: dict[str, EventType] = {
    "user": EventType.USER_MESSAGE,
    "assistant": EventType.ASSISTANT_MESSAGE,
    "system": EventType.SYSTEM_MESSAGE,
    "tool_call": EventType.TOOL_CALL,
    "tool_result": EventType.TOOL_RESULT,
}


@dataclass(frozen=True)
class EvaluationConfig:
    """One experiment.

    Small on purpose. Anything a benchmark will sweep belongs here; anything it
    will not belongs in a strategy.
    """

    provider: LLMProvider
    tokenizer: Tokenizer
    target_tokens: int = 2_000
    strategies: Sequence[str] = ()
    """Names to run. Empty means every registered strategy."""

    continuation_prompt: Prompt = CONTINUATION_PROMPT_V1
    max_output_tokens: int | None = None
    judge: EvaluationJudge | None = None
    registry: dict[str, EvaluationStrategy] = field(default_factory=default_strategies)
    evaluation_mode: str = "deterministic_test"
    """``deterministic_test`` or ``real_model``.

    Part of the fingerprint, so a fake-provider run and a real-model run of
    otherwise identical configuration are never filed under one identity. A
    number produced against a fake provider measures the harness, not a strategy,
    and the two must not be mistaken for each other in a results file."""

    extra: dict[str, Any] = field(default_factory=dict)

    def selected(self) -> list[tuple[str, EvaluationStrategy]]:
        names = list(self.strategies) if self.strategies else sorted(self.registry)
        missing = [name for name in names if name not in self.registry]
        if missing:
            raise KeyError(f"no strategy registered as {missing}; have {sorted(self.registry)}")
        return [(name, self.registry[name]) for name in names]


def scenario_events(scenario: EvaluationScenario) -> Iterator[RawEvent]:
    """Turn a scenario's conversation into normalized events.

    Uses the same event model the runtime imports into, so a scenario is not a
    parallel conversation format that could drift from the real one.
    """
    for index, record in enumerate(scenario.conversation):
        kind = str(record.get("role") or record.get("type") or "user")
        yield RawEvent(
            type=ROLES.get(kind, EventType.OTHER),
            provider="evaluation-dataset",
            source_id=str(record.get("id") or f"{scenario.scenario_id}-{index}"),
            source_type=kind,
            text=record.get("content"),
            tool_name=record.get("tool_name"),
            tool_call_id=record.get("tool_call_id"),
            raw=dict(record),
        )


class ScenarioMaterializationError(ValueError):
    """A scenario's session exists but does not hold what the scenario says."""


def build_session(archive: Archive, scenario: EvaluationScenario) -> SessionLog:
    """Materialise a scenario as an archived session, at most once.

    **Idempotent.** The archive is append-only, so importing a scenario a second
    time into the same session would append a second copy of the conversation
    and silently double it — an experiment repeated against one archive root
    would then evaluate a conversation that says everything twice. So the
    session is created and imported only when it is not already there.

    Identity is derived from the scenario's **content**, not just its id: a
    scenario that is edited becomes a different session rather than colliding
    with a stale materialisation of its earlier self. That is also what makes
    reuse safe, since a session that exists under this id can only have come
    from exactly these events.

    Nothing about the production archive is changed to achieve this. The
    append-only semantics are the reason for the care, not an obstacle to work
    around.
    """
    session_id = session_id_for(scenario)
    expected = len(scenario.conversation)

    if archive.exists(session_id):
        log = archive.open(session_id)
        if log.count != expected:
            raise ScenarioMaterializationError(
                f"session {session_id} for scenario {scenario.scenario_id!r} holds "
                f"{log.count} events but the scenario has {expected}; the archive was "
                "written by something other than this scenario"
            )
        return log

    log = archive.create(session_id)
    import_events(
        log,
        scenario_events(scenario),
        provider="evaluation-dataset",
        source_key=f"{scenario.dataset_version}:{scenario.scenario_id}",
    )
    return log


def original_token_count(log: SessionLog, tokenizer: Tokenizer) -> TokenCount:
    """What the untouched conversation costs, measured the way a context is.

    Rendered and counted as text, exactly as ``FullContextStrategy`` measures
    what it produces. Counting the original one way and the context another
    would put the difference between two counting methods into every
    compression ratio, and the reference condition would not come out at 1.0
    even though it changed nothing.
    """
    return tokenizer.count_text(render_events(tuple(stream_archive(log))))


def session_id_for(scenario: EvaluationScenario) -> str:
    """A session id derived from the scenario's identity and its content.

    Content is included so an edited scenario cannot reuse the session its
    earlier version materialised. Nothing machine-specific goes in, so the same
    scenario lands in the same session on every machine.
    """
    material = canonical_json(
        {
            "dataset_version": scenario.dataset_version,
            "scenario_id": scenario.scenario_id,
            "conversation": list(scenario.conversation),
        }
    )
    return f"ses_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def run_fingerprint(
    config: EvaluationConfig, dataset_version: str, scenario_ids: Sequence[str]
) -> dict[str, Any]:
    """Everything about an experiment that can change its outcome.

    Exposed rather than hidden inside ``run_id_for`` so that two runs which
    produced different ids can be diffed to find out why.

    Deliberately excluded: timestamps, filesystem paths, hostnames, and anything
    else that varies between machines running the same experiment. An id that
    moved with the clock or the working directory could not identify an
    experiment at all.
    """
    model = config.provider.model_info()
    tokenizer = config.tokenizer.model_info()
    return {
        "dataset_version": dataset_version,
        "scenario_ids": sorted(scenario_ids),
        "provider": model.provider,
        "model": model.model,
        "context_window": model.context_window,
        "max_output_tokens": config.max_output_tokens,
        "tokenizer": tokenizer.tokenizer,
        "tokenizer_provider": tokenizer.provider,
        "tokenizer_model": tokenizer.model,
        "tokenizer_exact": tokenizer.counts_exactly,
        "target_tokens": config.target_tokens,
        "continuation_prompt_id": config.continuation_prompt.identifier,
        "continuation_prompt_hash": config.continuation_prompt.content_hash,
        "evaluation_mode": config.evaluation_mode,
        "judge": None if config.judge is None else config.judge.name,
        "strategies": [
            {
                "name": name,
                "version": strategy.version,
                "configuration": strategy.configuration(),
            }
            for name, strategy in config.selected()
        ],
        "configuration": dict(config.extra),
    }


def run_id_from_fingerprint(fingerprint: Mapping[str, Any]) -> str:
    """Hash a fingerprint into a run id.

    Separate from ``run_fingerprint`` so a caller can compute the fingerprint
    once, persist it, and derive the id from that same object. There is exactly
    one place a run id comes from, and exactly one object it comes from, so a
    stored fingerprint can always be re-hashed to check the id it travels with.
    """
    material = canonical_json(dict(fingerprint))
    return "run-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def run_id_for(config: EvaluationConfig, dataset_version: str, scenario_ids: Sequence[str]) -> str:
    """A run's identity, derived from what it ran rather than when.

    A canonical serialisation of ``run_fingerprint``: sorted keys, fixed
    separators, so the same experiment hashes identically on any machine and two
    result files can be compared without matching clocks.

    The point of including so much is that a run id which stayed the same across
    a changed tokenizer, a changed budget, or a changed strategy configuration
    would let two incomparable experiments be filed under one identity, which is
    a worse failure than an id that changes too eagerly.
    """
    return run_id_from_fingerprint(run_fingerprint(config, dataset_version, scenario_ids))


class ExperimentRunner:
    """Runs scenarios against strategies and records what happened."""

    def __init__(self, config: EvaluationConfig, *, root: str | Path) -> None:
        self.config = config
        self.archive = Archive(root)

    # ------------------------------------------------------------------

    def run(self, scenarios: Iterable[EvaluationScenario]) -> list[EvaluationResult]:
        """Every result, once the run has finished.

        A thin wrapper over ``stream``. Callers that must not lose work if the
        run stops early — anything spending a metered request quota — should
        consume ``stream`` and persist as they go.
        """
        return list(self.stream(scenarios))

    def stream(
        self,
        scenarios: Iterable[EvaluationScenario],
        *,
        skip: SkipPredicate | None = None,
    ) -> Iterator[EvaluationResult]:
        """Yield each result as it is produced.

        **Yielded, not accumulated, so an interrupted run keeps what it paid
        for.** Against a metered free tier a request is not repeatable on
        demand, and a run that collected results in a list and raised before
        returning it would discard every measurement it had already bought.

        ``skip`` is consulted per cell, before any work, and is how a resumed
        run avoids re-running what a previous one already recorded. It receives
        the run id, the scenario id, and the strategy name — the three fields
        that identify a cell, given that the run id already encodes the budget,
        the repetition, the model, and everything else in the fingerprint. A
        skipped cell yields nothing at all: it is neither re-measured nor
        recorded a second time.
        """
        ordered = list(scenarios)
        if not ordered:
            return
        dataset_version = ordered[0].dataset_version
        metadata = self._metadata(dataset_version, [s.scenario_id for s in ordered])

        for scenario in ordered:
            if scenario.dataset_version != dataset_version:
                yield self._failed(
                    metadata,
                    scenario,
                    strategy_name="",
                    strategy_version=0,
                    status=RunStatus.INVALID_SCENARIO,
                    error=(
                        f"scenario is dataset {scenario.dataset_version!r} but the run is "
                        f"{dataset_version!r}; mixing versions in one run would make the "
                        "results incomparable"
                    ),
                )
                continue
            yield from self._run_scenario(metadata, scenario, skip)

    def _metadata(self, dataset_version: str, scenario_ids: Sequence[str]) -> RunMetadata:
        """Run metadata built from one fingerprint, computed once.

        The fingerprint is the object that gets hashed into ``run_id`` and the
        object that gets persisted, and the convenience fields beside it are read
        out of it rather than gathered again from the provider. So a stored
        result is self-describing: the fingerprint it carries re-hashes to the id
        it carries, and no field can quietly disagree with the identity it
        contributed to.
        """
        fingerprint = run_fingerprint(self.config, dataset_version, scenario_ids)
        return RunMetadata(
            run_id=run_id_from_fingerprint(fingerprint),
            fingerprint=fingerprint,
            billing_class=self._billing_class(fingerprint),
            started_at=datetime.now(UTC).isoformat(),
            software_version=SOFTWARE_VERSION,
            git_commit=_git_commit(),
            **{field: fingerprint[field] for field in MIRRORED_FINGERPRINT_FIELDS},
        )

    @staticmethod
    def _billing_class(fingerprint: Mapping[str, Any]) -> str:
        """What the configured model costs, per the free-model allowlist.

        Read from the registry rather than taken on trust from configuration:
        a caller asserting a model is free is exactly what the allowlist exists
        to check. ``none`` in deterministic mode, where nothing was called.
        """
        if fingerprint.get("evaluation_mode") != "real_model":
            return "none"
        return billing_class_of(str(fingerprint["provider"]), str(fingerprint["model"])).value

    def _run_scenario(
        self,
        metadata: RunMetadata,
        scenario: EvaluationScenario,
        skip: SkipPredicate | None = None,
    ) -> Iterator[EvaluationResult]:
        pending = [
            (name, strategy)
            for name, strategy in self.config.selected()
            if skip is None or not skip(metadata.run_id, scenario.scenario_id, name)
        ]
        if not pending:
            return

        try:
            log = build_session(self.archive, scenario)
            original = original_token_count(log, self.config.tokenizer)
        except TokenizationUnavailableError as exc:
            yield self._failed(metadata, scenario, "", 0, RunStatus.TOKENIZATION_ERROR, str(exc))
            return
        except (LLMError, ValueError) as exc:
            yield self._failed(metadata, scenario, "", 0, RunStatus.INVALID_SCENARIO, str(exc))
            return

        for name, strategy in pending:
            yield self._run_one(metadata, scenario, log, original.count, name, strategy)

    def _run_one(
        self,
        metadata: RunMetadata,
        scenario: EvaluationScenario,
        log: SessionLog,
        original_tokens: int,
        name: str,
        strategy: EvaluationStrategy,
    ) -> EvaluationResult:
        context = StrategyContext(
            log=log,
            provider=self.config.provider,
            tokenizer=self.config.tokenizer,
            target_tokens=self.config.target_tokens,
        )

        try:
            prepared = strategy.prepare(context)
        except TokenizationUnavailableError as exc:
            return self._failed(
                metadata, scenario, name, strategy.version, RunStatus.TOKENIZATION_ERROR, str(exc)
            )
        except LLMError as exc:
            return self._failed(
                metadata, scenario, name, strategy.version, RunStatus.MODEL_ERROR, str(exc)
            )

        if not prepared.available:
            return self._failed(
                metadata,
                scenario,
                name,
                strategy.version,
                RunStatus.CONTEXT_ERROR,
                prepared.unavailable_reason,
                original_tokens=original_tokens,
                context_tokens=prepared.tokens.count,
            )

        continuation_started = time.perf_counter()
        try:
            answer = self.config.provider.generate(
                build_continuation_request(
                    prepared.text,
                    scenario.task.instruction,
                    prompt=self.config.continuation_prompt,
                    max_output_tokens=self.config.max_output_tokens,
                )
            )
        except LLMError as exc:
            return self._failed(
                metadata,
                scenario,
                name,
                strategy.version,
                RunStatus.MODEL_ERROR,
                str(exc),
                original_tokens=original_tokens,
                context_tokens=prepared.tokens.count,
            )

        continuation_elapsed = time.perf_counter() - continuation_started
        checks, completion, judged = self._evaluate(scenario, answer.text)
        usage = answer.usage

        return EvaluationResult(
            run=metadata,
            scenario_id=scenario.scenario_id,
            failure_modes=tuple(mode.value for mode in scenario.failure_modes),
            task_id=scenario.task.task_id,
            task_kind=scenario.task.kind.value,
            strategy=name,
            strategy_version=strategy.version,
            status=RunStatus.SUCCESS,
            response=answer.text,
            original_tokens=original_tokens,
            context_tokens=prepared.tokens.count,
            output_tokens=usage.output_tokens if usage else None,
            tokens_exact=prepared.tokens.exact,
            evaluation_mode=self.config.evaluation_mode,
            compaction_input_tokens=prepared.input_tokens,
            compaction_output_tokens=prepared.output_tokens,
            compaction_llm_calls=prepared.llm_calls,
            compaction_latency_seconds=prepared.latency_seconds,
            continuation_input_tokens=usage.input_tokens if usage else None,
            continuation_output_tokens=usage.output_tokens if usage else None,
            continuation_llm_calls=1,
            continuation_latency_seconds=continuation_elapsed,
            latency_seconds=(prepared.latency_seconds or 0.0) + continuation_elapsed,
            strategy_detail=dict(prepared.detail),
            retention_checks=checks,
            completion=completion,
            judged=judged,
        )

    def _evaluate(
        self, scenario: EvaluationScenario, response: str
    ) -> tuple[
        tuple[RetentionOutcome, ...], tuple[CompletionOutcome, ...], tuple[JudgedOutcome, ...]
    ]:
        checks = []
        for check in scenario.retention_checks:
            passed, problems = check.evaluate(response)
            checks.append(
                RetentionOutcome(
                    check_id=check.check_id,
                    category=check.category.value,
                    passed=passed,
                    problems=problems,
                )
            )

        completion = []
        for criterion in scenario.completion:
            status, detail = criterion.evaluate(response)
            completion.append(
                CompletionOutcome(
                    criterion_id=criterion.criterion_id,
                    kind=criterion.kind.value,
                    status=status.value,
                    detail=detail,
                )
            )

        judged = []
        for question in scenario.judged:
            if self.config.judge is None:
                judged.append(
                    JudgedOutcome(
                        question_id=question.question_id,
                        category=question.category.value,
                        evaluated=False,
                        reasoning="no judge configured",
                    )
                )
                continue
            verdict = self.config.judge.judge(response, question.question)
            judged.append(
                JudgedOutcome(
                    question_id=question.question_id,
                    category=question.category.value,
                    evaluated=verdict.evaluated,
                    passed=verdict.passed if verdict.evaluated else None,
                    reasoning=verdict.reasoning,
                )
            )
        return tuple(checks), tuple(completion), tuple(judged)

    def _failed(
        self,
        metadata: RunMetadata,
        scenario: EvaluationScenario,
        strategy_name: str,
        strategy_version: int,
        status: RunStatus,
        error: str,
        *,
        original_tokens: int = 0,
        context_tokens: int = 0,
    ) -> EvaluationResult:
        """A failure carries no checks and no metrics. It is not a score of zero."""
        return EvaluationResult(
            run=metadata,
            scenario_id=scenario.scenario_id,
            failure_modes=tuple(mode.value for mode in scenario.failure_modes),
            task_id=scenario.task.task_id,
            task_kind=scenario.task.kind.value,
            strategy=strategy_name,
            strategy_version=strategy_version,
            status=status,
            error=error,
            original_tokens=original_tokens,
            context_tokens=context_tokens,
            evaluation_mode=self.config.evaluation_mode,
        )


def _git_commit() -> str | None:
    """The commit under evaluation, when there is one.

    Best effort and never fatal: an evaluation run in an exported tree has no
    commit, and that is worth recording as absent rather than crashing over.
    """
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


__all__ = [
    "SOFTWARE_VERSION",
    "EvaluationConfig",
    "ExperimentRunner",
    "ScenarioMaterializationError",
    "SkipPredicate",
    "build_session",
    "original_token_count",
    "run_fingerprint",
    "run_id_for",
    "run_id_from_fingerprint",
    "scenario_events",
    "session_id_for",
]
