"""What a run produced, and how it is stored.

One JSON object per line, each independently interpretable. A line carries the
run metadata it belongs to rather than referring to a header, which costs some
repetition and buys the ability to concatenate, filter, and grep result files
without a parser that tracks state.

**A failure is not a zero.** ``RunStatus`` distinguishes a model error from a
continuation that scored nothing, and metrics are absent rather than zero on a
failed run. Averaging a failure as zero silently reports a broken experiment as
a bad strategy.

**No result in this repository is fabricated.** The harness produces the
machinery to generate numbers; it ships none, and nothing here should ever be
populated by hand.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

RESULT_FORMAT_VERSION = 5

_DISOWNED = "should not be read as"
"""The phrase a strategy uses to disown a cell it produced.

Matched on text because the warning is already the strategy's public way of
saying so, and adding a parallel flag would create two sources of truth that can
disagree. Kept as a constant so the strategy and the reader agree on the wording;
a test pins that `HybridCompaction`'s warning still contains it.
"""
"""Bumped when the shape of a result line changes.

Version 2 renamed ``checks`` to ``retention_checks``: a field whose name implied
it measured task completion was worth renaming in the stored format and not only
in the code that reads it.

Version 3 added ``RunMetadata.fingerprint``, so a stored result carries the
configuration its ``run_id`` was derived from rather than only the id.

Version 5 added ``RunMetadata.billing_class``, so a real-model result records
that it was produced by a model approved as free.

Version 4 added task completion, the evaluation mode, and the split between
compaction and continuation cost. ``strategy_llm_calls`` and
``downstream_llm_calls`` became ``compaction_llm_calls`` and
``continuation_llm_calls``, since a benchmark comparing strategies has to be able
to say what each one cost to produce."""


class RunStatus(StrEnum):
    """How a single scenario-strategy run ended."""

    SUCCESS = "success"
    MODEL_ERROR = "model_error"
    CONTEXT_ERROR = "context_error"
    TOKENIZATION_ERROR = "tokenization_error"
    EVALUATION_ERROR = "evaluation_error"
    INVALID_SCENARIO = "invalid_scenario"


class RetentionOutcome(BaseModel):
    """One deterministic retention check against one continuation.

    Says whether an expected fact, constraint, or decision survived into the
    response. It does **not** say whether the continuation was any good; see
    ``EvaluationResult.retention_score``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    category: str
    passed: bool
    problems: tuple[str, ...] = ()


class CompletionOutcome(BaseModel):
    """One task-completion criterion against one continuation.

    ``status`` distinguishes an omission from a fabrication from something that
    could not be checked, because "failed" alone hides which of those happened
    and they have different causes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion_id: str
    kind: str
    status: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def fabricated(self) -> bool:
        """Used something the conversation ruled out. The deterministic hallucination signal."""
        return self.status == "fabricated"


class JudgedOutcome(BaseModel):
    """One judged question, answered or explicitly not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    category: str
    evaluated: bool = False
    passed: bool | None = None
    reasoning: str = ""


MIRRORED_FINGERPRINT_FIELDS = (
    "dataset_version",
    "provider",
    "model",
    "context_window",
    "max_output_tokens",
    "tokenizer",
    "tokenizer_exact",
    "target_tokens",
    "continuation_prompt_id",
    "continuation_prompt_hash",
    "configuration",
)
"""Fingerprint entries also exposed as named fields on ``RunMetadata``.

Convenience for reading and reporting, not a second source of truth. Listed once
here so that populating them and checking them cannot drift apart, and validated
on construction so a stored result can never say one thing in its fingerprint and
another beside it.
"""


class RunMetadata(BaseModel):
    """Enough to reproduce the run that produced a result.

    **Self-describing.** ``fingerprint`` is the exact object that was hashed into
    ``run_id``, persisted verbatim, so a result read back years later can be
    re-hashed to confirm its own identity and diffed against another run to find
    out why the two disagree. A result carrying only an opaque ``run-a1b2c3``
    would name an experiment nobody could reconstruct.

    ``run_id`` is derived from that configuration rather than from the clock, so
    the same experiment produces the same identity on any machine and two runs
    can be compared without matching timestamps. ``started_at``, ``git_commit``,
    and ``software_version`` are recorded beside the fingerprint and deliberately
    outside it: they describe the occasion, not the experiment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    run_id: str
    format_version: int = RESULT_FORMAT_VERSION
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    """Everything that determined ``run_id``. See ``runner.run_fingerprint``."""

    dataset_version: str
    started_at: str = ""
    software_version: str = ""
    git_commit: str | None = None

    provider: str
    model: str
    billing_class: str = "none"
    """What calling this model cost, from the free-model allowlist.

    ``free`` for a real-model run, which is the only class the benchmark will
    execute. ``none`` in deterministic mode, where no model was called at all.
    Deliberately outside the fingerprint: it is a fact about the model that the
    provider and model fields already identify, not an independent variable of
    the experiment.
    """

    context_window: int | None = None
    tokenizer: str | None = None
    tokenizer_exact: bool = False

    continuation_prompt_id: str
    continuation_prompt_hash: str
    target_tokens: int
    max_output_tokens: int | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _mirrors_match_the_fingerprint(self) -> RunMetadata:
        """Refuse metadata whose convenience fields contradict its fingerprint.

        Only checked when a fingerprint is present, so hand-built metadata in a
        test stays easy to write. When one is present it is authoritative, and a
        disagreement is a bug worth failing on rather than persisting.
        """
        if not self.fingerprint:
            return self
        disagreements = [
            f"{field}: fingerprint says {self.fingerprint[field]!r}, field says "
            f"{getattr(self, field)!r}"
            for field in MIRRORED_FINGERPRINT_FIELDS
            if field in self.fingerprint and self.fingerprint[field] != getattr(self, field)
        ]
        if disagreements:
            raise ValueError(
                "run metadata contradicts its own fingerprint: " + "; ".join(disagreements)
            )
        return self


class EvaluationResult(BaseModel):
    """One scenario, run under one strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    run: RunMetadata
    scenario_id: str
    failure_modes: tuple[str, ...] = ()
    task_id: str
    task_kind: str
    strategy: str
    strategy_version: int
    status: RunStatus = RunStatus.SUCCESS
    error: str = ""

    response: str = ""

    original_tokens: int = 0
    context_tokens: int = 0
    output_tokens: int | None = None
    tokens_exact: bool = False
    evaluation_mode: str = "deterministic_test"
    """``deterministic_test`` or ``real_model``.

    Recorded on every result so a fake-provider number can never be read as a
    measurement of model quality. Deterministic mode validates the machinery;
    only real-model mode measures anything about a strategy.
    """

    compaction_input_tokens: int | None = None
    compaction_output_tokens: int | None = None
    compaction_llm_calls: int = 0
    compaction_latency_seconds: float | None = None

    continuation_input_tokens: int | None = None
    continuation_output_tokens: int | None = None
    continuation_llm_calls: int = 0
    continuation_latency_seconds: float | None = None

    latency_seconds: float | None = None
    strategy_detail: dict[str, Any] = Field(default_factory=dict)

    retention_checks: tuple[RetentionOutcome, ...] = ()
    completion: tuple[CompletionOutcome, ...] = ()
    judged: tuple[JudgedOutcome, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.SUCCESS

    @property
    def strategy_disowned(self) -> bool:
        """The strategy says this cell is not what its name claims.

        A composite strategy can fail a part and still return a context. Hybrid
        compaction does exactly that: when extraction yields no state it returns
        the Phase 5 baseline and warns, in terms, that the result must not be
        read as hybrid. That is the honest behaviour — the alternative is an arm
        that is sometimes hybrid and sometimes not, averaged under one name.

        **But warning is only half of it.** Phase 8.5 ran 24 hybrid cells on an
        8B model; extraction produced no usable state in **21** of them, and the
        arm's headline score was the Phase 5 baseline wearing hybrid's name. The
        warning was in the results file the whole time. Nothing read it.

        So the disowning is surfaced as a property and excluded from aggregates
        rather than left as prose in a list nobody aggregates.
        """
        return any(_DISOWNED in warning for warning in self.strategy_warnings)

    @property
    def strategy_warnings(self) -> tuple[str, ...]:
        """Warnings the strategy attached to this cell, if any."""
        warnings = self.strategy_detail.get("warnings")
        if not isinstance(warnings, list):
            return ()
        return tuple(str(warning) for warning in warnings)

    @property
    def measures_its_strategy(self) -> bool:
        """Whether this cell may be counted towards its arm's score."""
        return self.succeeded and not self.strategy_disowned

    @property
    def compression_ratio(self) -> float | None:
        """Original tokens per context token.

        **Not a quality measure.** A strategy that compresses harder can
        continue the work worse, which is the whole reason this harness exists.
        ``None`` when the run failed, so a broken run never contributes a number.
        """
        if not self.succeeded or self.context_tokens == 0:
            return None
        return self.original_tokens / self.context_tokens

    @property
    def retention_checks_passed(self) -> int | None:
        return sum(1 for check in self.retention_checks if check.passed) if self.succeeded else None

    @property
    def retention_score(self) -> float | None:
        """Fraction of deterministic retention checks passed.

        **Retention is not task completion.** This says how much of what the
        scenario named explicitly survived into the response. It does not
        establish that the agent continued the work well, and a strategy that
        scores 1.0 here may still have produced a useless continuation. What
        "almost as effectively as full context" means is deliberately left
        undefined; see docs/evaluation.md.

        ``None`` if the run failed or the scenario declared no checks.
        """
        if not self.succeeded or not self.retention_checks:
            return None
        return sum(1 for check in self.retention_checks if check.passed) / len(
            self.retention_checks
        )

    @property
    def task_completed(self) -> bool | None:
        """Whether the task was completed: every criterion passed.

        **Not the same as ``retention_score``, and never averaged with it.**
        Retention asks what survived; this asks whether the agent acted
        correctly. ``None`` when the run failed, when the scenario declares no
        criteria, or when any criterion could not be evaluated — a task whose
        success conditions were only partly checked has not been shown either
        way.
        """
        from open_context_eval.completion import CompletionStatus, scenario_completed

        if not self.succeeded:
            return None
        return scenario_completed(
            tuple(CompletionStatus(outcome.status) for outcome in self.completion)
        )

    @property
    def deterministic_hallucination_checks(self) -> int:
        """Criteria where the response asserted something the conversation ruled out.

        Deliberately not called a hallucination *rate*. It counts a narrow,
        deterministic signal over scenario-declared alternatives, and says
        nothing about invention in general.
        """
        return sum(1 for outcome in self.completion if outcome.fabricated) if self.succeeded else 0

    @property
    def total_tokens(self) -> int | None:
        """Every token the strategy and the continuation together consumed.

        A strategy producing an 8K context from 100K tokens of compaction work
        is not cheap, and comparing it on context size alone would say it was.
        ``None`` when the provider reported nothing.
        """
        parts = [
            self.compaction_input_tokens,
            self.compaction_output_tokens,
            self.continuation_input_tokens,
            self.continuation_output_tokens,
        ]
        measured = [value for value in parts if value is not None]
        return sum(measured) if measured else None

    def scores_by_category(self) -> dict[str, float]:
        """Pass rate per metric category. Empty when the run failed."""
        if not self.succeeded:
            return {}
        totals: dict[str, list[bool]] = {}
        for check in self.retention_checks:
            totals.setdefault(check.category, []).append(check.passed)
        return {
            category: sum(results) / len(results) for category, results in sorted(totals.items())
        }


def write_results(path: str | Path, results: Iterable[EvaluationResult]) -> Path:
    """Write one JSON object per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")
    return destination


def append_results(path: str | Path, results: Iterable[EvaluationResult]) -> Path:
    """Add results to a file, keeping whatever is already there.

    **The append is what makes a metered run survivable.** JSON Lines is
    append-friendly by construction — a line is a complete record — so a run
    that stops after forty requests leaves forty valid results rather than a
    truncated document, and the next run adds to them.

    Callers persisting one result at a time should use ``ResultWriter``, which
    holds the file open and flushes each line, rather than reopening per result.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")
    return destination


class ResultWriter:
    """Persist results one at a time, durably enough to survive the process.

    Each line is written and flushed as it arrives, so a run killed partway
    through — or one that stopped because its request budget ran out — leaves
    every result it had already paid for on disk. Nothing is buffered until
    close.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self.written = 0

    def write(self, result: EvaluationResult) -> None:
        self._handle.write(result.model_dump_json() + "\n")
        self._handle.flush()
        self.written += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> ResultWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_results(path: str | Path) -> Iterator[EvaluationResult]:
    """Read results back, one at a time."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield EvaluationResult.model_validate(json.loads(stripped))


def completed_cells(path: str | Path) -> set[tuple[str, str, str]]:
    """Which ``(run_id, scenario_id, strategy)`` cells a results file already holds.

    The set a resumed run consults so it does not spend a request on a cell it
    already has. Missing file means nothing is done, which is the honest reading
    of "no results exist".

    **Only successful cells count as done.** A cell that failed is worth
    retrying — a rate limit or a timeout says nothing about the strategy — and
    treating it as complete would freeze a transient failure into the results
    permanently. Retrying it appends a second row for the same cell, and the
    later row is the one that measured something.
    """
    source = Path(path)
    if not source.exists():
        return set()
    return {
        (result.run.run_id, result.scenario_id, result.strategy)
        for result in read_results(source)
        if result.status is RunStatus.SUCCESS
    }


__all__ = [
    "MIRRORED_FINGERPRINT_FIELDS",
    "RESULT_FORMAT_VERSION",
    "CompletionOutcome",
    "EvaluationResult",
    "JudgedOutcome",
    "ResultWriter",
    "RetentionOutcome",
    "RunMetadata",
    "RunStatus",
    "append_results",
    "completed_cells",
    "read_results",
    "write_results",
]
