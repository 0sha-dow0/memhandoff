"""What is being evaluated.

A scenario is one sentence: **here is a conversation that happened, and here is
the task the next agent must perform.** Everything else is bookkeeping.

**Model input and evaluator expectations are separate types, on purpose.** A
scenario holds both the conversation the agent may see and the answers only the
evaluator may see, and nothing stops an author putting the answer in the task
description by accident. So the two never travel together: the runner builds a
prompt from a context string and a task string and is never handed a scenario,
which makes leaking an expectation impossible rather than merely discouraged.
See ``prompts.build_continuation_request``.

**Checks are deliberately one shape.** A check names strings that must appear,
strings of which at least one must appear, and strings that must not. That
covers a critical fact, a negative constraint, a decision taken over its
predecessor, an avoided failed approach, and a completed task, without a class
per metric. The metric category is a label on the check rather than a different
mechanism.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_context_eval.completion import CompletionCriterion


class MetricCategory(StrEnum):
    """What a retention check is evidence about.

    Categories exist so results aggregate into the measures the product cares
    about. Two are declared but not deterministically checkable and are answered
    by a judge or not at all; see ``judge.py`` and docs/evaluation.md.

    There is deliberately no ``task_completion`` category. A substring check
    cannot establish that a task was completed, and a category by that name
    sitting among these would invite exactly the conflation the whole
    "retention" naming exists to prevent. Task completion is unmeasured.
    """

    CRITICAL_FACT = "critical_fact_retention"
    CONSTRAINT = "constraint_retention"
    DECISION = "decision_retention"
    RATIONALE = "decision_rationale"
    TEMPORAL_STATE = "temporal_state_accuracy"
    OPEN_TASK = "open_task_retention"
    FAILED_APPROACH = "failed_approach_retention"
    ENTITY = "entity_accuracy"

    PROVENANCE = "provenance_accuracy"
    """Requires provenance in the context. Nothing produces it yet."""

    HALLUCINATION = "hallucination_rate"
    """Requires a judge. Not deterministically checkable."""


class FailureMode(StrEnum):
    """The retention trap a scenario is built around.

    Named so a benchmark can report which kinds of forgetting a strategy is
    prone to, rather than one undifferentiated score.
    """

    EARLY_CRITICAL_FACT = "early_critical_fact"
    LATE_CRITICAL_FACT = "late_critical_fact"
    LONG_IRRELEVANT = "long_irrelevant_section"
    NEGATIVE_CONSTRAINT = "negative_constraint"
    REVERSED_DECISION = "reversed_decision"
    FAILED_APPROACH = "failed_approach"
    EXACT_VALUE = "exact_value"
    SIMILAR_ENTITIES = "similar_entities"
    TEMPORAL_STATE = "temporal_state"
    TOOL_RESULT = "tool_result"
    OPEN_TASK = "open_task"
    RATIONALE = "rationale"
    MULTI_STEP = "multi_step"
    REPEATED_INFORMATION = "repeated_information"
    POSITIVE_CONSTRAINT = "positive_constraint"
    CURRENT_STATE = "current_state"
    DECISION = "decision"


class RetentionCheck(BaseModel):
    """One deterministic retention check.

    **Retention is not task completion.** A check asks whether an expected fact,
    constraint, or decision survived into the response. It does not ask whether
    the continuation was any good, and passing every check in a scenario does not
    establish that the agent could have carried the work forward. The two are
    reported separately and only the first is measured here.

    Matching is case-insensitive substring by default, which is crude and
    admitted to be so: it cannot tell a mention from a use, and a model that
    writes "we should not use Redis" satisfies a check looking for "Redis".
    Scenarios are written with that in mind, and where it is not good enough the
    honest answer is a judged question rather than a cleverer regular expression.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str = Field(min_length=1)
    category: MetricCategory
    description: str = Field(min_length=1)

    required: tuple[str, ...] = ()
    """Every one of these must appear."""

    required_any: tuple[str, ...] = ()
    """At least one of these must appear. For a fact with several spellings."""

    forbidden: tuple[str, ...] = ()
    """None of these may appear."""

    case_sensitive: bool = False

    @model_validator(mode="after")
    def _must_check_something(self) -> RetentionCheck:
        if not (self.required or self.required_any or self.forbidden):
            raise ValueError(f"check {self.check_id!r} asserts nothing")
        return self

    def evaluate(self, text: str) -> tuple[bool, tuple[str, ...]]:
        """Whether the text satisfies this check, and what was wrong if not."""
        haystack = text if self.case_sensitive else text.lower()

        def present(needle: str) -> bool:
            return (needle if self.case_sensitive else needle.lower()) in haystack

        problems: list[str] = []
        problems.extend(f"missing {value!r}" for value in self.required if not present(value))
        if self.required_any and not any(present(value) for value in self.required_any):
            problems.append(f"none of {list(self.required_any)} present")
        problems.extend(
            f"forbidden {value!r} present" for value in self.forbidden if present(value)
        )
        return not problems, tuple(problems)


class JudgedQuestion(BaseModel):
    """An expectation no substring match can settle.

    Recorded so a scenario can state what it actually cares about even where the
    deterministic checks cannot reach it. Answered only when a judge is supplied;
    otherwise reported as not evaluated, never as passed and never as failed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(min_length=1)
    category: MetricCategory
    question: str = Field(min_length=1)


class DownstreamTask(BaseModel):
    """What the next agent is asked to do.

    Deliberately separate from the conversation. The evaluation does not ask
    whether a fact survived into the context; it asks whether the agent could
    continue the work. The instruction here is the only thing besides the
    strategy's context that reaches the model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    kind: TaskKind
    instruction: str = Field(min_length=1)


class TaskKind(StrEnum):
    """A small starting set. New kinds need no change to the runner."""

    QUESTION_ANSWERING = "question_answering"
    CODE_CONTINUATION = "code_continuation"
    CODE_MODIFICATION = "code_modification"
    DECISION_CONTINUATION = "decision_continuation"
    CONSTRAINT_SENSITIVE = "constraint_sensitive"
    MULTI_STEP = "multi_step"


class EvaluationScenario(BaseModel):
    """A conversation, a task, and what a good continuation looks like."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    failure_modes: tuple[FailureMode, ...] = ()

    conversation: tuple[dict[str, Any], ...] = Field(min_length=1)
    """The session that happened, as records the JSON Lines importer accepts."""

    task: DownstreamTask
    retention_checks: tuple[RetentionCheck, ...] = ()
    completion: tuple[CompletionCriterion, ...] = ()
    """Objective conditions on a correct continuation. See ``completion.py``.

    Separate from ``retention_checks`` because they answer different questions:
    retention asks what survived, completion asks whether the agent acted on it.
    A scenario may have both, and the two are never collapsed into one number.
    """

    judged: tuple[JudgedQuestion, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _must_be_measurable(self) -> EvaluationScenario:
        if not (self.retention_checks or self.completion or self.judged):
            raise ValueError(
                f"scenario {self.scenario_id!r} defines no checks and no judged questions, "
                "so running it could not tell anyone anything"
            )
        return self

    @property
    def expectation_strings(self) -> tuple[str, ...]:
        """Everything only the evaluator may know.

        Used by the leakage tests to assert that none of it reaches a model.
        """
        values: list[str] = []
        for check in self.retention_checks:
            values.extend(check.required)
            values.extend(check.required_any)
            values.extend(check.forbidden)
            values.append(check.description)
        for criterion in self.completion:
            values.extend(criterion.expected)
            values.extend(criterion.alternatives)
            values.append(criterion.description)
        values.extend(question.question for question in self.judged)
        return tuple(values)


DownstreamTask.model_rebuild()
EvaluationScenario.model_rebuild()


__all__ = [
    "DownstreamTask",
    "EvaluationScenario",
    "FailureMode",
    "JudgedQuestion",
    "MetricCategory",
    "RetentionCheck",
    "TaskKind",
]
