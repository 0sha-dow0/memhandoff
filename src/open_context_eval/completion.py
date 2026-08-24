"""Did the agent do the right thing?

Separate from retention, and deliberately a different shape.

**Retention** asks whether a piece of information survived into the response.
**Completion** asks whether the agent acted correctly on it. A response that
mentions SQLite has retained the decision; a response that mentions SQLite *and*
does not also propose PostgreSQL has acted on it. The second is what a handover
is for, and it is not a substring match — it is a discrimination between the
right answer and the wrong ones the conversation ruled out.

Three things make a criterion more than a keyword search:

* ``alternatives`` must be **absent**. This is what turns "the word appears"
  into "the agent chose correctly", and it is the half that catches an answer
  hedging across every option it can remember.
* ``min_words`` sets a substance floor, so a one-word reply cannot satisfy a
  task that asked for a change to be described.
* **Criteria are conjunctive.** A scenario is completed when every criterion
  passes, not when most do. Partial credit is what retention is for.

**A criterion that cannot be checked is not a failure.** ``requires_execution``
marks a criterion needing a sandbox nobody has built; it reports
``not_evaluable``, and a scenario containing one cannot be scored complete or
incomplete. Guessing either way would be worse than admitting the gap.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CompletionKind(StrEnum):
    """What sort of correctness a criterion is checking.

    A label for reporting and for grouping failures by type. The evaluation
    mechanism is shared; a kind does not get its own algorithm, because seven
    slightly different string matchers would be seven things to get wrong.
    """

    EXACT_VALUE = "exact_value"
    EXPECTED_DECISION = "expected_decision"
    FORBIDDEN_ACTION = "forbidden_action"
    EXPECTED_ENTITY = "expected_entity"
    EXPECTED_STATE = "expected_state"
    EXPECTED_OUTPUT_PROPERTY = "expected_output_property"
    EXPECTED_CODE_CHANGE = "expected_code_change"


class CompletionStatus(StrEnum):
    """How a criterion came out.

    ``NOT_EVALUABLE`` is a first-class outcome rather than a silent pass or a
    charitable fail.
    """

    PASSED = "passed"
    OMITTED = "omitted"
    """The right answer is not there."""

    FABRICATED = "fabricated"
    """Something the conversation ruled out is there.

    Reported separately from omission because they are different failures with
    different causes: forgetting versus asserting. The fabrication count is what
    the deterministic hallucination signal is built from.
    """

    INSUBSTANTIAL = "insubstantial"
    """Technically present, but too thin to be a continuation of the work."""

    NOT_EVALUABLE = "not_evaluable"


class CompletionCriterion(BaseModel):
    """One objective condition on a correct continuation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion_id: str = Field(min_length=1)
    kind: CompletionKind
    description: str = Field(min_length=1)

    expected: tuple[str, ...] = ()
    """At least one must appear. Several entries are spellings of one answer."""

    alternatives: tuple[str, ...] = ()
    """None may appear. The options the conversation ruled out."""

    min_words: int = 0
    """Substance floor for the whole response."""

    requires_execution: bool = False
    """Needs a sandbox this project does not have. Reported, never guessed."""

    @model_validator(mode="after")
    def _must_be_checkable(self) -> CompletionCriterion:
        if self.requires_execution:
            return self
        if not (self.expected or self.alternatives or self.min_words):
            raise ValueError(f"criterion {self.criterion_id!r} asserts nothing")
        return self

    def evaluate(self, response: str) -> tuple[CompletionStatus, str]:
        """Judge one response. Returns the outcome and why."""
        if self.requires_execution:
            return (
                CompletionStatus.NOT_EVALUABLE,
                "needs code execution, which this project does not do",
            )

        lowered = response.lower()
        fabricated = [value for value in self.alternatives if value.lower() in lowered]
        if fabricated:
            return (
                CompletionStatus.FABRICATED,
                f"used {fabricated}, which the conversation ruled out",
            )

        if self.expected and not any(value.lower() in lowered for value in self.expected):
            return CompletionStatus.OMITTED, f"none of {list(self.expected)} present"

        if self.min_words and len(response.split()) < self.min_words:
            return (
                CompletionStatus.INSUBSTANTIAL,
                f"{len(response.split())} words, fewer than the {self.min_words} required",
            )

        return CompletionStatus.PASSED, ""


def scenario_completed(
    outcomes: tuple[CompletionStatus, ...],
) -> bool | None:
    """Whether a scenario's task was completed.

    Conjunctive: every criterion must pass. ``None`` when any criterion could not
    be evaluated, because a task whose success conditions were only partly
    checked has not been shown either way.
    """
    if not outcomes:
        return None
    if any(status is CompletionStatus.NOT_EVALUABLE for status in outcomes):
        return None
    return all(status is CompletionStatus.PASSED for status in outcomes)


__all__ = [
    "CompletionCriterion",
    "CompletionKind",
    "CompletionStatus",
    "scenario_completed",
]
