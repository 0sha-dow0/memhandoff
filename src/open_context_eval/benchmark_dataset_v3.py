"""The benchmark dataset, version 3.

**v2 is not modified.** It is built from v2 the way v2 was built from v1: by
transformation, leaving the earlier version importable, unchanged, and asserted
so by test. The conversations, tasks, and failure-mode tags are v2's verbatim.
What changes is how the scorer is allowed to reach a verdict.

## Why there is a v3

The Phase 5.8 run against a real model produced ten failure findings. **Eight
were artefacts**, and every one of them came from the same mechanism: asserting
that something is *absent* from free text by substring match.

```
"Since Redis is explicitly prohibited, I recommend an in-memory cache"
        ^^^^^
        forbidden term present -> constraint scored as lost
```

The answer honours the constraint. Naming what it rejected is what a good answer
does, and the check cannot tell a mention from a use. `RetentionCheck` already
documented that limitation; what the run showed is that against a real model the
limitation is not an edge case but the common case, and that it is **biased**:
naming a rejected option requires knowing it was rejected, so the arms that
retained most trip it most. The reference condition collected three of the eight.

Deterministic fakes hid this for two phases. One fixed reply never contains a
prohibited term, so every negative check passed for every arm.

## The rule v3 applies

**Deterministic checks assert presence. Judged questions assert absence.**

Presence is decidable by substring: a required term is either in the text or it
is not, and a false pass needs the model to have written the term by accident.
Absence is not decidable that way, because the same term appears in an
endorsement and in a rejection.

So every negative substring assertion is removed, and where one carried the
whole meaning of a check it becomes a `JudgedQuestion` — which reports **not
evaluated** without a judge, rather than passed or failed. An unanswerable
question left unanswered is the honest outcome; the artefact was answering it
anyway.

**A cleverer pattern was rejected.** A heuristic skipping terms near a negation
would reclassify real failures too, and would be a guess dressed as a
measurement. `RetentionCheck`'s own documentation says the honest answer is a
judged question, and this is that.

## What that costs

**Hedging is no longer caught deterministically.** Phase 5.6 built completion
criteria to discriminate — the right answer present *and* the ruled-out options
absent — so that a response listing every remembered option failed. The absent
half is exactly the broken half. A response saying "use SQLite or PostgreSQL"
now satisfies the deterministic criterion for `reversed-decision`.

That is a real loss of power, and it is accepted rather than papered over: a
check that catches hedging and also fails correct answers was not measuring
hedging. The judged questions carry the discrimination, and until a judge is
configured the honest report is that it went unmeasured. Running v3 without a
judge measures less than v2 appeared to, and more than v2 actually did.
"""

from __future__ import annotations

from open_context_eval.benchmark_dataset import BENCHMARK_SCENARIOS as V2_SCENARIOS
from open_context_eval.completion import CompletionCriterion
from open_context_eval.scenario import (
    EvaluationScenario,
    JudgedQuestion,
    MetricCategory,
    RetentionCheck,
)

BENCHMARK_DATASET_V3_VERSION = "v3"

STRANDED: dict[str, tuple[JudgedQuestion, ...]] = {
    "negative-constraint": (
        JudgedQuestion(
            question_id="redis-not-recommended",
            category=MetricCategory.CONSTRAINT,
            question=(
                "The conversation prohibited Redis outright and never said what to use "
                "instead. Does the response avoid recommending Redis? Answer yes if it "
                "names Redis only to note that it is excluded, and no only if it actually "
                "proposes using Redis."
            ),
        ),
    ),
    "failed-approach": (
        JudgedQuestion(
            question_id="in-memory-sort-not-retried",
            category=MetricCategory.FAILED_APPROACH,
            question=(
                "Loading the whole table into memory and sorting it there was tried and ran "
                "out of memory, and the conversation never said what to do instead. Does "
                "the response avoid proposing that approach again? Answer yes if it names "
                "the approach only to rule it out, and no only if it actually proposes "
                "doing it."
            ),
        ),
    ),
}
"""Hand-written, for the two scenarios that lose everything.

**Phrased so that YES is the desirable answer**, like every other judged
question. The harness records `passed` directly from the verdict, so a question
asking "does the response recommend Redis?" would score a correct answer — one
that recommends nothing of the sort — as a failure. An earlier draft did exactly
that, and a real judge caught it by answering NO with the reasoning "the
response does not mention Redis at all", which the harness dutifully filed as a
failed check.

Both state a prohibition and never say what to do instead, so the space of
acceptable answers is open and there is no positive assertion to fall back on.
They were only ever measurable by the mechanism that turned out to be broken,
which is why they need a question written for them rather than one derived from
an expected answer that does not exist.
"""

_KIND_CATEGORY: dict[str, MetricCategory] = {
    "exact_value": MetricCategory.CRITICAL_FACT,
    "expected_decision": MetricCategory.DECISION,
    "forbidden_action": MetricCategory.CONSTRAINT,
    "expected_entity": MetricCategory.ENTITY,
    "expected_state": MetricCategory.TEMPORAL_STATE,
    "expected_output_property": MetricCategory.CONSTRAINT,
    "expected_code_change": MetricCategory.OPEN_TASK,
}
"""Which retention category a completion kind's hedging question belongs to.

So a judged result lands in the same bucket the deterministic checks use, rather
than in a category invented for it.
"""


def _hedging_question(criterion: CompletionCriterion) -> JudgedQuestion:
    """The discrimination a criterion loses when its ``alternatives`` go.

    **Derived from the criterion rather than written afresh.** It restates that
    criterion's own description and the exact options that were removed, so the
    question cannot drift from what it replaced, and adding a criterion cannot
    leave a silently unreplaced assertion behind.

    The closing sentences are the whole point. They tell the judge two things
    the wording must not leave implicit: that naming a rejected option is not a
    violation, and that **silence about an option is not a violation either**.

    An earlier draft asked whether the response "commits to" the answer "rather
    than also offering" the alternatives, and a real judge read that as
    requiring an explicit denunciation — failing an answer that used the right
    figure and simply never mentioned the wrong one, with the reasoning "does
    not rule out 99.9 percent". Absence of hedging is what this asks about, and
    absence had to be said out loud.
    """
    ruled_out = ", ".join(criterion.alternatives)
    return JudgedQuestion(
        question_id=f"{criterion.criterion_id}-not-hedged",
        category=_KIND_CATEGORY[criterion.kind.value],
        question=(
            f"{criterion.description} Does the response settle on that without ALSO "
            f"putting forward any of these as something to do: {ruled_out}? "
            "Answer yes if none of them is put forward — including when the response "
            "never mentions them at all, and including when it mentions one only to "
            "rule it out. Answer no only if the response actually proposes one of them "
            "as an option. Not mentioning something is not the same as failing to "
            "rule it out."
        ),
    )


def _positive_only(check: RetentionCheck) -> RetentionCheck | None:
    """A check with its negative half removed, or nothing if that was all of it.

    A check keeping a `required` or `required_any` list still measures something
    after `forbidden` goes; one that was only a `forbidden` list has nothing left
    and is dropped, its meaning carried by a judged question instead.
    """
    if not check.forbidden:
        return check
    if not check.required and not check.required_any:
        return None
    return check.model_copy(update={"forbidden": ()})


def _positive_criterion(criterion: CompletionCriterion) -> CompletionCriterion | None:
    """The same rule for a completion criterion.

    ``alternatives`` is the criterion's negative half and is removed. One with no
    ``expected`` was purely negative and is dropped.
    """
    if not criterion.alternatives:
        return criterion
    if not criterion.expected:
        return None
    return criterion.model_copy(update={"alternatives": ()})


def _upgrade(scenario: EvaluationScenario) -> EvaluationScenario:
    checks = tuple(
        upgraded
        for upgraded in (_positive_only(check) for check in scenario.retention_checks)
        if upgraded is not None
    )
    completion = tuple(
        upgraded
        for upgraded in (_positive_criterion(item) for item in scenario.completion)
        if upgraded is not None
    )
    hedging = tuple(
        _hedging_question(criterion)
        for criterion in scenario.completion
        if criterion.alternatives and criterion.expected
    )
    return scenario.model_copy(
        update={
            "dataset_version": BENCHMARK_DATASET_V3_VERSION,
            "retention_checks": checks,
            "completion": completion,
            "judged": scenario.judged + STRANDED.get(scenario.scenario_id, ()) + hedging,
        }
    )


BENCHMARK_SCENARIOS_V3: tuple[EvaluationScenario, ...] = tuple(
    _upgrade(scenario) for scenario in V2_SCENARIOS
)


def by_id(scenario_id: str) -> EvaluationScenario:
    for scenario in BENCHMARK_SCENARIOS_V3:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(
        f"no v3 scenario {scenario_id!r}; have {[s.scenario_id for s in BENCHMARK_SCENARIOS_V3]}"
    )


__all__ = [
    "BENCHMARK_DATASET_V3_VERSION",
    "BENCHMARK_SCENARIOS_V3",
    "STRANDED",
    "by_id",
]
