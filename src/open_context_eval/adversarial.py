"""Dataset v4: conversations where compaction actually has to choose.

Phase 8.5. The benchmark's job is to distinguish strategies, and through Phases
5.8 and 8 it could not.

## Why v3 could not discriminate

Its scenarios are 220 to 385 tokens across 18 to 32 messages. At the budget the
benchmark used, that is **1.4x to 2.4x compression** — so little pressure that a
plain one-call summary keeps essentially everything, and every arm scores near
1.00. Two separate findings followed from that and neither was about compaction:

- On a 70B model **every arm scored 1.00 on every measure**, a ceiling effect
  indistinguishable from a real measurement.
- On an 8B model the arms separated by 0.08 — and two runs of the *same*
  configuration also differed by 0.08, so the separation was noise.

A benchmark whose scenarios fit comfortably inside the budget is not measuring
compaction. It is measuring whether the model can read.

## What v4 changes

**Length.** Conversations run to thousands of tokens across a hundred or more
messages, so a 160-token budget is 15x to 30x compression and something must
genuinely be dropped.

**Filler that resists summarization.** v1's filler repeats one sentence with a
changing index, which a summarizer collapses to a line at no cost — it consumes
prompt space without consuming *summary* space, so it never forces a choice.
v4's filler is varied, plausible, and individually forgettable, which is what
real sessions bury things under.

**Position is a variable, not an accident.** Each trap is planted at a known
place — start, middle, or end — because "recency wins" and "primacy wins" are
different failures with different fixes, and a dataset that scatters critical
information cannot tell them apart.

## The rule v3 established is kept

Deterministic checks assert **presence**; judged questions assert **absence**.
Nothing here asserts that a string is missing, because a correct answer that
names what it rejected would fail such a check — the defect that made eight of
ten findings false in Phase 5.8.
"""

from __future__ import annotations

from typing import Any

from open_context_eval.completion import CompletionCriterion, CompletionKind
from open_context_eval.scenario import (
    DownstreamTask,
    EvaluationScenario,
    FailureMode,
    JudgedQuestion,
    MetricCategory,
    RetentionCheck,
    TaskKind,
)

ADVERSARIAL_DATASET_VERSION = "v4"

MINIMUM_SUBSTANCE_WORDS = 12

_TOPICS = (
    "log formatting",
    "test fixture naming",
    "CI cache keys",
    "lint rule exceptions",
    "changelog wording",
    "docstring style",
    "commit message prefixes",
    "dependency pinning",
    "editor config",
    "branch naming",
    "PR template fields",
    "code review checklists",
    "type stub coverage",
    "fixture teardown order",
    "log sampling rates",
    "retry jitter defaults",
)

_PACKAGES = ("worker", "api", "cli", "scheduler", "ingest", "reporting")

_NOTES = (
    "we kept the existing pattern and moved on",
    "nobody had a strong opinion so it stayed as it was",
    "it was already consistent with the rest of the repo",
    "the change would have been cosmetic, so we skipped it",
    "we agreed to revisit it if it ever caused a problem",
    "it turned out not to matter for our case",
)


def noise(turns: int, seed: int = 0) -> list[dict[str, Any]]:
    """Varied, plausible, individually forgettable conversation.

    **Deterministic, and deliberately not repetitive.** v1's filler repeats one
    sentence with a changing index; a summarizer collapses that to a single line
    at no cost, so it fills the prompt without ever forcing the summary to
    choose what to keep. Varied filler cannot be collapsed that cheaply, which
    is the whole point of burying something in it.

    The index steps topic, package, and note on coprime cycles so the
    combination does not repeat inside a scenario. An earlier version stepped
    the note by three through a six-element list, which cycles after two, and a
    test caught it producing 24 distinct messages out of 40.

    No randomness: the same call produces the same conversation on any machine,
    because a benchmark whose inputs move cannot be rerun.
    """
    out: list[dict[str, Any]] = []
    for turn in range(turns):
        index = seed + turn
        topic = _TOPICS[index % len(_TOPICS)]
        package = _PACKAGES[(index // len(_TOPICS)) % len(_PACKAGES)]
        note = _NOTES[(index * 5) % len(_NOTES)]
        out.append(
            {
                "role": "user",
                "content": (
                    f"Quick one on {topic} — should we change how it is handled in the "
                    f"{package} package?"
                ),
            }
        )
        out.append(
            {
                "role": "assistant",
                "content": (
                    f"I checked {topic} in the {package} package. {note.capitalize()}. "
                    "There is nothing further worth recording about it."
                ),
            }
        )
    return out


def _scenario(**kwargs: object) -> EvaluationScenario:
    return EvaluationScenario(dataset_version=ADVERSARIAL_DATASET_VERSION, **kwargs)


# ----------------------------------------------------------------------
# Position: the same trap, planted early, in the middle, and late.


def _positional(
    scenario_id: str,
    where: str,
    before: int,
    after: int,
) -> EvaluationScenario:
    """One critical requirement, buried at a controlled depth.

    The three positions share a requirement, a task, and a check, and differ
    only in where the requirement sits. Anything that separates them is a
    position effect rather than a property of the content, which is what makes
    a comparison across them mean something.
    """
    requirement = {
        "role": "user",
        "content": (
            "Important, and I will only say it once: the export must be written as "
            "UTF-16LE with a byte order mark. The downstream mainframe reader cannot "
            "parse UTF-8 at all and silently produces empty rows."
        ),
    }
    return _scenario(
        scenario_id=scenario_id,
        description=f"A stated-once encoding requirement, buried {where} in a long session.",
        failure_modes=(
            FailureMode.EARLY_CRITICAL_FACT if where == "early" else FailureMode.LATE_CRITICAL_FACT,
            FailureMode.LONG_IRRELEVANT,
        ),
        conversation=[
            {"role": "system", "content": "You are helping build a reporting export."},
            *noise(before, seed=1),
            requirement,
            {"role": "assistant", "content": "Understood, UTF-16LE with a BOM."},
            *noise(after, seed=40),
            {"role": "user", "content": "Now write the export writer."},
        ],
        task=DownstreamTask(
            task_id=f"{scenario_id}-write-export",
            kind=TaskKind.CODE_CONTINUATION,
            instruction="Write the export writer, stating the encoding it uses.",
        ),
        retention_checks=(
            RetentionCheck(
                check_id="utf16-required",
                category=MetricCategory.CRITICAL_FACT,
                description="The UTF-16LE requirement survived.",
                required_any=("utf-16", "utf16"),
            ),
        ),
        completion=(
            CompletionCriterion(
                criterion_id="uses-utf16",
                kind=CompletionKind.EXPECTED_OUTPUT_PROPERTY,
                description="The writer uses UTF-16LE.",
                expected=("utf-16", "utf16"),
                min_words=MINIMUM_SUBSTANCE_WORDS,
            ),
        ),
        judged=(
            JudgedQuestion(
                question_id="encoding-not-hedged",
                category=MetricCategory.CRITICAL_FACT,
                question=(
                    "The session required UTF-16LE with a BOM. Does the response settle on "
                    "that encoding without also offering UTF-8 or another encoding as an "
                    "option? Answer yes if no alternative is put forward, including when "
                    "one is mentioned only to rule it out."
                ),
            ),
        ),
    )


EARLY = _positional("adv-critical-early", "early", before=2, after=48)
MIDDLE = _positional("adv-critical-middle", "in the middle", before=25, after=25)
LATE = _positional("adv-critical-late", "late", before=48, after=2)


# ----------------------------------------------------------------------

REVERSED_TWICE = _scenario(
    scenario_id="adv-reversed-twice",
    description="A decision made, reversed, and reversed again across a long session.",
    failure_modes=(FailureMode.REVERSED_DECISION, FailureMode.TEMPORAL_STATE),
    conversation=[
        {"role": "user", "content": "Which queue should the scheduler use?"},
        {"role": "assistant", "content": "Let us use RabbitMQ. The ops team already runs it."},
        *noise(20, seed=2),
        {
            "role": "user",
            "content": (
                "Change of plan on the queue: RabbitMQ is out, ops are decommissioning it "
                "this quarter. Use Amazon SQS instead."
            ),
        },
        {"role": "assistant", "content": "Understood, SQS."},
        *noise(20, seed=60),
        {
            "role": "user",
            "content": (
                "Final word on the queue, and this one is settled: SQS is out too, because "
                "the service has to run on-premise with no cloud dependency. We are using "
                "a Postgres-backed job table."
            ),
        },
        {"role": "assistant", "content": "Understood, a Postgres job table it is."},
        *noise(10, seed=90),
        {"role": "user", "content": "Write the scheduler's enqueue function."},
    ],
    task=DownstreamTask(
        task_id="adv-enqueue",
        kind=TaskKind.CODE_CONTINUATION,
        instruction="Write the enqueue function for the queue we settled on.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="final-queue",
            category=MetricCategory.DECISION,
            description="The final decision, not either superseded one.",
            required_any=("postgres", "postgresql", "job table"),
        ),
    ),
    completion=(
        CompletionCriterion(
            criterion_id="uses-postgres-queue",
            kind=CompletionKind.EXPECTED_DECISION,
            description="Enqueues into the Postgres job table.",
            expected=("postgres", "postgresql", "job table"),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    judged=(
        JudgedQuestion(
            question_id="settled-on-postgres",
            category=MetricCategory.DECISION,
            question=(
                "The queue decision changed twice and settled on a Postgres-backed job "
                "table. Does the response use that, without also proposing RabbitMQ or SQS "
                "as something to do? Answer yes if neither is put forward, including when "
                "one is named only to note it was ruled out."
            ),
        ),
    ),
)

EXACT_VALUES = _scenario(
    scenario_id="adv-exact-values",
    description="Several similar exact numbers, only one of which is the answer.",
    failure_modes=(FailureMode.EXACT_VALUE, FailureMode.SIMILAR_ENTITIES),
    conversation=[
        {"role": "system", "content": "You are helping tune a batch pipeline."},
        *noise(15, seed=3),
        {
            "role": "user",
            "content": (
                "Numbers for the pipeline, and they are all different so please keep them "
                "straight: the read timeout is 47 seconds, the write timeout is 74 seconds, "
                "the connect timeout is 7 seconds, and the batch size is 4700 rows."
            ),
        },
        {"role": "assistant", "content": "Noted: read 47s, write 74s, connect 7s, batch 4700."},
        *noise(25, seed=70),
        {
            "role": "user",
            "content": "Write the client configuration, using the write timeout specifically.",
        },
    ],
    task=DownstreamTask(
        task_id="adv-write-timeout",
        kind=TaskKind.CODE_CONTINUATION,
        instruction="Write the client configuration, naming the write timeout value.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="write-timeout-74",
            category=MetricCategory.CRITICAL_FACT,
            description="The write timeout is 74 seconds.",
            required=("74",),
        ),
    ),
    completion=(
        CompletionCriterion(
            criterion_id="uses-74",
            kind=CompletionKind.EXACT_VALUE,
            description="Uses 74 seconds as the write timeout.",
            expected=("74",),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    judged=(
        JudgedQuestion(
            question_id="right-timeout-chosen",
            category=MetricCategory.CRITICAL_FACT,
            question=(
                "The write timeout is 74 seconds; 47, 7, and 4700 are other values from the "
                "same session. Does the response use 74 as the write timeout, rather than "
                "one of the others? Answer yes if 74 is the value used."
            ),
        ),
    ),
)

FAILED_APPROACH = _scenario(
    scenario_id="adv-failed-approach",
    description="An approach was tried, failed for a stated reason, and must not return.",
    failure_modes=(FailureMode.FAILED_APPROACH, FailureMode.LONG_IRRELEVANT),
    conversation=[
        {"role": "user", "content": "The nightly reconciliation job is too slow."},
        {
            "role": "assistant",
            "content": (
                "I tried doing the whole reconciliation in a single SQL statement with a "
                "correlated subquery per row. It ran for six hours and was killed by the "
                "job timeout. The query planner will not use the index inside that "
                "subquery, so that approach cannot work here."
            ),
        },
        *noise(30, seed=4),
        {"role": "user", "content": "Pick the reconciliation work back up."},
    ],
    task=DownstreamTask(
        task_id="adv-reconciliation",
        kind=TaskKind.DECISION_CONTINUATION,
        instruction="Propose the next thing to try for the nightly reconciliation.",
    ),
    retention_checks=(),
    completion=(),
    judged=(
        JudgedQuestion(
            question_id="no-correlated-subquery",
            category=MetricCategory.FAILED_APPROACH,
            question=(
                "A single SQL statement with a correlated subquery per row was already "
                "tried and failed, because the planner will not use the index inside the "
                "subquery. Does the response avoid proposing that approach again? Answer "
                "yes if it does not propose it, including when it names the approach only "
                "to rule it out."
            ),
        ),
    ),
)

TOOL_HEAVY = _scenario(
    scenario_id="adv-tool-heavy",
    description="The answer is inside one tool result among many.",
    failure_modes=(FailureMode.TOOL_RESULT, FailureMode.SIMILAR_ENTITIES),
    conversation=[
        {"role": "system", "content": "You are investigating a failing deployment."},
        *noise(10, seed=5),
        {"role": "tool_call", "content": "kubectl get pods -n staging", "tool_name": "shell"},
        {
            "role": "tool_result",
            "content": (
                "api-7c9f 1/1 Running; worker-3b2a 0/1 CrashLoopBackOff; cron-91fe 1/1 Running"
            ),
            "tool_name": "shell",
        },
        {
            "role": "tool_call",
            "content": "kubectl logs worker-3b2a -n staging",
            "tool_name": "shell",
        },
        {
            "role": "tool_result",
            "content": (
                "FATAL: could not bind to 0.0.0.0:9443 — address already in use. "
                "The sidecar is already listening on 9443."
            ),
            "tool_name": "shell",
        },
        *noise(25, seed=80),
        {"role": "user", "content": "So what is actually breaking the worker?"},
    ],
    task=DownstreamTask(
        task_id="adv-diagnose",
        kind=TaskKind.QUESTION_ANSWERING,
        instruction="Explain what is breaking the worker pod, naming the port involved.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="port-9443",
            category=MetricCategory.CRITICAL_FACT,
            description="The conflicting port from the tool result survived.",
            required=("9443",),
        ),
    ),
    completion=(
        CompletionCriterion(
            criterion_id="names-port-conflict",
            kind=CompletionKind.EXPECTED_OUTPUT_PROPERTY,
            description="Identifies the port bind conflict on 9443.",
            expected=("9443",),
            min_words=MINIMUM_SUBSTANCE_WORDS,
        ),
    ),
    judged=(
        JudgedQuestion(
            question_id="diagnoses-bind-conflict",
            category=MetricCategory.CRITICAL_FACT,
            question=(
                "The worker fails because it cannot bind port 9443, which the sidecar "
                "already holds. Does the response identify that as the cause?"
            ),
        ),
    ),
)

NEGATIVE_CONSTRAINT = _scenario(
    scenario_id="adv-negative-constraint",
    description="A prohibition stated once early, then buried, then a task that invites it.",
    failure_modes=(FailureMode.NEGATIVE_CONSTRAINT, FailureMode.LONG_IRRELEVANT),
    conversation=[
        {
            "role": "user",
            "content": (
                "Ground rule before we start: this service must not write to the shared "
                "NFS mount under any circumstances. Two outages last year came from that, "
                "and the storage team has forbidden it."
            ),
        },
        {"role": "assistant", "content": "Understood, no writes to the shared NFS mount."},
        *noise(35, seed=6),
        {"role": "user", "content": "We need somewhere to put the generated report files."},
    ],
    task=DownstreamTask(
        task_id="adv-report-storage",
        kind=TaskKind.CONSTRAINT_SENSITIVE,
        instruction="Recommend where the generated report files should be written.",
    ),
    retention_checks=(),
    completion=(),
    judged=(
        JudgedQuestion(
            question_id="nfs-avoided",
            category=MetricCategory.CONSTRAINT,
            question=(
                "The session forbade writing to the shared NFS mount. Does the response "
                "avoid recommending that the report files be written there? Answer yes if "
                "it does not recommend it, including when it names the mount only to note "
                "it is excluded."
            ),
        ),
    ),
)

ADVERSARIAL_SCENARIOS: tuple[EvaluationScenario, ...] = (
    EARLY,
    MIDDLE,
    LATE,
    REVERSED_TWICE,
    EXACT_VALUES,
    FAILED_APPROACH,
    TOOL_HEAVY,
    NEGATIVE_CONSTRAINT,
)


def by_id(scenario_id: str) -> EvaluationScenario:
    for scenario in ADVERSARIAL_SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(
        f"no adversarial scenario {scenario_id!r}; have "
        f"{[s.scenario_id for s in ADVERSARIAL_SCENARIOS]}"
    )


__all__ = [
    "ADVERSARIAL_DATASET_VERSION",
    "ADVERSARIAL_SCENARIOS",
    "MINIMUM_SUBSTANCE_WORDS",
    "by_id",
    "noise",
]
