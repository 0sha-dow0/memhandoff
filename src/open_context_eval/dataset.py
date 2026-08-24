"""The synthetic evaluation dataset.

Fifteen small scenarios, each built around one way a summary loses something
that mattered. They are written, not sampled: no private conversation is used
and nothing is downloaded.

**Versioned, and the version means something.** ``DATASET_VERSION`` must be
incremented whenever a scenario changes materially, because a result file names
the version it ran against and comparing across a silent edit would compare two
different benchmarks. A run refuses to mix versions.

**Small enough for CI.** Filler is generated deterministically rather than
written out, so a scenario can contain a long irrelevant stretch without the
file containing one.

Nothing here records a result. The dataset states what a good continuation
looks like; whether any strategy achieves it is not known and is not claimed.
"""

from __future__ import annotations

from typing import Any

from open_context_eval.scenario import (
    DownstreamTask,
    EvaluationScenario,
    FailureMode,
    JudgedQuestion,
    MetricCategory,
    RetentionCheck,
    TaskKind,
)

DATASET_VERSION = "v1"


def filler(turns: int, topic: str = "logging") -> list[dict[str, Any]]:
    """Deterministic irrelevant conversation.

    Real sessions bury what matters under this. A scenario that omitted it would
    be easier than the thing being modelled.
    """
    out: list[dict[str, Any]] = []
    for i in range(turns):
        out.append({"role": "user", "content": f"Small question {i} about {topic} formatting."})
        out.append(
            {
                "role": "assistant",
                "content": (
                    f"Answer {i}: adjust the {topic} formatter, keep the existing pattern, "
                    "and there is nothing else worth noting about it."
                ),
            }
        )
    return out


def _scenario(**kwargs: object) -> EvaluationScenario:
    return EvaluationScenario(dataset_version=DATASET_VERSION, **kwargs)


EARLY_CRITICAL_FACT = _scenario(
    scenario_id="early-critical-fact",
    description="A hard requirement is stated once at the start, then buried.",
    failure_modes=(FailureMode.EARLY_CRITICAL_FACT, FailureMode.LONG_IRRELEVANT),
    conversation=[
        {"role": "system", "content": "You are helping build an ingestion service."},
        {
            "role": "user",
            "content": (
                "Before anything else: this service must run entirely offline. "
                "It has no outbound network access at all, ever."
            ),
        },
        {"role": "assistant", "content": "Understood, offline only."},
        *filler(14),
        {"role": "user", "content": "Now let us finish the ingestion pipeline."},
    ],
    task=DownstreamTask(
        task_id="propose-ingestion",
        kind=TaskKind.CONSTRAINT_SENSITIVE,
        instruction=(
            "Propose how the ingestion service should fetch its reference data, and state "
            "any constraints you must respect."
        ),
    ),
    retention_checks=(
        RetentionCheck(
            check_id="offline-respected",
            category=MetricCategory.CONSTRAINT,
            description="The offline requirement stated at the start survives.",
            required_any=("offline", "no network", "without network"),
        ),
    ),
)

LATE_CRITICAL_FACT = _scenario(
    scenario_id="late-critical-fact",
    description="The deciding detail arrives in the last few turns.",
    failure_modes=(FailureMode.LATE_CRITICAL_FACT,),
    conversation=[
        {"role": "system", "content": "You are helping tune a batch job."},
        *filler(12),
        {
            "role": "user",
            "content": "One more thing before you continue: the job must finish inside 45 minutes.",
        },
        {"role": "assistant", "content": "Noted, a 45 minute ceiling."},
    ],
    task=DownstreamTask(
        task_id="tune-batch",
        kind=TaskKind.CONSTRAINT_SENSITIVE,
        instruction="Suggest a batch size, and state the runtime budget you are working to.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="deadline-kept",
            category=MetricCategory.CRITICAL_FACT,
            description="The 45 minute ceiling survives.",
            required_any=("45 minute", "45 minutes", "45-minute"),
        ),
    ),
)

NEGATIVE_CONSTRAINT = _scenario(
    scenario_id="negative-constraint",
    description="A thing was forbidden, then discussed at length anyway.",
    failure_modes=(FailureMode.NEGATIVE_CONSTRAINT,),
    conversation=[
        {"role": "user", "content": "We need a cache in front of the read path."},
        {
            "role": "user",
            "content": (
                "Do NOT use Redis. Operations refuse to run another stateful service, "
                "and that decision is final."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Understood. For reference, Redis would have offered pub/sub and TTL "
                "eviction, but we will not be using Redis."
            ),
        },
        *filler(8, topic="metrics"),
        {"role": "user", "content": "Let us settle the caching approach."},
    ],
    task=DownstreamTask(
        task_id="choose-cache",
        kind=TaskKind.CONSTRAINT_SENSITIVE,
        instruction="Recommend a caching approach for the read path and justify it briefly.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="redis-avoided",
            category=MetricCategory.CONSTRAINT,
            description="The prohibition on Redis is respected.",
            forbidden=("redis",),
        ),
    ),
)

REVERSED_DECISION = _scenario(
    scenario_id="reversed-decision",
    description="A decision is made, then explicitly reversed later.",
    failure_modes=(FailureMode.REVERSED_DECISION, FailureMode.TEMPORAL_STATE),
    conversation=[
        {"role": "user", "content": "Which database should the service use?"},
        {"role": "assistant", "content": "Let us use PostgreSQL. It matches the existing stack."},
        *filler(10, topic="test fixtures"),
        {
            "role": "user",
            "content": (
                "We have changed that decision. PostgreSQL is out; the deployment target has "
                "no server. Use SQLite instead."
            ),
        },
        {"role": "assistant", "content": "Understood, SQLite it is."},
    ],
    task=DownstreamTask(
        task_id="write-schema",
        kind=TaskKind.DECISION_CONTINUATION,
        instruction="Write the connection setup for the database we settled on.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="latest-decision",
            category=MetricCategory.DECISION,
            description="SQLite is used and PostgreSQL is not.",
            required=("sqlite",),
            forbidden=("postgresql", "postgres"),
        ),
    ),
)

FAILED_APPROACH = _scenario(
    scenario_id="failed-approach",
    description="An approach was tried and failed for a specific reason.",
    failure_modes=(FailureMode.FAILED_APPROACH,),
    conversation=[
        {"role": "user", "content": "The nightly join is too slow."},
        {
            "role": "assistant",
            "content": (
                "I tried loading the whole table into memory and sorting it there. It ran out "
                "of memory at 40GB and the box has 16GB. That approach cannot work here."
            ),
        },
        *filler(8, topic="cron"),
        {"role": "user", "content": "Pick up the join work again."},
    ],
    task=DownstreamTask(
        task_id="speed-up-join",
        kind=TaskKind.MULTI_STEP,
        instruction="Propose a next step for making the nightly join faster.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="no-repeat-in-memory",
            category=MetricCategory.FAILED_APPROACH,
            description="The in-memory sort is not proposed again.",
            forbidden=("load the whole table into memory", "in-memory sort", "sort it in memory"),
        ),
    ),
)

EXACT_VALUE = _scenario(
    scenario_id="exact-value",
    description="A precise number decides the answer.",
    failure_modes=(FailureMode.EXACT_VALUE,),
    conversation=[
        {
            "role": "user",
            "content": (
                "The upstream provider drops connections at 17 seconds, so our client timeout "
                "must be 17 seconds exactly. Not 15, not 30."
            ),
        },
        {"role": "assistant", "content": "Client timeout pinned to 17 seconds."},
        *filler(10, topic="retries"),
    ],
    task=DownstreamTask(
        task_id="write-client",
        kind=TaskKind.CODE_CONTINUATION,
        instruction="Write the HTTP client configuration, including the timeout.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="timeout-exact",
            category=MetricCategory.CRITICAL_FACT,
            description="The timeout is 17 seconds and not a rounder number.",
            required=("17",),
            forbidden=("30 second", "60 second"),
        ),
    ),
)

SIMILAR_ENTITIES = _scenario(
    scenario_id="similar-entities",
    description="Two services differ by one word and only one is in scope.",
    failure_modes=(FailureMode.SIMILAR_ENTITIES,),
    conversation=[
        {
            "role": "user",
            "content": (
                "We run two services: billing-api and billing-worker. The memory leak is in "
                "billing-worker only. billing-api is healthy and must not be touched."
            ),
        },
        *filler(8, topic="dashboards"),
        {"role": "user", "content": "Continue the leak investigation."},
    ],
    task=DownstreamTask(
        task_id="investigate-leak",
        kind=TaskKind.MULTI_STEP,
        instruction="Name the service you will investigate and the first thing you will check.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="right-service",
            category=MetricCategory.ENTITY,
            description="billing-worker is investigated, not billing-api.",
            required=("billing-worker",),
        ),
    ),
)

TEMPORAL_STATE = _scenario(
    scenario_id="temporal-state",
    description="State changed twice; only the latest is true.",
    failure_modes=(FailureMode.TEMPORAL_STATE,),
    conversation=[
        {"role": "assistant", "content": "The migration is not started."},
        *filler(6, topic="naming"),
        {"role": "assistant", "content": "The migration is now half applied, stopped at step 3."},
        *filler(6, topic="linting"),
        {
            "role": "user",
            "content": "Update: the migration finished completely this morning. All steps applied.",
        },
    ],
    task=DownstreamTask(
        task_id="report-migration",
        kind=TaskKind.QUESTION_ANSWERING,
        instruction="What is the current state of the migration, and what is the next step?",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="current-state",
            category=MetricCategory.TEMPORAL_STATE,
            description="The migration is reported complete, not stalled at step 3.",
            required_any=("complete", "finished", "applied"),
            forbidden=("step 3", "not started", "half"),
        ),
    ),
)

TOOL_RESULT = _scenario(
    scenario_id="tool-result",
    description="The deciding fact came back from a tool, not from a person.",
    failure_modes=(FailureMode.TOOL_RESULT,),
    conversation=[
        {"role": "user", "content": "How big is the events table?"},
        {
            "type": "tool_call",
            "tool_name": "psql",
            "tool_call_id": "c1",
            "content": "SELECT count(*) FROM events;",
        },
        {
            "type": "tool_result",
            "tool_name": "psql",
            "tool_call_id": "c1",
            "content": "count = 4823991",
        },
        {"role": "assistant", "content": "So the events table has 4823991 rows."},
        *filler(8, topic="indexes"),
    ],
    task=DownstreamTask(
        task_id="plan-backfill",
        kind=TaskKind.QUESTION_ANSWERING,
        instruction="How many rows must the backfill process, and how will you batch it?",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="row-count",
            category=MetricCategory.CRITICAL_FACT,
            description="The row count measured by the tool survives.",
            required_any=("4823991", "4,823,991", "4.8 million", "4.8m"),
        ),
    ),
)

OPEN_TASK = _scenario(
    scenario_id="open-task",
    description="Three tasks; two are finished and one is not.",
    failure_modes=(FailureMode.OPEN_TASK,),
    conversation=[
        {
            "role": "user",
            "content": "Three things to do: rate limiting, audit log, and CSV export.",
        },
        {"role": "assistant", "content": "Rate limiting is done and merged."},
        *filler(6, topic="changelogs"),
        {"role": "assistant", "content": "The audit log is done too, deployed yesterday."},
        {"role": "user", "content": "The CSV export has not been started at all."},
        *filler(6, topic="fixtures"),
    ],
    task=DownstreamTask(
        task_id="continue-work",
        kind=TaskKind.MULTI_STEP,
        instruction="Which piece of work is outstanding, and how will you start it?",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="right-open-task",
            category=MetricCategory.OPEN_TASK,
            description="CSV export is identified as the unfinished work.",
            required=("csv",),
        ),
    ),
)

RATIONALE = _scenario(
    scenario_id="rationale",
    description="A decision only makes sense with the reason behind it.",
    failure_modes=(FailureMode.RATIONALE,),
    conversation=[
        {
            "role": "user",
            "content": (
                "We write timestamps as strings rather than native date types, because the "
                "downstream mainframe reader cannot parse anything else. It looks wrong and "
                "it is deliberate."
            ),
        },
        *filler(8, topic="serialization"),
        {"role": "user", "content": "Someone new is about to touch the export format."},
    ],
    task=DownstreamTask(
        task_id="explain-format",
        kind=TaskKind.QUESTION_ANSWERING,
        instruction=(
            "A colleague wants to change timestamps to native date types. What do you tell them?"
        ),
    ),
    retention_checks=(
        RetentionCheck(
            check_id="rationale-kept",
            category=MetricCategory.RATIONALE,
            description="The mainframe reason is given, not just the rule.",
            required_any=("mainframe", "downstream reader", "cannot parse"),
        ),
    ),
    judged=(
        JudgedQuestion(
            question_id="rationale-understood",
            category=MetricCategory.RATIONALE,
            question=(
                "Does the answer explain why string timestamps are required, "
                "not merely that they are?"
            ),
        ),
    ),
)

MULTI_STEP = _scenario(
    scenario_id="multi-step",
    description="Work with an established order that must be continued, not restarted.",
    failure_modes=(FailureMode.MULTI_STEP,),
    conversation=[
        {
            "role": "user",
            "content": (
                "Migration plan: 1 add column, 2 backfill, 3 add index, 4 drop old column."
            ),
        },
        {"role": "assistant", "content": "Step 1 is applied in production."},
        {"role": "assistant", "content": "Step 2 backfill completed over the weekend."},
        *filler(8, topic="alerts"),
        {"role": "user", "content": "Carry on with the plan."},
    ],
    task=DownstreamTask(
        task_id="next-migration-step",
        kind=TaskKind.MULTI_STEP,
        instruction="What is the next step in the migration plan, and what will you run?",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="correct-next-step",
            category=MetricCategory.OPEN_TASK,
            description="The index step is next, not a repeat of the backfill.",
            required=("index",),
        ),
    ),
)

REPEATED_INFORMATION = _scenario(
    scenario_id="repeated-information",
    description="A wrong figure is repeated often; the correction is stated once.",
    failure_modes=(FailureMode.REPEATED_INFORMATION, FailureMode.EXACT_VALUE),
    conversation=[
        {"role": "assistant", "content": "The SLA is 99.9 percent."},
        {"role": "assistant", "content": "Designing against the 99.9 percent SLA."},
        {"role": "assistant", "content": "Again, the 99.9 percent SLA drives this."},
        {
            "role": "user",
            "content": (
                "Correction, and this is the only time I will say it: the contract says 99.99 "
                "percent, not 99.9. Everything must be designed to 99.99."
            ),
        },
        *filler(8, topic="paging"),
    ],
    task=DownstreamTask(
        task_id="state-sla",
        kind=TaskKind.QUESTION_ANSWERING,
        instruction="What availability target are we designing to, and what does it allow monthly?",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="corrected-sla",
            category=MetricCategory.CRITICAL_FACT,
            description="The corrected 99.99 figure is used.",
            required=("99.99",),
        ),
    ),
)

CONFUSABLE_NUMBERS = _scenario(
    scenario_id="confusable-numbers",
    description="Several similar numbers, only one of which answers the task.",
    failure_modes=(FailureMode.EXACT_VALUE, FailureMode.SIMILAR_ENTITIES),
    conversation=[
        {
            "role": "user",
            "content": (
                "Port map: the API listens on 8080, the admin panel on 8081, and the metrics "
                "endpoint on 8082. The firewall rule we need is for the metrics endpoint."
            ),
        },
        *filler(8, topic="deployment"),
        {"role": "user", "content": "Write up the firewall change."},
    ],
    task=DownstreamTask(
        task_id="firewall-rule",
        kind=TaskKind.CODE_MODIFICATION,
        instruction="Write the firewall rule we need, naming the port.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="right-port",
            category=MetricCategory.ENTITY,
            description="Port 8082 is used, not the API or admin port.",
            required=("8082",),
        ),
    ),
)

DECISION_WITH_CONSTRAINT = _scenario(
    scenario_id="decision-with-constraint",
    description="A decision and a constraint that must both survive together.",
    failure_modes=(FailureMode.NEGATIVE_CONSTRAINT, FailureMode.REVERSED_DECISION),
    conversation=[
        {"role": "user", "content": "We considered Kafka for the event bus."},
        {
            "role": "user",
            "content": (
                "Kafka is rejected: single machine deployment, no cluster. We are using "
                "SQLite-backed queueing instead, and it must stay in-process."
            ),
        },
        *filler(10, topic="backpressure"),
        {"role": "user", "content": "Continue the event bus work."},
    ],
    task=DownstreamTask(
        task_id="event-bus",
        kind=TaskKind.DECISION_CONTINUATION,
        instruction="Describe how the event bus should be implemented.",
    ),
    retention_checks=(
        RetentionCheck(
            check_id="kafka-rejected",
            category=MetricCategory.CONSTRAINT,
            description="Kafka is not reintroduced.",
            forbidden=("kafka",),
        ),
        RetentionCheck(
            check_id="queue-decision",
            category=MetricCategory.DECISION,
            description="The in-process SQLite queue decision is carried forward.",
            required_any=("sqlite", "in-process", "in process"),
        ),
    ),
)

SCENARIOS: tuple[EvaluationScenario, ...] = (
    EARLY_CRITICAL_FACT,
    LATE_CRITICAL_FACT,
    NEGATIVE_CONSTRAINT,
    REVERSED_DECISION,
    FAILED_APPROACH,
    EXACT_VALUE,
    SIMILAR_ENTITIES,
    TEMPORAL_STATE,
    TOOL_RESULT,
    OPEN_TASK,
    RATIONALE,
    MULTI_STEP,
    REPEATED_INFORMATION,
    CONFUSABLE_NUMBERS,
    DECISION_WITH_CONSTRAINT,
)


def by_id(scenario_id: str) -> EvaluationScenario:
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(f"no scenario {scenario_id!r}; have {[s.scenario_id for s in SCENARIOS]}")


__all__ = ["DATASET_VERSION", "SCENARIOS", "by_id", "filler"]
