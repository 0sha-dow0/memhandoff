"""MemHandoff evaluation harness.

```
scenario -> archive -> strategy -> context -> same task -> same model -> metrics
```

**The primary evaluation target is downstream task continuation, not summary
quality.** A summary that reads well to a human, or that contains many true
facts, tells us nothing on its own. The question is whether a different agent,
given only the compacted representation, can carry the work forward.

**This package is not part of the runtime.** It depends on ``open_context``;
nothing in ``open_context`` depends on it, and it is excluded from the
distributed wheel. The dependency direction is checked by a test rather than
trusted.

Everything runs offline against the deterministic fakes from Phase 4.5: no key,
no network, no local model server, no database. Real-model runs go through the
same ``LLMProvider`` interface, and CI never needs one.

**No result in this repository is measured.** The harness builds the machinery
to produce numbers. It ships none, and nothing here should ever be filled in by
hand.
"""

from open_context_eval.benchmark import (
    BENCHMARK_STRATEGIES,
    DEFAULT_BUDGETS,
    DETERMINISTIC_TEST,
    REAL_MODEL,
    REFERENCE_BUDGET,
    BenchmarkConfig,
    InvalidBenchmarkError,
    check_comparable,
    estimated_requests,
    experiment_configs,
    run_benchmark,
    stream_benchmark,
)
from open_context_eval.benchmark_dataset import BENCHMARK_DATASET_VERSION, BENCHMARK_SCENARIOS
from open_context_eval.benchmark_report import (
    Cell,
    failure_modes_table,
    format_benchmark,
    format_failures,
    summarize_benchmark,
)
from open_context_eval.budget import (
    BudgetedProvider,
    RequestBudget,
    RequestBudgetExhausted,
)
from open_context_eval.completion import (
    CompletionCriterion,
    CompletionKind,
    CompletionStatus,
    scenario_completed,
)
from open_context_eval.dataset import DATASET_VERSION, SCENARIOS, by_id
from open_context_eval.judge import EvaluationJudge, JudgeVerdict, KeywordJudge
from open_context_eval.prompts import (
    CONTINUATION_PROMPT_V1,
    SIMPLE_SUMMARY_PROMPT_V1,
    build_continuation_request,
)
from open_context_eval.report import (
    StrategySummary,
    format_scenarios,
    format_table,
    pareto_points,
    summarize,
)
from open_context_eval.results import (
    MIRRORED_FINGERPRINT_FIELDS,
    RESULT_FORMAT_VERSION,
    CompletionOutcome,
    EvaluationResult,
    JudgedOutcome,
    ResultWriter,
    RetentionOutcome,
    RunMetadata,
    RunStatus,
    append_results,
    completed_cells,
    read_results,
    write_results,
)
from open_context_eval.runner import (
    EvaluationConfig,
    ExperimentRunner,
    ScenarioMaterializationError,
    build_session,
    original_token_count,
    run_fingerprint,
    run_id_for,
    run_id_from_fingerprint,
    scenario_events,
    session_id_for,
)
from open_context_eval.scenario import (
    DownstreamTask,
    EvaluationScenario,
    FailureMode,
    JudgedQuestion,
    MetricCategory,
    RetentionCheck,
    TaskKind,
)
from open_context_eval.strategies import (
    FULL_CONTEXT,
    HYBRID_V1,
    PHASE_5_BASELINE,
    SIMPLE_SUMMARY_V1,
    EvaluationStrategy,
    FullContextStrategy,
    Phase5BaselineStrategy,
    PreparedContext,
    SimpleSummaryStrategy,
    StrategyContext,
    default_strategies,
)
from open_context_eval.taxonomy import (
    UNMEASURED,
    FailureClass,
    Observation,
    classify,
    classify_result,
    counts_by_class,
    format_taxonomy,
)

__all__ = [
    "BENCHMARK_DATASET_VERSION",
    "BENCHMARK_SCENARIOS",
    "BENCHMARK_STRATEGIES",
    "CONTINUATION_PROMPT_V1",
    "DATASET_VERSION",
    "DEFAULT_BUDGETS",
    "DETERMINISTIC_TEST",
    "FULL_CONTEXT",
    "HYBRID_V1",
    "MIRRORED_FINGERPRINT_FIELDS",
    "PHASE_5_BASELINE",
    "REAL_MODEL",
    "REFERENCE_BUDGET",
    "RESULT_FORMAT_VERSION",
    "SCENARIOS",
    "SIMPLE_SUMMARY_PROMPT_V1",
    "SIMPLE_SUMMARY_V1",
    "UNMEASURED",
    "BenchmarkConfig",
    "BudgetedProvider",
    "Cell",
    "CompletionCriterion",
    "CompletionKind",
    "CompletionOutcome",
    "CompletionStatus",
    "DownstreamTask",
    "EvaluationConfig",
    "EvaluationJudge",
    "EvaluationResult",
    "EvaluationScenario",
    "EvaluationStrategy",
    "ExperimentRunner",
    "FailureClass",
    "FailureMode",
    "FullContextStrategy",
    "InvalidBenchmarkError",
    "JudgeVerdict",
    "JudgedOutcome",
    "JudgedQuestion",
    "KeywordJudge",
    "MetricCategory",
    "Observation",
    "Phase5BaselineStrategy",
    "PreparedContext",
    "RequestBudget",
    "RequestBudgetExhausted",
    "ResultWriter",
    "RetentionCheck",
    "RetentionOutcome",
    "RunMetadata",
    "RunStatus",
    "ScenarioMaterializationError",
    "SimpleSummaryStrategy",
    "StrategyContext",
    "StrategySummary",
    "TaskKind",
    "append_results",
    "build_continuation_request",
    "build_session",
    "by_id",
    "check_comparable",
    "classify",
    "classify_result",
    "completed_cells",
    "counts_by_class",
    "default_strategies",
    "estimated_requests",
    "experiment_configs",
    "failure_modes_table",
    "format_benchmark",
    "format_failures",
    "format_scenarios",
    "format_table",
    "format_taxonomy",
    "original_token_count",
    "pareto_points",
    "read_results",
    "run_benchmark",
    "run_fingerprint",
    "run_id_for",
    "run_id_from_fingerprint",
    "scenario_completed",
    "scenario_events",
    "session_id_for",
    "stream_benchmark",
    "summarize",
    "summarize_benchmark",
    "write_results",
]
