"""A small command line for the harness.

```
python -m open_context_eval list-scenarios
python -m open_context_eval run --out results.jsonl
python -m open_context_eval compare results.jsonl
```

**Deliberately a module runner, not an installed command.** The project has no
CLI conventions yet and an open question about what its command should be
called; claiming a console script here would answer that question by accident,
in the evaluation harness of all places. ``python -m`` needs no entry point and
no name.

``run`` uses the deterministic fakes unless a provider is configured through the
environment, so it works with nothing installed and never silently reaches the
network. A real-model run is a manual act, never a default.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from open_context.llm import (
    AuthenticationError,
    ModelInfo,
    ProviderConfig,
    ProviderUnavailableError,
    Tokenizer,
    create_provider,
    provider_from_env,
)
from open_context.llm.fakes import FakeProvider, WordTokenizer
from open_context_eval.adversarial import ADVERSARIAL_SCENARIOS
from open_context_eval.adversarial import by_id as adversarial_by_id
from open_context_eval.benchmark import (
    DEFAULT_BUDGETS,
    DETERMINISTIC_TEST,
    REAL_MODEL,
    BenchmarkConfig,
    InvalidBenchmarkError,
    check_comparable,
    estimated_requests,
    stream_benchmark,
)
from open_context_eval.benchmark_dataset import BENCHMARK_DATASET_VERSION, BENCHMARK_SCENARIOS
from open_context_eval.benchmark_dataset import by_id as benchmark_by_id
from open_context_eval.benchmark_dataset_v3 import BENCHMARK_SCENARIOS_V3
from open_context_eval.benchmark_dataset_v3 import by_id as benchmark_v3_by_id
from open_context_eval.benchmark_report import (
    failure_modes_table,
    format_benchmark,
    format_failures,
)
from open_context_eval.budget import BudgetedProvider, RequestBudget, RequestBudgetExhausted
from open_context_eval.dataset import DATASET_VERSION, SCENARIOS, by_id
from open_context_eval.judge import LLMJudge
from open_context_eval.report import format_scenarios, format_table
from open_context_eval.results import (
    EvaluationResult,
    ResultWriter,
    completed_cells,
    read_results,
    write_results,
)
from open_context_eval.runner import EvaluationConfig, ExperimentRunner
from open_context_eval.strategies import default_strategies
from open_context_eval.taxonomy import format_taxonomy


def _providers(
    use_real: bool, *, provider_name: str | None = None, model: str | None = None
) -> tuple[object, Tokenizer, str]:
    """The model and tokenizer to run with, and a note about which.

    Deterministic unless ``use_real``. A real-model run that could not build its
    provider fails: falling back to the fakes would produce numbers labelled as
    measurements, which is the one thing this must never do.
    """
    if not use_real:
        return (
            FakeProvider(reply="(fake provider: no model was called)", context_window=100_000),
            WordTokenizer(exact=False),
            "deterministic fakes, offline",
        )

    # Imported here rather than at module scope so the abstraction, and anyone
    # running deterministically, never pulls in an HTTP client.
    import open_context.llm.providers  # noqa: F401  importing registers them

    if provider_name and model:
        built = create_provider(ProviderConfig(provider=provider_name, model=model))
    elif provider_name or model:
        raise ProviderUnavailableError("--provider and --model go together; give both or neither")
    else:
        built = provider_from_env()

    info: ModelInfo = built.model_info()
    return built, WordTokenizer(exact=False), f"{info.provider}/{info.model}"


def command_list(args: argparse.Namespace) -> int:
    dataset = BENCHMARK_SCENARIOS if args.benchmark else SCENARIOS
    version = BENCHMARK_DATASET_VERSION if args.benchmark else DATASET_VERSION
    print(f"dataset {version}, {len(dataset)} scenarios\n")
    for scenario in dataset:
        modes = ", ".join(mode.value for mode in scenario.failure_modes)
        print(f"{scenario.scenario_id:<28}{scenario.task.kind.value:<24}{modes}")
        if args.verbose:
            print(f"    {scenario.description}")
            print(f"    task: {scenario.task.instruction}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    try:
        provider, tokenizer, note = _providers(
            args.real_model, provider_name=args.provider, model=args.model
        )
    except (ProviderUnavailableError, AuthenticationError) as exc:
        print(f"cannot run against a real model: {exc}", file=sys.stderr)
        return 2

    scenarios = [by_id(name) for name in args.scenario] if args.scenario else list(SCENARIOS)
    config = EvaluationConfig(
        provider=provider,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        target_tokens=args.target_tokens,
        strategies=args.strategy or (),
    )

    with tempfile.TemporaryDirectory() as workspace:
        runner = ExperimentRunner(config, root=Path(workspace) / "archive")
        results = runner.run(scenarios)

    print(f"model: {note}")
    print(f"scenarios: {len(scenarios)}  strategies: {len(config.selected())}")
    if args.out:
        write_results(args.out, results)
        print(f"wrote {len(results)} results to {args.out}")
    print()
    print(format_table(results))
    if not args.real_model:
        print(
            "\nThese numbers came from a fake model that returns a fixed string. "
            "They measure the harness, not any strategy."
        )
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    """Run the benchmark matrix and print what actually happened."""
    try:
        provider, tokenizer, note = _providers(
            args.real_model, provider_name=args.provider, model=args.model
        )
    except (ProviderUnavailableError, AuthenticationError) as exc:
        print(f"cannot run a real-model benchmark: {exc}", file=sys.stderr)
        print(
            "Set the provider's credential in the environment and pass --provider and "
            "--model. Nothing falls back to the fakes: a real-model run that quietly "
            "became a deterministic one would produce numbers labelled as measurements.",
            file=sys.stderr,
        )
        return 2

    if args.dataset == "v4":
        catalogue, lookup = ADVERSARIAL_SCENARIOS, adversarial_by_id
    elif args.dataset == "v3":
        catalogue, lookup = BENCHMARK_SCENARIOS_V3, benchmark_v3_by_id
    else:
        catalogue, lookup = BENCHMARK_SCENARIOS, benchmark_by_id
    scenarios = [lookup(name) for name in args.scenario] if args.scenario else catalogue

    # The budget wraps the provider *before* anything else holds a reference to
    # it. A judge built from the raw provider would send its requests around the
    # ceiling, and the ceiling would silently stop meaning what it says.
    budget = RequestBudget(args.max_requests) if args.max_requests else None
    if budget is not None:
        provider = BudgetedProvider(
            provider,  # type: ignore[arg-type]
            budget,
            min_interval_seconds=args.min_interval,
        )

    judge = None
    if args.judge:
        if not args.real_model:
            print("--judge needs --real-model: the fakes cannot grade", file=sys.stderr)
            return 2
        judge = LLMJudge(provider)  # type: ignore[arg-type]

    try:
        config = BenchmarkConfig(
            provider=provider,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            evaluation_mode=REAL_MODEL if args.real_model else DETERMINISTIC_TEST,
            budgets=args.budget or DEFAULT_BUDGETS,
            repetitions=args.repetitions,
            scenarios=list(scenarios),
            judge=judge,
        )
    except InvalidBenchmarkError as exc:
        print(f"invalid benchmark configuration: {exc}", file=sys.stderr)
        return 2

    print(f"dataset {config.dataset_version}   model {note}")
    if judge is not None:
        print(f"judge {judge.name} — judged questions cost one request each")
    print(config.describe())
    print(f"at least {estimated_requests(config)} model requests")

    done: set[tuple[str, str, str]] = set()
    if args.resume:
        if not args.out:
            print("--resume needs --out: there is nothing to resume from", file=sys.stderr)
            return 2
        done = completed_cells(args.out)
        print(f"resuming: {len(done)} cells already recorded in {args.out}")

    if budget is not None:
        print(f"request budget: {budget.limit}, min interval {args.min_interval}s")
    print()

    exhausted = False
    results: list[EvaluationResult] = []
    writer = ResultWriter(args.out) if args.out else None
    try:
        with tempfile.TemporaryDirectory() as workspace:
            skip = (
                (lambda run, scenario, strategy: (run, scenario, strategy) in done)
                if done
                else None
            )
            try:
                for result in stream_benchmark(config, root=Path(workspace), skip=skip):
                    results.append(result)
                    if writer is not None:
                        writer.write(result)
            except RequestBudgetExhausted as stop:
                exhausted = True
                print(f"stopped: {stop}")
    finally:
        if writer is not None:
            writer.close()

    if args.out:
        print(f"wrote {len(results)} results to {args.out}")
        results = list(read_results(args.out))
        print(f"{len(results)} results in the file in total\n")

    if not results:
        print("no results", file=sys.stderr)
        return 1

    print(format_benchmark(results))

    problems = check_comparable(results)
    if problems:
        print("\nTHIS IS NOT A VALID ALGORITHM COMPARISON:")
        for problem in problems:
            print(f"  {problem}")

    print()
    print(format_taxonomy(results, strategies=list(config.strategies)))

    if exhausted:
        print(
            "\nThe matrix is incomplete: the request budget ran out before every cell ran. "
            "Cells that were never attempted are absent rather than recorded as failures. "
            "Re-run with --resume and the same --out to continue."
        )

    if args.verbose:
        print("\nFailures by scenario:")
        print(format_failures(results))
        print()
        print(failure_modes_table(results))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    """Report on a results file that already exists.

    Sends no request, so re-reading a finished run costs nothing — which matters
    when the run cost a day's quota to produce and the reporting code has since
    been improved.
    """
    results = list(read_results(args.results))
    if not results:
        print("no results in that file", file=sys.stderr)
        return 1
    run = results[0].run
    print(f"run {run.run_id}  dataset {run.dataset_version}  model {run.provider}/{run.model}")
    print(f"prompt {run.continuation_prompt_id} ({run.continuation_prompt_hash})")
    print()

    if args.benchmark:
        print(format_benchmark(results))
        print()
        print(
            format_taxonomy(results, strategies=sorted({r.strategy for r in results if r.strategy}))
        )
    else:
        print(format_table(results))

    problems = check_comparable(results)
    if problems:
        print("\nTHIS IS NOT A VALID ALGORITHM COMPARISON:")
        for problem in problems:
            print(f"  {problem}")

    if args.verbose:
        print()
        print(format_scenarios(results))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open_context_eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list-scenarios", help="show the synthetic dataset")
    listing.add_argument("-v", "--verbose", action="store_true")
    listing.add_argument(
        "--benchmark", action="store_true", help="show the benchmark dataset instead of v1"
    )
    listing.set_defaults(handler=command_list)

    running = sub.add_parser("run", help="run scenarios against strategies")
    running.add_argument("--scenario", action="append", help="scenario id; repeatable")
    running.add_argument(
        "--strategy", action="append", choices=sorted(default_strategies()), help="repeatable"
    )
    running.add_argument("--target-tokens", type=int, default=2_000)
    running.add_argument("--out", help="write results as JSON Lines")
    running.add_argument(
        "--real-model",
        action="store_true",
        help="use the provider configured in the environment instead of the fakes",
    )
    running.add_argument("--provider", help="registered provider name, such as groq")
    running.add_argument("--model", help="model identifier, as the provider names it")
    running.set_defaults(handler=command_run)

    benchmarking = sub.add_parser("benchmark", help="run the benchmark matrix")
    benchmarking.add_argument("--scenario", action="append", help="scenario id; repeatable")
    benchmarking.add_argument(
        "--budget", action="append", type=int, help="target token budget; repeatable"
    )
    benchmarking.add_argument("--repetitions", type=int, default=1)
    benchmarking.add_argument("--out", help="write results as JSON Lines")
    benchmarking.add_argument(
        "--real-model",
        action="store_true",
        help="use the provider configured in the environment instead of the fakes",
    )
    benchmarking.add_argument("--provider", help="registered provider name, such as groq")
    benchmarking.add_argument("--model", help="model identifier, as the provider names it")
    benchmarking.add_argument(
        "--max-requests",
        type=int,
        help=(
            "stop after this many model requests. Cells not reached are absent from the "
            "results rather than recorded as failures"
        ),
    )
    benchmarking.add_argument(
        "--resume",
        action="store_true",
        help="skip cells already recorded in --out, and append to it",
    )
    benchmarking.add_argument(
        "--min-interval",
        type=float,
        default=0.0,
        help=(
            "seconds to leave between requests, to stay under a per-minute rate limit. "
            "Applies only with --max-requests"
        ),
    )
    benchmarking.add_argument(
        "--dataset",
        choices=("v2", "v3", "v4"),
        default="v2",
        help=(
            "which benchmark dataset to run. v3 asserts presence deterministically and "
            "absence through judged questions; v4 is the adversarial set, whose "
            "conversations are long enough that the budget forces a real choice"
        ),
    )
    benchmarking.add_argument(
        "--judge",
        action="store_true",
        help=(
            "grade judged questions with the same model, through the same request budget. "
            "Roughly doubles the request cost of a run"
        ),
    )
    benchmarking.add_argument("-v", "--verbose", action="store_true", help="show failure detail")
    benchmarking.set_defaults(handler=command_benchmark)

    comparing = sub.add_parser("compare", help="summarise a results file")
    comparing.add_argument("results")
    comparing.add_argument(
        "--benchmark",
        action="store_true",
        help="report as a benchmark: per-arm table and failure taxonomy",
    )
    comparing.add_argument("-v", "--verbose", action="store_true")
    comparing.set_defaults(handler=command_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
