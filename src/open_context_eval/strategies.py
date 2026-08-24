"""The arms being compared.

```
conversation -> EvaluationStrategy -> PreparedContext -> the same downstream task
```

A strategy turns a session into the context a continuing agent will receive.
That is the only thing that varies between arms: the task, the prompt, the
model, and the tokenizer are held identical, so a difference in outcome is
attributable to the representation rather than to the harness.

**A strategy may fail to produce a context, and that is not a score of zero.**
``PreparedContext.available`` is false when the full conversation does not fit
the receiving model, and the run is recorded as a context error. Silently
truncating would turn the reference condition into a different, unnamed
strategy and make every comparison against it meaningless.

**Phase 5 is used through its public interface and is not modified.** The
baseline arm constructs ``BaselineRecentPlusSummary`` and calls ``compact``.
Nothing here reaches into it, and nothing here reimplements it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from open_context.archive import SessionLog
from open_context.compaction import (
    BaselineConfig,
    BaselineRecentPlusSummary,
    CompactionRequest,
    render_events,
    stream_archive,
)
from open_context.hybrid import HybridCompaction, HybridConfig
from open_context.llm import LLMProvider, TokenCount, Tokenizer

from open_context.compaction import Prompt  # isort: skip
from open_context_eval.prompts import SIMPLE_SUMMARY_PROMPT_V1, build_summary_request

FULL_CONTEXT = "full_context"
PHASE_5_BASELINE = "phase_5_baseline"

SIMPLE_SUMMARY_V1 = "simple_summary_v1"
"""The plain-summarization arm, versioned in its own name.

Alone among the arms, because it is the one most likely to be replaced by a
different idea of what "a plain summary" means. When a v2 exists, a result file
mentioning ``simple_summary_v1`` still says exactly which baseline produced it,
without a reader having to cross-reference a version field."""

HYBRID_V1 = "hybrid_v1"
"""Phase 8: structured state plus a recent window plus a historical summary."""


@dataclass(frozen=True)
class PreparedContext:
    """What one strategy hands to the downstream agent."""

    text: str
    tokens: TokenCount
    available: bool = True
    unavailable_reason: str = ""
    llm_calls: int = 0

    input_tokens: int | None = None
    output_tokens: int | None = None
    """What producing this context cost, as the provider reported it.

    Recorded because a strategy that yields a small context from a great deal of
    model work is not cheap, and comparing strategies on context size alone
    would say it was. ``None`` when the provider reported nothing, never zero.
    """

    latency_seconds: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    """Strategy-specific numbers worth keeping, such as a compaction report."""


@dataclass(frozen=True)
class StrategyContext:
    """Everything a strategy is allowed to use.

    Notably not the scenario: a strategy sees the conversation and the budget,
    never the expected answers, so it cannot optimise for the checks.
    """

    log: SessionLog
    provider: LLMProvider
    tokenizer: Tokenizer
    target_tokens: int


@runtime_checkable
class EvaluationStrategy(Protocol):
    """Produces the context representation for one arm."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> int: ...

    @property
    def budget_independent(self) -> bool:
        """Whether the target budget changes what this strategy produces.

        A reference condition ignores the budget, so running it once per budget
        would be three independent samples of one condition — which averaging
        then hides. The benchmark runs such an arm once per scenario and
        repetition instead. Declared on the strategy because it is a fact about
        the strategy, not a list the orchestrator has to keep in step.
        """
        ...

    @property
    def requests_per_run(self) -> int:
        """Model requests this strategy sends to build one context, at minimum.

        Excludes the continuation call, which every arm pays equally and which
        the orchestrator counts itself.

        **A lower bound.** A strategy that chunks a long history or retries a
        summary that overshot sends more, and cannot know how many in advance.
        It is declared here for the same reason as ``budget_independent``: it is
        a fact about the strategy, and a run sizing itself against a metered
        quota needs it before the first request rather than after the last.
        """
        ...

    def prepare(self, context: StrategyContext) -> PreparedContext: ...

    def configuration(self) -> dict[str, Any]:
        """Settings that change what this strategy produces.

        Folded into the run id, so an experiment rerun with a differently
        configured strategy cannot be filed under the same identity.
        """
        ...


class FullContextStrategy:
    """The reference condition: the original conversation, unmodified.

    Every other arm is measured against this, which is why it must never quietly
    become something else. When the conversation does not fit the receiving
    model's context window it reports itself unavailable rather than truncating:
    a truncated full-context arm is a different strategy wearing the reference
    condition's name, and comparisons against it would flatter everything else.
    """

    name = FULL_CONTEXT
    version = 1
    budget_independent = True
    """It hands over the whole conversation; no target changes that."""

    requests_per_run = 0
    """It calls no model. Rendering the archive is all it does."""

    def configuration(self) -> dict[str, Any]:
        """Nothing to configure. It hands over the conversation as it stands."""
        return {}

    def prepare(self, context: StrategyContext) -> PreparedContext:
        text = render_events(tuple(stream_archive(context.log)))
        tokens = context.tokenizer.count_text(text)
        window = context.provider.model_info().context_window

        if window is not None and tokens.count > window:
            return PreparedContext(
                text="",
                tokens=tokens,
                available=False,
                unavailable_reason=(
                    f"the conversation is {tokens.count} tokens and the model's context "
                    f"window is {window}; the full-context condition cannot be run and "
                    "was not truncated to fake it"
                ),
            )
        return PreparedContext(text=text, tokens=tokens, latency_seconds=0.0)


class SimpleSummaryStrategy:
    """The plain-summarization baseline. Exactly three steps, and no more.

    ```
    FULL CONVERSATION -> ONE LLM SUMMARY -> DETERMINISTIC OUTPUT-BUDGET ENFORCEMENT
    ```

    **This is an experimental comparison arm, not a compaction engine.** It lives
    in the evaluation harness because that is what it is for: representing what a
    developer would get from asking a model to summarize the session, so that
    everything else has something ordinary to be measured against. It is not
    exported by the runtime, is not offered as a strategy anyone should use, and
    should never grow features. If it starts to look good, that is a finding
    about the problem, not an invitation to promote it.

    **It shares no compaction logic with Phase 5**, deliberately. No recent
    window is preserved, so everything the agent receives has been through the
    model — which is exactly what makes the value of keeping recent turns
    verbatim measurable rather than assumed. Borrowing the baseline's budget
    machinery would mean this arm was partly measuring that machinery.

    Budget enforcement is therefore its own: count the summary, and cut it at a
    word boundary if the model overshot. No retry, no fallback ladder, no
    negotiation with the model.
    """

    name = SIMPLE_SUMMARY_V1
    version = 1
    budget_independent = False

    requests_per_run = 1
    """One summarization call, and exactly one. It has no retry ladder."""

    def __init__(self, prompt: Prompt = SIMPLE_SUMMARY_PROMPT_V1) -> None:
        self.prompt = prompt

    def configuration(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt.identifier, "prompt_hash": self.prompt.content_hash}

    def prepare(self, context: StrategyContext) -> PreparedContext:
        started = time.perf_counter()
        conversation = render_events(tuple(stream_archive(context.log)))
        request = build_summary_request(
            conversation, prompt=self.prompt, max_output_tokens=context.target_tokens
        )
        result = context.provider.generate(request)
        text = result.text.strip()
        tokens = context.tokenizer.count_text(text)

        truncated = False
        if tokens.count > context.target_tokens:
            text = _cut_to_budget(text, context.target_tokens, context.tokenizer)
            tokens = context.tokenizer.count_text(text)
            truncated = True

        usage = result.usage
        return PreparedContext(
            text=text,
            tokens=tokens,
            llm_calls=1,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            latency_seconds=time.perf_counter() - started,
            detail={"prompt_id": self.prompt.identifier, "truncated": truncated},
        )


class Phase5BaselineStrategy:
    """The Phase 5 compactor, called through its public interface.

    Nothing here modifies or reaches inside ``BaselineRecentPlusSummary``. The
    compaction report travels into the result's detail so a benchmark can see
    which fallbacks fired without the harness having to infer them.
    """

    name = PHASE_5_BASELINE
    budget_independent = False

    requests_per_run = 1
    """One summarization call in the ordinary case, and a floor rather than a count.

    The compactor chunks history that exceeds the model's context window and
    retries a summary that overshot its budget, so a run can cost more. It never
    costs less: a conversation already inside the target is returned whole with
    no model call at all, which is zero — but sizing a quota against the cheap
    path is how a run discovers it was wrong at request fifty-one.
    """

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.version = BaselineRecentPlusSummary.version

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)

    def prepare(self, context: StrategyContext) -> PreparedContext:
        strategy = BaselineRecentPlusSummary(context.provider, context.tokenizer)
        result = strategy.compact(
            CompactionRequest(
                log=context.log, target_tokens=context.target_tokens, config=self.config
            )
        )
        text = result.rendered()
        return PreparedContext(
            text=text,
            tokens=context.tokenizer.count_text(text),
            llm_calls=result.model.llm_calls,
            input_tokens=result.model.input_tokens_reported,
            output_tokens=result.model.output_tokens_reported,
            latency_seconds=result.elapsed_seconds,
            detail={
                "compaction_needed": result.compaction_needed,
                "fallbacks": list(result.fallbacks),
                "warnings": list(result.warnings),
                "recent_event_count": result.recent_event_count,
                "historical_event_count": result.historical_event_count,
                "dropped_recent_event_count": result.dropped_recent_event_count,
                "prompt_id": result.prompt_id,
                "prompt_hash": result.prompt_hash,
            },
        )


class HybridStrategy:
    """The Phase 8 hybrid compactor, called through its public interface.

    Nothing here modifies or reaches inside `HybridCompaction`. It is registered
    as a fourth arm so the claim that structured state preserves exact values is
    measured against the same scenarios, model, and prompt as everything else,
    rather than demonstrated on the two cases that motivated it.
    """

    name = HYBRID_V1
    budget_independent = False

    requests_per_run = 2
    """An extraction call and a summarization call, and a floor rather than a count.

    It is deliberately the most expensive arm. Phase 5.8 already found compaction
    spending more tokens than sending the whole conversation at these lengths;
    this spends more still, which is why the exit criterion is a measured
    improvement rather than a plausible one.
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self.version = HybridCompaction.version

    def configuration(self) -> dict[str, Any]:
        return {"state_fraction": self.config.state_fraction}

    def prepare(self, context: StrategyContext) -> PreparedContext:
        strategy = HybridCompaction(context.provider, context.tokenizer, self.config)
        result = strategy.compact(
            CompactionRequest(
                log=context.log,
                target_tokens=context.target_tokens,
                config=self.config.baseline,
            )
        )
        text = result.rendered()
        return PreparedContext(
            text=text,
            tokens=context.tokenizer.count_text(text),
            llm_calls=result.model.llm_calls,
            input_tokens=result.model.input_tokens_reported,
            output_tokens=result.model.output_tokens_reported,
            latency_seconds=result.elapsed_seconds,
            detail={
                "state_items_extracted": result.configuration.get("state_items_extracted"),
                "state_items_kept": result.configuration.get("state_items_kept"),
                "state_tokens": result.state_tokens.count if result.state_tokens else 0,
                "warnings": list(result.warnings),
                "recent_event_count": result.recent_event_count,
            },
        )


def _cut_to_budget(text: str, budget: int, tokenizer: Tokenizer) -> str:
    """Deterministic word-boundary cut. Binary search, bounded."""
    if budget <= 0:
        return ""
    words = text.split()
    low, high, best = 0, len(words), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = " ".join(words[:middle])
        if tokenizer.count_text(candidate).count <= budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def default_strategies() -> dict[str, EvaluationStrategy]:
    """The arms available today.

    Retrieval and MemHandoff are absent because neither exists. A named arm with
    no implementation behind it would appear in reports as a strategy that
    scored nothing, which is not the same as one that was never run.
    """
    return {
        FULL_CONTEXT: FullContextStrategy(),
        SIMPLE_SUMMARY_V1: SimpleSummaryStrategy(),
        PHASE_5_BASELINE: Phase5BaselineStrategy(),
        HYBRID_V1: HybridStrategy(),
    }


__all__ = [
    "FULL_CONTEXT",
    "PHASE_5_BASELINE",
    "SIMPLE_SUMMARY_V1",
    "EvaluationStrategy",
    "FullContextStrategy",
    "Phase5BaselineStrategy",
    "PreparedContext",
    "SimpleSummaryStrategy",
    "StrategyContext",
    "default_strategies",
]
