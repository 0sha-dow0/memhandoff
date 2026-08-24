"""Hybrid compaction: structured state, plus recent verbatim, plus history.

**It lives outside ``compaction`` because it is not a peer of the baseline.**
Putting it beside `BaselineRecentPlusSummary` created an import cycle —
`compaction` would import `extraction`, which already imports `compaction` for
its prompt and rendering types — and the cycle was the design telling the truth:
`compaction` and `extraction` are peers, and this composes both. A package above
them depends on both and is depended on by neither.

```
RAW ARCHIVE
     |
 +---+--------------+
 |   |              |
STATE RECENT     HISTORY
 |   |              |
 +---+--------------+
     |
BUDGET ALLOCATION
     |
PORTABLE CONTEXT
```

**It exists to fix one measured failure, not to be more sophisticated.** The
Phase 5.8 benchmark found the baseline and a plain one-call summary scoring
identically, and found that what compaction actually lost was **exact values** —
a port number among several similar ports, a row count reported by a tool. Full
context lost neither. Structured state is the mechanism that can carry an exact
value through compaction intact, because an extracted `Fact` keeps the number as
written instead of hoping a summarizer chose to repeat it.

So the bet this phase makes is narrow and falsifiable: *state preserves exact
values that prose drops*. If the adversarial benchmark says otherwise, the
answer is to delete this, not to add another layer to it.

## What it costs

An extraction call **on top of** a summarization call. Phase 5.8 already found
compaction spending more tokens than sending the whole conversation at these
lengths; this spends more still. That is stated on every result rather than
discovered later, and it is the reason the exit criterion is a measured
improvement rather than a plausible one.

## Priority is a fixed order, not a score

High-value state goes in first and is never traded away for prose: goals,
constraints, decisions, unresolved tasks. Everything else competes for what is
left. An importance model that learned or guessed a ranking would need its own
evaluation before anything built on it could be trusted, and there is no
evidence yet that would justify one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from open_context.archive import SessionLog
from open_context.compaction.baseline import BaselineRecentPlusSummary
from open_context.compaction.budget import BaselineConfig
from open_context.compaction.result import CompactionResult, TokenCountInfo
from open_context.compaction.strategy import CompactionRequest
from open_context.extraction.extractor import ExtractionConfig, StructuredExtractor
from open_context.llm import LLMProvider, Tokenizer
from open_context.models.enums import StateStatus, StateType
from open_context.models.state import StateItem

STATE_ORDER: tuple[StateType, ...] = (
    StateType.GOAL,
    StateType.CONSTRAINT,
    StateType.FACT,
    StateType.DECISION,
    StateType.TASK,
    StateType.OPEN_QUESTION,
    StateType.ENTITY,
)
"""The order state is rendered and, when the budget bites, kept.

Goals and constraints first, because a continuation that violates a constraint
is wrong regardless of what else it got right.

**Facts third, and the first draft had them second from last.** The reasoning
then was that a fact without the decision it supports is a number with no reason
to be trusted. That is a fine sentence and it was the wrong call: facts are
where exact values live, and exact values are the one failure Phase 5.8 actually
measured. On the `confusable-numbers` scenario the first ordering extracted the
fact carrying the port numbers and then dropped it to the budget, keeping a goal,
a decision about formatting, and a task — losing precisely the thing this phase
was built to preserve.

The lesson generalises past this list: a priority order argued from plausibility
will quietly contradict the measurement that motivated the work.
"""

_HEADINGS: dict[StateType, str] = {
    StateType.GOAL: "GOALS",
    StateType.CONSTRAINT: "CONSTRAINTS",
    StateType.DECISION: "DECISIONS",
    StateType.TASK: "OPEN WORK",
    StateType.FACT: "FACTS",
    StateType.OPEN_QUESTION: "OPEN QUESTIONS",
    StateType.ENTITY: "ENTITIES",
}


@dataclass(frozen=True)
class HybridConfig:
    """How the budget is split, and what the summary half does.

    ``state_fraction`` is the share reserved for structured state *before*
    anything else is allocated. The remainder goes to the baseline's own split
    between recent verbatim events and historical summary, so this phase does
    not reimplement a decision Phase 5 already made and measured.
    """

    state_fraction: float = 0.35
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.state_fraction < 1.0:
            raise ValueError(
                f"state_fraction must leave room for both halves, got {self.state_fraction}"
            )


def render_state(items: list[StateItem]) -> str:
    """Structured state as text a model can read.

    Grouped by type under headings, in `STATE_ORDER`. Superseded items are
    omitted: this renders the state as it now stands, and a reader given both
    sides of a reversal without being told which won would be worse off than one
    given neither.

    A decision carries its rationale inline, because a decision whose reason was
    dropped is the thing people reopen a long session to recover.
    """
    active = [item for item in items if item.status is StateStatus.ACTIVE]
    if not active:
        return ""

    blocks = []
    for state_type in STATE_ORDER:
        group = [item for item in active if item.type is state_type]
        if not group:
            continue
        lines = [_HEADINGS[state_type]]
        for item in group:
            rationale = getattr(item, "rationale", "")
            suffix = f" (because: {rationale})" if rationale else ""
            lines.append(f"- {item.content}{suffix}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def fit_state(items: list[StateItem], tokenizer: Tokenizer, budget: int) -> tuple[str, int]:
    """The largest prefix of state that fits, and how many items it holds.

    Drops from the **end** of `STATE_ORDER` first, so a budget too small for
    everything loses entities and open questions before it loses constraints.
    Truncating the rendered text instead would cut mid-sentence and could leave
    a constraint reading as its own opposite.
    """
    active = [item for item in items if item.status is StateStatus.ACTIVE]
    ordered = sorted(active, key=lambda item: STATE_ORDER.index(item.type))

    kept = list(ordered)
    while kept:
        text = render_state(kept)
        if tokenizer.count_text(text).count <= budget:
            return text, len(kept)
        kept.pop()
    return "", 0


class HybridCompaction:
    """Structured state, a verbatim recent window, and a historical summary.

    Composes the Phase 5 baseline rather than reimplementing it: the recent
    window and the summary come from `BaselineRecentPlusSummary` called through
    its public interface, so the only thing being measured here is what adding
    state does.
    """

    name = "hybrid_v1"
    version = 1

    def __init__(
        self,
        provider: LLMProvider,
        tokenizer: Tokenizer,
        config: HybridConfig | None = None,
    ) -> None:
        self.provider = provider
        self.tokenizer = tokenizer
        self.config = config or HybridConfig()
        self._extractor = StructuredExtractor(provider, self.config.extraction)
        self._baseline = BaselineRecentPlusSummary(provider, tokenizer)

    def compact(self, request: CompactionRequest) -> CompactionResult:
        """Extract state, then compact the rest into what is left of the budget."""
        # The state share must not starve the rest below the baseline's own
        # floor. Without this cap a small target raises `TargetTooSmallError`
        # from inside the baseline, which reads as the target being too small
        # when in fact the split was.
        floor = self.config.baseline.minimum_target_tokens
        wanted = max(0, int(request.target_tokens * self.config.state_fraction))
        state_budget = max(0, min(wanted, request.target_tokens - floor))

        extraction = self._extractor.extract(request.log)
        items = [entry.item for entry in extraction.items]
        state_text, kept = fit_state(items, self.tokenizer, state_budget)

        state_count = self.tokenizer.count_text(state_text) if state_text else None
        spent = state_count.count if state_count else 0

        inner = self._baseline.compact(
            CompactionRequest(
                log=request.log,
                target_tokens=max(floor, request.target_tokens - spent),
                config=self.config.baseline,
            )
        )

        warnings = list(inner.warnings)
        if extraction.report.malformed_response:
            warnings.append(
                "extraction produced no usable state; this result is the Phase 5 baseline "
                "with an empty state section, and should not be read as hybrid compaction"
            )
        elif kept < len([i for i in items if i.status is StateStatus.ACTIVE]):
            warnings.append(
                f"state budget held {kept} of "
                f"{len([i for i in items if i.status is StateStatus.ACTIVE])} items; "
                "the rest were dropped from the end of the priority order"
            )

        return inner.model_copy(
            update={
                "strategy": self.name,
                "strategy_version": self.version,
                "state_section": state_text,
                "allocated_state_tokens": state_budget,
                "state_tokens": (
                    TokenCountInfo(
                        count=state_count.count,
                        exact=state_count.exact,
                        method=state_count.method,
                    )
                    if state_count
                    else None
                ),
                "target_tokens": request.target_tokens,
                # The extraction call is added to the baseline's own count. A
                # hybrid result reporting only the summarization call would
                # understate its bill by exactly the thing this phase added.
                "model": inner.model.model_copy(
                    update={"llm_calls": inner.model.llm_calls + extraction.report.llm_calls}
                ),
                "warnings": tuple(warnings),
                "configuration": {
                    "state_fraction": self.config.state_fraction,
                    "extraction_prompt_id": extraction.prompt_id,
                    "extraction_prompt_hash": extraction.prompt_hash,
                    "state_items_extracted": len(items),
                    "state_items_kept": kept,
                    **inner.configuration,
                },
            }
        )


def state_of(log: SessionLog, provider: LLMProvider) -> list[StateItem]:
    """Convenience for callers that want the state without the compaction."""
    return [entry.item for entry in StructuredExtractor(provider).extract(log).items]


__all__ = [
    "STATE_ORDER",
    "HybridCompaction",
    "HybridConfig",
    "fit_state",
    "render_state",
    "state_of",
]
