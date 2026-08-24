"""A hard ceiling on how many requests a benchmark may send.

```
matrix -> budget -> provider -> service
                 -> exhausted -> stop, having spent exactly the ceiling
```

**The reason is arithmetic, not caution.** The first real benchmark's full
matrix is 105 runs and roughly 195 requests. OpenRouter's free tier allows 50
requests a day to a `:free` model. So the matrix cannot be run in a day, and
what happens at request 51 decides whether the run is a measurement or a
wreck.

Without a ceiling the answer is: every remaining cell calls the service, gets a
429, and is recorded as ``MODEL_ERROR``. The results file then holds real
measurements for the first fifty cells and rate-limit failures for the rest,
mixed together and distinguishable only by reading error strings. Worse, the
failures look like findings — a strategy that "failed" on two thirds of the
dataset — when nothing was measured about it at all.

With a ceiling the run stops. Cells that were never attempted are simply
absent, the results file holds only cells that actually ran, and resuming later
adds the rest. Absent and failed are different claims, and only one of them is
true of a cell nobody asked about.

**This is not a retry policy.** Nothing here catches a 429, backs off, or tries
again. A 429 that arrives is a genuine provider failure and is recorded as one:
the budget's job is to stop us spending requests we do not have, not to disguise
the provider's answer when we do.

It does pace, which is a different thing. The same free tier that allows 50
requests a day allows 20 a minute, and a 45-cell matrix sent as fast as the
loop can send it would spend a third of its daily quota discovering that. Pacing
is a property of the run, not of the provider — which is why it lives here,
beside the count of how many requests a run may send, and not behind the
provider boundary where Phase 4.5 deliberately left retry policy out.

**Exhaustion is not an ``LLMError``.** The runner turns any ``LLMError`` into a
``MODEL_ERROR`` result, which is exactly the outcome this exists to prevent, so
``RequestBudgetExhausted`` deliberately sits outside that hierarchy and
propagates past the runner to whoever owns the budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from open_context.llm import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ModelInfo,
    StructuredResult,
)


class RequestBudgetExhausted(Exception):
    """The run asked for one more request than it was allowed.

    **Not an ``LLMError``, on purpose.** The provider was never called, so
    nothing about the model failed. Recording this as a model error would put a
    row in the results file asserting that a strategy was tried and did not
    work, when it was not tried.
    """

    def __init__(self, budget: RequestBudget) -> None:
        super().__init__(
            f"request budget exhausted: {budget.spent} of {budget.limit} requests spent. "
            "The remaining cells were not attempted and are absent from the results "
            "rather than recorded as failures."
        )
        self.spent = budget.spent
        self.limit = budget.limit


@dataclass
class RequestBudget:
    """How many requests remain.

    Mutable and shared: one budget object is handed to one provider wrapper, and
    the count it keeps is the run's, not any single strategy's.
    """

    limit: int
    spent: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError(f"a request budget cannot be negative, got {self.limit}")
        if self.spent < 0:
            raise ValueError(f"spend cannot be negative, got {self.spent}")

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit

    def spend(self) -> None:
        """Charge one request, or refuse.

        **Charged before the call, not after.** A request that was sent and then
        failed still consumed quota — the provider counts attempts, not
        successes — so charging on the way out would let a run of failures
        overspend a budget that reads as untouched.
        """
        if self.exhausted:
            raise RequestBudgetExhausted(self)
        self.spent += 1

    def describe(self) -> str:
        return f"{self.spent}/{self.limit} requests spent, {self.remaining} remaining"


class BudgetedProvider:
    """An ``LLMProvider`` that stops when the budget does, and paces itself.

    Delegates everything. ``model_info`` is neither charged nor paced: both
    shipped providers answer it from configuration without a request, and
    charging a local attribute read against a network quota would make the count
    wrong in the direction that matters.

    ``min_interval_seconds`` is the smallest gap between two requests. Zero, the
    default, paces nothing, so an offline run against the fakes is not slowed by
    machinery meant for a metered service.
    """

    def __init__(
        self,
        provider: LLMProvider,
        budget: RequestBudget,
        *,
        min_interval_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError(f"an interval cannot be negative, got {min_interval_seconds}")
        self.provider = provider
        self.budget = budget
        self.min_interval_seconds = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None

    def model_info(self) -> ModelInfo:
        return self.provider.model_info()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.budget.spend()
        self._pace()
        return self.provider.generate(request)

    def structured_output(
        self, request: GenerationRequest, schema: Mapping[str, Any]
    ) -> StructuredResult:
        self.budget.spend()
        self._pace()
        return self.provider.structured_output(request, schema)

    def _pace(self) -> None:
        """Wait long enough that this request does not break the per-minute rate.

        Paced after the budget is charged, so an exhausted budget refuses
        without first sleeping for a request it is about to decline.

        Measured from the start of the previous request rather than its end: the
        provider's limit counts requests per minute, and a slow call has already
        supplied part of the gap.
        """
        if self.min_interval_seconds <= 0:
            self._last_request = self._clock()
            return
        now = self._clock()
        if self._last_request is not None:
            waited = now - self._last_request
            if waited < self.min_interval_seconds:
                self._sleep(self.min_interval_seconds - waited)
                now = self._clock()
        self._last_request = now

    def __repr__(self) -> str:
        return f"BudgetedProvider({self.provider!r}, {self.budget.describe()})"


__all__ = [
    "BudgetedProvider",
    "RequestBudget",
    "RequestBudgetExhausted",
]
