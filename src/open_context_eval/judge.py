"""The judge seam, and an implementation behind it.

Some things a scenario cares about cannot be settled by substring matching:
whether a rationale was understood rather than repeated, whether an answer is
semantically right, whether anything was invented. Those are judged questions.

**No judge is required, and none is wired in by default.** A question with no
judge is reported as not evaluated — never as passed, never as failed. A harness
that scored unanswerable questions as failures would penalise strategies for the
evaluator's limitations, and one that scored them as passes would be worse.

Dataset `v3` made this load-bearing rather than optional. Every assertion about
something being *absent* moved here, because absence is not decidable by
substring: the same term appears in an endorsement and in a rejection. So a `v3`
run without a judge leaves real questions unanswered, and two of its scenarios
have no deterministic evidence at all.

## Three problems an LLM judge brings

**It is nondeterministic.** Two runs of one experiment can disagree. Temperature
is pinned to zero, which reduces it and does not remove it, and the judge's model
identity goes into the run fingerprint so two runs judged differently are never
filed under one id.

**It can be the model under test.** Nothing here prevents that, and a model
grading its own continuation is a known bias. `LLMJudge.name` carries the
provider and model so a reader can see when it happened; refusing the
configuration outright would be worse, since on a free tier there may be only one
model available.

**It costs a request per question.** `v3` has 21 judged questions across 15
scenarios, so judging roughly doubles the request cost of a run. Pass the same
budgeted provider used for the run and the judge's requests are counted against
the same ceiling; pass an unbudgeted one and the ceiling silently stops meaning
anything.

## Why the prompt lives here

`prompts.py` is the leakage boundary for *downstream* requests and guarantees
that an expected answer cannot reach the model under test. A judge prompt is the
opposite kind of thing: it is supposed to carry what the evaluator knows. Keeping
it out of that module keeps the guarantee there easy to state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from open_context.compaction import Prompt
from open_context.llm import ChatMessage, GenerationRequest, LLMError, LLMProvider
from open_context.llm.free_models import REASONING_FLOOR, find_free_model


@dataclass(frozen=True)
class JudgeVerdict:
    """One judged answer, or an explicit refusal to answer.

    ``evaluated`` is false when the judge could not reach a verdict — the model
    errored, or answered in a shape the parser could not read. **That is not a
    failure**, and must not be recorded as one: a judge having a bad afternoon
    would otherwise read as a strategy that lost a constraint.
    """

    passed: bool
    reasoning: str = ""
    evaluated: bool = True

    @classmethod
    def undecided(cls, reasoning: str) -> JudgeVerdict:
        return cls(passed=False, reasoning=reasoning, evaluated=False)


@runtime_checkable
class EvaluationJudge(Protocol):
    """Answers a question about a continuation that no exact match can."""

    @property
    def name(self) -> str: ...

    def judge(self, response: str, question: str) -> JudgeVerdict: ...


@dataclass(frozen=True)
class KeywordJudge:
    """A deterministic stand-in, for tests and for offline runs.

    Answers by looking for phrases the caller supplies per question. It is not a
    judge in any meaningful sense and makes no pretence of being one; it exists
    so the judged path is exercised, and so a scenario's judged questions are not
    silently unreachable code.
    """

    answers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    name: str = "keyword"

    def judge(self, response: str, question: str) -> JudgeVerdict:
        needles = self.answers.get(question)
        if needles is None:
            return JudgeVerdict.undecided("no rule for this question")
        lowered = response.lower()
        hits = [needle for needle in needles if needle.lower() in lowered]
        return JudgeVerdict(
            passed=bool(hits),
            reasoning=f"matched {hits}" if hits else f"none of {list(needles)} present",
        )


JUDGE_PROMPT_V1 = Prompt(
    name="judge",
    version=1,
    text="""\
You are grading one answer produced by an AI agent. You are given a question \
about that answer, and the answer itself. Decide the question.

Answer with exactly one word on the first line: YES or NO. On the second line \
give one short sentence of reason. Write nothing else.

YES means the answer satisfies what the question asks. NO means it does not.

Judge only what the answer actually says. Do not reward an answer for sounding \
confident, and do not penalise it for being brief. If the question asks whether \
something was avoided, an answer that names the thing purely in order to reject \
or exclude it has avoided it, and the correct verdict is YES.

If the answer is empty or does not address the question at all, reply NO.""",
)
"""Versioned, and hashed into the run fingerprint like every other prompt.

The paragraph about naming a thing in order to reject it is the whole reason
this judge exists: it states, to the grader, exactly the distinction the
substring checks could not make.
"""


VERDICT_TOKENS = 120
"""Enough for ``YES`` or ``NO`` and one sentence, with room to spare."""


def _output_cap_for(provider: str, model: str) -> int:
    """A cap the model can actually answer within, reasoning included.

    A judge that reasons before answering spends tokens on the reasoning first.
    Budget only for the verdict and the reasoning consumes the whole allowance,
    leaving an empty completion — the failure that voided an entire benchmark
    run once already. So the measured overhead for this model is added to what
    the verdict itself needs.

    A model that is not on the free allowlist has no measured overhead, so it is
    given the floor rather than assumed to answer directly. Being too generous
    here costs nothing: the cap bounds a two-line reply, not a budget.
    """
    spec = find_free_model(provider, model)
    overhead = spec.reasoning_overhead if spec is not None else REASONING_FLOOR
    return overhead + VERDICT_TOKENS


class LLMJudge:
    """A judge backed by a real model, through the ordinary provider interface.

    Knows nothing about any vendor. Give it a budgeted provider and its requests
    are counted against the run's ceiling along with everything else.
    """

    def __init__(self, provider: LLMProvider, *, prompt: Prompt = JUDGE_PROMPT_V1) -> None:
        self.provider = provider
        self.prompt = prompt
        info = provider.model_info()
        self._name = f"llm:{info.provider}/{info.model}:{prompt.identifier}"
        self._output_cap = _output_cap_for(info.provider, info.model)

    @property
    def name(self) -> str:
        """Carries the judging model, so the fingerprint records who graded.

        Two runs judged by different models are different experiments, and an id
        that did not move between them would file both under one identity.
        """
        return self._name

    def judge(self, response: str, question: str) -> JudgeVerdict:
        """Ask the model, and refuse to guess when it does not answer.

        **``LLMError`` becomes an undecided verdict, not a failed one.** A rate
        limit says nothing about the answer being graded.

        ``RequestBudgetExhausted`` is deliberately not caught: it is not an
        ``LLMError``, and a judge quietly absorbing it would let a run continue
        past the ceiling it was given.
        """
        request = GenerationRequest.of(
            ChatMessage.system(self.prompt.text),
            ChatMessage.user(f"QUESTION:\n{question}\n\nANSWER TO GRADE:\n{response}"),
            max_output_tokens=self._output_cap,
            temperature=0.0,
        )
        try:
            result = self.provider.generate(request)
        except LLMError as exc:
            return JudgeVerdict.undecided(f"judge model failed: {exc}")
        return parse_verdict(result.text)


def parse_verdict(text: str) -> JudgeVerdict:
    """Read YES or NO off the model's first meaningful line.

    **Anything else is undecided.** A judge that guessed when it could not read
    its own model's reply would manufacture verdicts, which is the failure this
    whole module exists to avoid — and it would do so silently, since a
    manufactured NO looks exactly like a real one.

    Tolerant of the shapes a model actually produces around a one-word answer:
    surrounding whitespace, markdown emphasis, and trailing punctuation. Not
    tolerant of a verdict buried in prose, because finding one there means
    deciding which of several words was the answer.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return JudgeVerdict.undecided("judge returned nothing")

    head = lines[0].strip().strip("*_`#").strip().rstrip(".:,!").strip().upper()
    reasoning = lines[1] if len(lines) > 1 else ""

    if head == "YES":
        return JudgeVerdict(passed=True, reasoning=reasoning)
    if head == "NO":
        return JudgeVerdict(passed=False, reasoning=reasoning)
    return JudgeVerdict.undecided(f"could not read a verdict from {lines[0]!r}")


__all__ = [
    "JUDGE_PROMPT_V1",
    "EvaluationJudge",
    "JudgeVerdict",
    "KeywordJudge",
    "LLMJudge",
    "parse_verdict",
]
