"""Deterministic implementations, for tests and for offline development.

Shipped in the package rather than kept in the test suite, because every later
phase will need a provider that answers predictably: a compaction engine cannot
be tested against a model whose output changes between runs, and neither can
anything a user builds on top of this.

**Nothing here reaches the network, a process, or a credential.** These are the
whole provider surface exercised by the test suite, which is why the suite runs
with no API key, no internet, and nothing installed.

**Deterministic means scripted, not clever.** ``FakeProvider`` returns responses
from a queue, or a fixed reply, or whatever a supplied callable computes. It
does not attempt to be a language model, and it should never grow toward being
one; a fake that guesses is a fake that fails differently from the real thing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from open_context.llm.config import ProviderConfig, register_provider
from open_context.llm.errors import (
    MalformedStructuredOutputError,
    UnsupportedCapabilityError,
)
from open_context.llm.info import Capability, ModelInfo
from open_context.llm.messages import ChatMessage
from open_context.llm.provider import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    StructuredResult,
    TokenUsage,
)
from open_context.llm.structured import parse_structured_output
from open_context.llm.tokens import TokenCount

PROVIDER_NAME = "fake"

DEFAULT_CAPABILITIES = frozenset(
    {
        Capability.TEXT_GENERATION,
        Capability.STRUCTURED_OUTPUT,
        Capability.EXACT_TOKEN_COUNT,
    }
)

Responder = Callable[[GenerationRequest], str]


class FakeProvider:
    """A provider that returns what it was told to return.

    ``responses`` is consumed in order; once it runs out, ``reply`` is used, or
    ``responder`` if one was given. ``requests`` records everything it was
    asked, so a test can assert on the prompt a caller built without a network
    in the way.
    """

    def __init__(
        self,
        *,
        model: str = "fake-model",
        responses: Sequence[str] | None = None,
        reply: str = "",
        responder: Responder | None = None,
        info: ModelInfo | None = None,
        capabilities: frozenset[Capability] | None = None,
        context_window: int | None = 8_192,
        usage: TokenUsage | None = None,
        finish_reason: FinishReason = FinishReason.STOP,
    ) -> None:
        self._info = info or ModelInfo(
            provider=PROVIDER_NAME,
            model=model,
            context_window=context_window,
            max_output_tokens=1_024,
            tokenizer="fake-words",
            capabilities=DEFAULT_CAPABILITIES if capabilities is None else capabilities,
        )
        self._queue = list(responses or ())
        self.reply = reply
        self.responder = responder
        self.usage = usage
        self.finish_reason = finish_reason
        self.requests: list[GenerationRequest] = []

    # ------------------------------------------------------------------

    def model_info(self) -> ModelInfo:
        return self._info

    def _require(self, capability: Capability) -> None:
        if not self._info.supports(capability):
            raise UnsupportedCapabilityError(
                capability.value, provider=self._info.provider, model=self._info.model
            )

    def _next_text(self, request: GenerationRequest) -> str:
        if self._queue:
            return self._queue.pop(0)
        if self.responder is not None:
            return self.responder(request)
        return self.reply

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._require(Capability.TEXT_GENERATION)
        self.requests.append(request)
        text = self._next_text(request)
        return GenerationResult(
            text=text,
            provider=self._info.provider,
            model=self._info.model,
            finish_reason=self.finish_reason,
            usage=self.usage,
            raw={"text": text},
        )

    def structured_output(
        self, request: GenerationRequest, schema: Mapping[str, Any]
    ) -> StructuredResult:
        self._require(Capability.STRUCTURED_OUTPUT)
        self.requests.append(request)
        text = self._next_text(request)
        data = parse_structured_output(text, schema)
        return StructuredResult(
            text=text,
            data=data,
            provider=self._info.provider,
            model=self._info.model,
            finish_reason=self.finish_reason,
            usage=self.usage,
            raw={"text": text},
        )


class FailingProvider:
    """A provider that raises whatever it was built with.

    For testing that the error model reaches a caller intact. It raises the
    given ``LLMError`` from every method, which is the shape a real provider's
    translation layer produces.
    """

    def __init__(self, error: Exception, *, info: ModelInfo | None = None) -> None:
        self.error = error
        self._info = info or ModelInfo(provider=PROVIDER_NAME, model="failing-model")

    def model_info(self) -> ModelInfo:
        return self._info

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise self.error

    def structured_output(
        self, request: GenerationRequest, schema: Mapping[str, Any]
    ) -> StructuredResult:
        raise self.error


class WordTokenizer:
    """Counts whitespace-separated words, exactly or as an estimate.

    ``exact`` is a constructor argument rather than a property of the algorithm,
    because the point of this class is to exercise both paths. Counting words is
    of course not how any real model counts tokens; what is being tested is that
    the *plumbing* keeps an exact count exact and an estimate estimated.

    ``exact_messages`` is separate because a tokenizer can know a model's
    vocabulary and still not know its chat template, which is exactly when text
    counts exactly and message counts do not.
    """

    def __init__(
        self,
        info: ModelInfo | None = None,
        *,
        exact: bool = True,
        exact_messages: bool | None = None,
        message_overhead: int = 3,
    ) -> None:
        self._info = info or ModelInfo(
            provider=PROVIDER_NAME,
            model="fake-model",
            context_window=8_192,
            tokenizer="fake-words",
            capabilities=frozenset({Capability.EXACT_TOKEN_COUNT}) if exact else frozenset(),
        )
        self.exact = exact
        self.exact_messages = exact if exact_messages is None else exact_messages
        self.message_overhead = message_overhead
        self.method = "fake-words"

    def model_info(self) -> ModelInfo:
        return self._info

    def count_text(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), exact=self.exact, method=self.method)

    def count_messages(self, messages: Sequence[ChatMessage]) -> TokenCount:
        words = sum(len(message.content.split()) for message in messages)
        return TokenCount(
            count=words + self.message_overhead * len(messages),
            exact=self.exact_messages,
            method=self.method,
        )


def _build(config: ProviderConfig) -> FakeProvider:
    """Factory used by the registry. Ignores credentials, because it needs none."""
    reply = str(config.options.get("reply", ""))
    return FakeProvider(model=config.model, reply=reply)


register_provider(PROVIDER_NAME, _build, replace_existing=True)


__all__ = [
    "DEFAULT_CAPABILITIES",
    "PROVIDER_NAME",
    "FailingProvider",
    "FakeProvider",
    "MalformedStructuredOutputError",
    "Responder",
    "WordTokenizer",
]
