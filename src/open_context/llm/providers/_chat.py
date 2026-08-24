"""Shared implementation for chat-completions providers.

Groq and OpenRouter both speak the chat-completions wire format. This module
holds the translation both need — request body in, ``GenerationResult`` out,
error status to ``LLMError`` — so it exists once rather than twice.

**This is shared implementation, not a second abstraction.** There is no
`OpenAICompatibleProvider` class that Groq and OpenRouter configure with a base
URL. Each is its own provider with its own name, credential, endpoint, headers,
and defaults; they call these functions the way two classes call a shared helper.
The module is private and exported nowhere. Anything that would make a caller
say "which OpenAI-compatible provider is this" belongs in the concrete provider,
not here.

**Nothing here invents.** A field the provider did not send becomes ``None``: an
absent usage block is not replaced with a tokenizer estimate, and an unknown
finish reason maps to ``UNKNOWN`` rather than being assumed to be a clean stop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from open_context.llm.errors import (
    AuthenticationError,
    ContextLimitExceededError,
    EmptyCompletionError,
    InvalidRequestError,
    ModelUnavailableError,
    ProviderUnavailableError,
    RateLimitError,
)
from open_context.llm.free_models import check_output_cap
from open_context.llm.provider import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    TokenUsage,
)
from open_context.llm.providers._http import HttpResponse

FINISH_REASONS: Mapping[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "end_turn": FinishReason.STOP,
    "eos": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "model_length": FinishReason.LENGTH,
    "content_filter": FinishReason.FILTERED,
    "tool_calls": FinishReason.OTHER,
    "function_call": FinishReason.OTHER,
}
"""Wire values to the neutral enum.

Anything absent maps to ``UNKNOWN``. A provider-specific reason does not get a
new member of the core enum: the enum describes what a caller can act on, and a
name only one vendor uses is not that.
"""

CONTEXT_LIMIT_MARKERS = (
    "context_length_exceeded",
    "context length",
    "maximum context",
    "too many tokens",
    "reduce the length",
)
"""Substrings that mean a 400 was really a context overflow.

Matched on the provider's own error text because these APIs report it as an
ordinary bad request. Crude, and a miss is not dangerous — it degrades to
``InvalidRequestError``, which is still true.
"""

BODY_SAMPLE = 400


def build_payload(request: GenerationRequest, model: str, *, provider: str = "") -> dict[str, Any]:
    """A chat-completions body from a neutral request.

    Every generic field the interface defines is translated. Nothing is dropped
    silently: if a field cannot be honoured the provider raises rather than
    quietly sending a request that means something else.

    ``options`` is merged last, so a caller who deliberately reaches for a
    provider-specific setting gets it, and gets to see that they did so.
    """
    if provider:
        check_output_cap(provider, model, request.max_output_tokens)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message(message) for message in request.messages],
    }
    if request.max_output_tokens is not None:
        payload["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.stop:
        payload["stop"] = list(request.stop)
    payload.update(request.options)
    return payload


def _message(message: Any) -> dict[str, Any]:  # noqa: ANN401  a ChatMessage
    wire: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.name:
        wire["name"] = message.name
    return wire


def parse_result(response: HttpResponse, *, provider: str, model: str) -> GenerationResult:
    """A neutral result from a successful chat-completions response."""
    document = _decode(response, provider)
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderUnavailableError(f"{provider} returned no choices: {_sample(response.body)}")

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise ProviderUnavailableError(
            f"{provider} returned a choice with no message: {_sample(response.body)}"
        )

    content = message.get("content")
    text = strip_inline_reasoning(content if isinstance(content, str) else "")
    finish_reason = finish_reason_of(first.get("finish_reason"))
    usage = usage_of(document.get("usage"))

    if not text.strip() and finish_reason is FinishReason.LENGTH:
        raise EmptyCompletionError(
            _empty_completion_detail(provider, model, usage),
            output_tokens=usage.output_tokens if usage else None,
            reasoning_tokens=usage.reasoning_tokens if usage else None,
        )

    return GenerationResult(
        text=text,
        provider=provider,
        model=str(document.get("model") or model),
        finish_reason=finish_reason,
        usage=usage,
        raw=document,
    )


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def strip_inline_reasoning(text: str) -> str:
    """Drop a leading ``<think>`` block that the model wrote into its own answer.

    Some reasoning models keep their thinking in a separate response field, where
    it is easy to ignore. Others — ``qwen/qwen3.6-27b`` among the approved free
    models — write it inline at the top of ``content`` and report no reasoning
    token count at all, so nothing in the accounting says it happened.

    Left in place it is not merely noise: it is *reasoning text presented as the
    answer*. A compactor would store it as the summary, a judge would grade it,
    and an extractor would mine it for state — every one of them reading the
    model's deliberation as its conclusion. Measured at a 160-token cap, this
    model spent all 158 tokens inside ``<think>`` and never reached an answer.

    Only a leading block is removed, and only through its terminator. An
    unterminated block means the whole response is thinking, which leaves an
    empty string — the truthful answer, and one the empty-completion check above
    turns into an error rather than an empty summary.
    """
    if not text.lstrip().startswith(THINK_OPEN):
        return text
    end = text.find(THINK_CLOSE)
    return "" if end == -1 else text[end + len(THINK_CLOSE) :].lstrip()


def _empty_completion_detail(provider: str, model: str, usage: TokenUsage | None) -> str:
    """Say what was spent, because that is what names the fix."""
    detail = f"{provider}/{model} hit the output cap without emitting any content"
    if usage is None or usage.output_tokens is None:
        return detail
    if usage.reasoning_tokens:
        return (
            f"{detail}: {usage.reasoning_tokens} of {usage.output_tokens} output tokens "
            f"went to reasoning. Raise max_output_tokens above the model's thinking cost, "
            f"or use a model that does not reason."
        )
    return f"{detail}: {usage.output_tokens} output tokens produced nothing"


def finish_reason_of(value: Any) -> FinishReason:  # noqa: ANN401  provider values are untyped
    if not isinstance(value, str):
        return FinishReason.UNKNOWN
    return FINISH_REASONS.get(value.lower(), FinishReason.UNKNOWN)


def usage_of(value: Any) -> TokenUsage | None:  # noqa: ANN401  provider values are untyped
    """Provider-reported usage, or nothing.

    ``None`` when the provider said nothing. It is never filled in from a
    tokenizer: a number that might be measured or might be estimated is worse
    than a missing one, and the two are separate types for that reason.
    """
    if not isinstance(value, dict):
        return None
    prompt = value.get("prompt_tokens")
    completion = value.get("completion_tokens")
    if not isinstance(prompt, int) and not isinstance(completion, int):
        return None
    details = value.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    return TokenUsage(
        input_tokens=prompt if isinstance(prompt, int) else None,
        output_tokens=completion if isinstance(completion, int) else None,
        reasoning_tokens=reasoning if isinstance(reasoning, int) else None,
    )


def raise_for_status(response: HttpResponse, *, provider: str, model: str) -> None:
    """Translate an error status into the neutral hierarchy.

    A vendor's status code and error document never escape this boundary. The
    provider's own message is carried through, because "which model" or "which
    field" is the useful part and the category alone cannot say it.
    """
    if response.ok:
        return

    detail = _error_detail(response)
    status = response.status

    if status in {401, 403}:
        raise AuthenticationError(f"{provider} rejected the credentials: {detail}")
    if status == 429:
        raise RateLimitError(
            f"{provider} is rate limiting: {detail}", retry_after=_retry_after(response)
        )
    if status == 404:
        raise ModelUnavailableError(f"{provider} has no model {model!r}: {detail}")
    if status in {400, 422}:
        lowered = detail.lower()
        if any(marker in lowered for marker in CONTEXT_LIMIT_MARKERS):
            raise ContextLimitExceededError(f"{provider} rejected the request: {detail}")
        raise InvalidRequestError(f"{provider} rejected the request: {detail}")
    if status >= 500:
        raise ProviderUnavailableError(f"{provider} returned {status}: {detail}")
    raise ProviderUnavailableError(f"{provider} returned an unexpected {status}: {detail}")


def _retry_after(response: HttpResponse) -> float | None:
    """Seconds the provider asked us to wait, when it said.

    Carried, not acted on. Nothing here retries.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _error_detail(response: HttpResponse) -> str:
    document = _try_decode(response.body)
    if isinstance(document, dict):
        error = document.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        if isinstance(error, str) and error:
            return error
        message = document.get("message")
        if isinstance(message, str) and message:
            return message
    return _sample(response.body)


def _decode(response: HttpResponse, provider: str) -> dict[str, Any]:
    document = _try_decode(response.body)
    if not isinstance(document, dict):
        raise ProviderUnavailableError(
            f"{provider} returned a body that is not a JSON object: {_sample(response.body)}"
        )
    return document


def _try_decode(body: bytes) -> Any:  # noqa: ANN401  arbitrary provider JSON
    try:
        return json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _sample(body: bytes) -> str:
    """A bounded, printable fragment of a response body.

    Bounded so an error cannot carry a whole document into a log. Response
    bodies never contain the credential, which travels in a request header and
    is redacted wherever a request is rendered.
    """
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return f"<{len(body)} bytes>"
    return text[:BODY_SAMPLE].strip() or f"<{len(body)} bytes>"


__all__ = [
    "CONTEXT_LIMIT_MARKERS",
    "FINISH_REASONS",
    "build_payload",
    "finish_reason_of",
    "parse_result",
    "raise_for_status",
    "usage_of",
]
