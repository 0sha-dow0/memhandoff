"""Groq and OpenRouter, against fake HTTP.

**No test here opens a socket or reads a credential.** Transport is injectable,
so these exercise the real request translation, response parsing, and error
mapping against canned responses. A test that needed the network would be a test
nobody runs.
"""

import json

import pytest

from open_context.llm import (
    AuthenticationError,
    Capability,
    ChatMessage,
    ContextLimitExceededError,
    EmptyCompletionError,
    FinishReason,
    GenerationRequest,
    InvalidRequestError,
    LLMProvider,
    ModelUnavailableError,
    ProviderConfig,
    ProviderUnavailableError,
    RateLimitError,
    UnsupportedCapabilityError,
    create_provider,
    registered_providers,
)
from open_context.llm.free_models import APPROVED_FREE_MODELS
from open_context.llm.providers import GroqProvider, OpenRouterProvider, openrouter
from open_context.llm.providers._http import HttpRequest, HttpResponse, TransportError

APPROVED = APPROVED_FREE_MODELS[0].model_id

KEY = "test-key-not-a-real-credential"

# Request translation and error mapping are properties of the implementation,
# not of the billing policy, so these construct providers with the allowlist
# gate off. Every production path leaves it on; that is asserted in
# tests/unit/test_free_models.py, which also proves an unapproved model costs
# zero HTTP requests.
UNGATED = {"enforce_free": False}
PROMPT = GenerationRequest.of(ChatMessage.user("hello"))


def completion(text="hi", *, model="served-model", finish="stop", usage=True):
    document = {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ],
    }
    if usage:
        document["usage"] = {"prompt_tokens": 31, "completion_tokens": 7, "total_tokens": 38}
    return HttpResponse(status=200, body=json.dumps(document).encode())


class Recorder:
    """A transport that records what it was asked and replies with a script."""

    def __init__(self, *responses, raises=None):
        self.responses = list(responses) or [completion()]
        self.raises = raises
        self.requests: list[HttpRequest] = []

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        assert isinstance(response, HttpResponse)
        return response

    @property
    def payload(self):
        return json.loads(self.requests[-1].body)


def providers(**kwargs):
    """Both implementations, so every behaviour is checked on each."""
    transport = kwargs.pop("transport")
    return [
        GroqProvider(model="m", api_key=KEY, transport=transport, **kwargs, **UNGATED),
        OpenRouterProvider(model="m", api_key=KEY, transport=transport, **kwargs, **UNGATED),
    ]


# ----------------------------------------------------------------------
# Registration and identity


def test_both_providers_register_themselves():
    assert "groq" in registered_providers()
    assert "openrouter" in registered_providers()


def test_both_satisfy_the_provider_interface():
    for provider in providers(transport=Recorder()):
        assert isinstance(provider, LLMProvider)


def test_each_reports_its_own_name():
    transport = Recorder()
    assert (
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).model_info().provider
        == "groq"
    )
    assert (
        OpenRouterProvider(model="m", api_key=KEY, transport=transport, **UNGATED)
        .model_info()
        .provider
        == "openrouter"
    )


def test_each_uses_its_own_endpoint():
    transport = Recorder()
    for provider in providers(transport=transport):
        provider.generate(PROMPT)
    assert "api.groq.com" in transport.requests[0].url
    assert "openrouter.ai" in transport.requests[1].url


def test_the_served_model_identifier_is_recorded_not_the_requested_one():
    """A provider may route elsewhere; the result says where it actually went."""
    transport = Recorder(completion(model="meta/llama-served"))
    for provider in providers(transport=transport):
        assert provider.generate(PROMPT).model == "meta/llama-served"


def test_no_model_is_hard_coded():
    transport = Recorder()
    provider = GroqProvider(
        model="whatever-the-account-has", api_key=KEY, transport=transport, **UNGATED
    )
    provider.generate(PROMPT)
    assert transport.payload["model"] == "whatever-the-account-has"


# ----------------------------------------------------------------------
# Request translation


def test_messages_are_translated():
    transport = Recorder()
    request = GenerationRequest.of(ChatMessage.system("be brief"), ChatMessage.user("hello"))
    GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(request)

    assert transport.payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_a_named_message_keeps_its_name():
    transport = Recorder()
    request = GenerationRequest.of(ChatMessage(role="user", content="x", name="dana"))
    GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(request)
    assert transport.payload["messages"][0]["name"] == "dana"


def test_max_output_tokens_temperature_and_stop_are_all_sent():
    """No generic field the interface defines is silently discarded."""
    transport = Recorder()
    request = GenerationRequest.of(
        ChatMessage.user("x"), max_output_tokens=128, temperature=0.25, stop=("END", "STOP")
    )
    for provider in providers(transport=transport):
        provider.generate(request)
        assert transport.payload["max_tokens"] == 128
        assert transport.payload["temperature"] == 0.25
        assert transport.payload["stop"] == ["END", "STOP"]


def test_absent_optional_fields_are_not_sent():
    transport = Recorder()
    GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)
    assert "max_tokens" not in transport.payload
    assert "temperature" not in transport.payload
    assert "stop" not in transport.payload


def test_provider_specific_options_pass_through():
    transport = Recorder()
    request = GenerationRequest.of(ChatMessage.user("x"), options={"top_p": 0.1, "seed": 7})
    GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(request)
    assert transport.payload["top_p"] == 0.1
    assert transport.payload["seed"] == 7


def test_both_identify_the_client():
    """Providers sit behind edge protection that rejects an unidentified client."""
    from open_context.llm.providers._http import USER_AGENT

    transport = Recorder()
    for provider in providers(transport=transport):
        provider.generate(PROMPT)
    for request in transport.requests:
        assert request.headers["User-Agent"] == USER_AGENT


def test_openrouter_sends_attribution_only_when_configured():
    transport = Recorder()
    OpenRouterProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)
    assert "HTTP-Referer" not in transport.requests[-1].headers

    OpenRouterProvider(
        model="m",
        api_key=KEY,
        transport=transport,
        referrer="https://example.test",
        title="t",
        **UNGATED,
    ).generate(PROMPT)
    assert transport.requests[-1].headers["HTTP-Referer"] == "https://example.test"
    assert transport.requests[-1].headers["X-Title"] == "t"


# ----------------------------------------------------------------------
# Responses


def test_a_successful_generation():
    transport = Recorder(completion("the answer"))
    for provider in providers(transport=transport):
        result = provider.generate(PROMPT)
        assert result.text == "the answer"
        assert result.finish_reason is FinishReason.STOP


def test_usage_is_parsed():
    transport = Recorder(completion())
    for provider in providers(transport=transport):
        usage = provider.generate(PROMPT).usage
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens) == (31, 7)


def test_reasoning_tokens_are_broken_out_of_the_output_allowance():
    """A reasoning model spends part of the output cap before it writes anything.

    Left inside ``output_tokens`` the spend is invisible, and the reason a cap
    that looks generous produces nothing cannot be seen in the accounting.
    """
    document = {
        "model": "m",
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 89,
            "completion_tokens": 71,
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
    }
    transport = Recorder(HttpResponse(status=200, body=json.dumps(document).encode()))
    usage = (
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT).usage
    )
    assert usage is not None
    assert usage.reasoning_tokens == 50
    assert usage.output_tokens == 71, "reasoning is part of the output spend, not extra"


def test_a_model_cut_off_before_it_wrote_anything_is_an_error():
    """The defect that voided the first Phase 8.5 run.

    ``openai/gpt-oss-20b`` at a 160-token cap spent 158 tokens reasoning and
    returned ``content: ""`` with ``finish_reason: length``. Handed back as an
    empty string it became an empty compacted context, which scored as a real
    measurement: every compaction arm near zero, the reference arm — which
    makes no compaction call — untouched at 1.00.
    """
    for provider in providers(transport=Recorder(completion("", finish="length"))):
        with pytest.raises(EmptyCompletionError):
            provider.generate(PROMPT)


def test_the_empty_completion_error_names_the_fix():
    document = {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 2400,
            "completion_tokens": 160,
            "completion_tokens_details": {"reasoning_tokens": 158},
        },
    }
    transport = Recorder(HttpResponse(status=200, body=json.dumps(document).encode()))
    provider = GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED)
    with pytest.raises(EmptyCompletionError) as caught:
        provider.generate(PROMPT)
    assert caught.value.reasoning_tokens == 158
    assert caught.value.output_tokens == 160
    assert "158 of 160" in str(caught.value)
    assert "max_output_tokens" in str(caught.value)


def test_whitespace_only_is_as_empty_as_empty():
    """A cut-off model that emitted one newline has still said nothing."""
    for provider in providers(transport=Recorder(completion("  \n ", finish="length"))):
        with pytest.raises(EmptyCompletionError):
            provider.generate(PROMPT)


def test_a_truncated_answer_that_said_something_is_still_an_answer():
    """Truncation is not itself the failure — losing the whole response is.

    A summary cut off mid-sentence is degraded but usable, and the caller can
    see ``finish_reason`` and decide. Raising here would discard real output.
    """
    for provider in providers(transport=Recorder(completion("the answer beg", finish="length"))):
        result = provider.generate(PROMPT)
        assert result.text == "the answer beg"
        assert result.finish_reason is FinishReason.LENGTH


def test_an_empty_answer_the_model_chose_is_not_an_error():
    """Only the cap turns silence into a failure. A clean stop is the model's call."""
    for provider in providers(transport=Recorder(completion("", finish="stop"))):
        assert provider.generate(PROMPT).text == ""


def test_inline_thinking_is_not_returned_as_the_answer():
    """``qwen/qwen3.6-27b`` writes its reasoning into ``content`` and reports no
    reasoning token count, so nothing in the accounting shows it happened.

    Kept, it is reasoning text presented as the answer: a compactor stores it as
    the summary, a judge grades it, an extractor mines it for state.
    """
    for provider in providers(
        transport=Recorder(completion("<think>\nWeigh it up\n</think>\n\nok"))
    ):
        assert provider.generate(PROMPT).text == "ok"


def test_a_response_that_is_nothing_but_unfinished_thinking_is_an_error():
    """At a 160-token cap this model spent all 158 inside ``<think>``.

    Stripping leaves nothing, which is the truthful reading — and the
    empty-completion check turns it into an error rather than an empty summary.
    """
    thinking = completion("<think>\nStill working through the first stage", finish="length")
    for provider in providers(transport=Recorder(thinking)):
        with pytest.raises(EmptyCompletionError):
            provider.generate(PROMPT)


def test_only_a_leading_block_is_stripped():
    """Prose that merely mentions the tag is the model's answer, not its thinking."""
    answer = "Use <think> tags to mark deliberation."
    for provider in providers(transport=Recorder(completion(answer))):
        assert provider.generate(PROMPT).text == answer


def test_stripping_leaves_an_ordinary_answer_untouched():
    for provider in providers(transport=Recorder(completion("plain answer"))):
        assert provider.generate(PROMPT).text == "plain answer"


def test_absent_usage_is_none_never_an_estimate():
    """A tokenizer estimate must never be passed off as provider accounting."""
    transport = Recorder(completion(usage=False))
    for provider in providers(transport=transport):
        assert provider.generate(PROMPT).usage is None


def test_partial_usage_is_kept_partial():
    document = {
        "model": "m",
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12},
    }
    transport = Recorder(HttpResponse(status=200, body=json.dumps(document).encode()))
    usage = (
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT).usage
    )
    assert usage is not None
    assert usage.input_tokens == 12
    assert usage.output_tokens is None


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("end_turn", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("max_tokens", FinishReason.LENGTH),
        ("content_filter", FinishReason.FILTERED),
        ("tool_calls", FinishReason.OTHER),
        ("something_new", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_finish_reasons_map_to_the_neutral_enum(wire, expected):
    transport = Recorder(completion(finish=wire))
    assert (
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED)
        .generate(PROMPT)
        .finish_reason
        is expected
    )


def test_the_provider_response_is_preserved():
    transport = Recorder(completion())
    raw = GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT).raw
    assert raw["id"] == "chatcmpl-1"


# ----------------------------------------------------------------------
# Errors


def error_response(status, message="something went wrong", headers=None):
    body = json.dumps({"error": {"message": message, "type": "x"}}).encode()
    return HttpResponse(status=status, body=body, headers=headers or {})


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_credential_is_an_authentication_error(status):
    transport = Recorder(error_response(status, "Invalid API Key"))
    for provider in providers(transport=transport):
        with pytest.raises(AuthenticationError, match="Invalid API Key"):
            provider.generate(PROMPT)


def test_a_rate_limit_carries_retry_after():
    transport = Recorder(error_response(429, "slow down", headers={"retry-after": "12.5"}))
    for provider in providers(transport=transport):
        with pytest.raises(RateLimitError) as info:
            provider.generate(PROMPT)
        assert info.value.retry_after == 12.5


def test_a_rate_limit_without_retry_after_carries_none():
    transport = Recorder(error_response(429))
    with pytest.raises(RateLimitError) as info:
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)
    assert info.value.retry_after is None


def test_an_unparsable_retry_after_is_none_not_zero():
    transport = Recorder(
        error_response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )
    with pytest.raises(RateLimitError) as info:
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)
    assert info.value.retry_after is None


def test_an_unknown_model_is_model_unavailable():
    transport = Recorder(error_response(404, "model not found"))
    with pytest.raises(ModelUnavailableError, match="model not found"):
        GroqProvider(model="ghost", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)


def test_a_bad_request_is_an_invalid_request():
    transport = Recorder(error_response(400, "temperature must be <= 2"))
    with pytest.raises(InvalidRequestError, match="temperature"):
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)


def test_a_context_overflow_is_recognised_behind_a_bad_request():
    """These APIs report it as a 400; the caller needs the specific error."""
    transport = Recorder(
        error_response(400, "Request too large: reduce the length of the messages")
    )
    for provider in providers(transport=transport):
        with pytest.raises(ContextLimitExceededError):
            provider.generate(PROMPT)


def test_a_server_error_is_provider_unavailable():
    transport = Recorder(error_response(503, "upstream is down"))
    with pytest.raises(ProviderUnavailableError, match="503"):
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)


def test_a_network_failure_becomes_provider_unavailable():
    transport = Recorder(raises=TransportError("connection refused"))
    for provider in providers(transport=transport):
        with pytest.raises(ProviderUnavailableError, match="connection refused"):
            provider.generate(PROMPT)


def test_a_timeout_becomes_provider_unavailable():
    transport = Recorder(raises=TransportError("timed out after 60.0s"))
    with pytest.raises(ProviderUnavailableError, match="timed out"):
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)


@pytest.mark.parametrize(
    "body",
    [b"not json at all", b"[]", b'{"choices": []}', b'{"choices": [{"no": "message"}]}'],
)
def test_a_malformed_response_is_provider_unavailable(body):
    transport = Recorder(HttpResponse(status=200, body=body))
    with pytest.raises(ProviderUnavailableError):
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)


def test_an_error_body_that_is_not_json_still_produces_a_message():
    transport = Recorder(HttpResponse(status=500, body=b"<html>Gateway Error</html>"))
    with pytest.raises(ProviderUnavailableError, match="Gateway Error"):
        GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)


def test_no_vendor_exception_escapes():
    """Every failure above arrived as an LLMError, not an HTTPError or a KeyError."""
    from open_context.llm import LLMError

    for status in (400, 401, 404, 429, 500):
        transport = Recorder(error_response(status))
        with pytest.raises(LLMError):
            GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)


# ----------------------------------------------------------------------
# Model info


def test_an_unconfigured_context_window_is_unknown_not_guessed():
    """Neither API returns one with a completion, so nothing here invents one."""
    for provider in providers(transport=Recorder()):
        assert provider.model_info().context_window is None
        assert provider.model_info().max_output_tokens is None


def test_a_configured_context_window_is_reported():
    provider = GroqProvider(
        model="m", api_key=KEY, transport=Recorder(), context_window=131_072, **UNGATED
    )
    assert provider.model_info().context_window == 131_072


def test_capabilities_claim_only_text_generation():
    for provider in providers(transport=Recorder()):
        info = provider.model_info()
        assert info.supports(Capability.TEXT_GENERATION)
        assert not info.supports(Capability.STRUCTURED_OUTPUT)
        assert not info.counts_exactly, "provider usage is not a tokenizer"


# ----------------------------------------------------------------------
# Authentication and construction


def test_a_missing_credential_is_refused_at_construction():
    with pytest.raises(AuthenticationError, match="GROQ_API_KEY"):
        GroqProvider(model="m", api_key="", **UNGATED)
    with pytest.raises(AuthenticationError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(model="m", api_key="", **UNGATED)


def test_the_allowlist_is_checked_before_the_credential():
    """An unapproved model is refused even when no key is present.

    Ordering matters: the billing gate runs first, so nothing about a request is
    assembled and no credential is even read for a model we may not call.
    """
    from open_context.llm.free_models import ModelNotApprovedError

    with pytest.raises(ModelNotApprovedError):
        GroqProvider(model="anything", api_key="")


def test_the_factory_reads_the_environment(monkeypatch):
    monkeypatch.setenv(openrouter.API_KEY_ENV, KEY)
    built = openrouter.build(ProviderConfig(provider="openrouter", model=APPROVED))
    assert built.model_info().model == APPROVED


def test_the_factory_fails_without_a_credential(monkeypatch):
    """Never a silent fall back to the fake or to another provider."""
    monkeypatch.delenv(openrouter.API_KEY_ENV, raising=False)
    with pytest.raises(AuthenticationError):
        create_provider(ProviderConfig(provider="openrouter", model=APPROVED))


def test_an_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv(openrouter.API_KEY_ENV, "from-environment")
    built = openrouter.build(
        ProviderConfig(provider="openrouter", model=APPROVED, api_key="explicit")
    )
    assert built.model_info().provider == "openrouter"


def test_selection_by_name_is_deterministic(monkeypatch):
    monkeypatch.setenv(openrouter.API_KEY_ENV, KEY)
    built = create_provider(ProviderConfig(provider="openrouter", model=APPROVED))
    assert built.model_info().provider == "openrouter"
    assert built.model_info().model == APPROVED


def test_options_configure_the_endpoint_and_timeout(monkeypatch):
    monkeypatch.setenv(openrouter.API_KEY_ENV, KEY)
    config = ProviderConfig(provider="openrouter", model=APPROVED).with_options(
        base_url="https://example.test/v1", timeout=5.0, context_window=8192
    )
    built = openrouter.build(config)
    assert built.base_url == "https://example.test/v1"
    assert built.timeout == 5.0
    assert built.model_info().context_window == 8192


# ----------------------------------------------------------------------
# Structured output


def test_structured_output_is_refused_rather_than_faked():
    """Prompting for JSON and calling it native would make the flag a guess."""
    for provider in providers(transport=Recorder()):
        with pytest.raises(UnsupportedCapabilityError, match="structured_output"):
            provider.structured_output(PROMPT, {"type": "object"})


# ----------------------------------------------------------------------
# Credentials never leak


def test_the_credential_travels_only_in_the_authorization_header():
    transport = Recorder()
    for provider in providers(transport=transport):
        provider.generate(PROMPT)
    for request in transport.requests:
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert KEY not in request.body.decode()
        assert KEY not in request.url


def test_redacted_headers_hide_the_credential():
    transport = Recorder()
    GroqProvider(model="m", api_key=KEY, transport=transport, **UNGATED).generate(PROMPT)
    redacted = transport.requests[-1].redacted_headers()

    assert redacted["Authorization"] == "<redacted>"
    assert KEY not in str(redacted)


def test_repr_never_renders_the_credential():
    for provider in providers(transport=Recorder()):
        assert KEY not in repr(provider)


def test_no_error_message_carries_the_credential():
    for status in (401, 429, 500):
        transport = Recorder(error_response(status))
        for provider in providers(transport=transport):
            try:
                provider.generate(PROMPT)
            except Exception as exc:
                assert KEY not in str(exc)
                assert KEY not in repr(exc)


def test_a_result_never_carries_the_credential():
    transport = Recorder(completion())
    for provider in providers(transport=transport):
        result = provider.generate(PROMPT)
        assert KEY not in result.model_dump_json()
