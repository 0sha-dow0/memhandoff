"""The provider interface: generation, structured output, metadata, and errors.

Every test here runs offline with no credentials. That is the point of the
fakes, not an incidental property of the suite.
"""

import pytest
from pydantic import BaseModel

from open_context.llm import (
    Capability,
    ChatMessage,
    ChatRole,
    ContextLimitExceededError,
    FinishReason,
    GenerationRequest,
    LLMError,
    LLMProvider,
    MalformedStructuredOutputError,
    ModelInfo,
    RateLimitError,
    TokenUsage,
    UnsupportedCapabilityError,
    parse_structured_output,
    schema_for,
)
from open_context.llm.fakes import FailingProvider, FakeProvider

PROMPT = GenerationRequest.of(ChatMessage.user("hello"))


# ----------------------------------------------------------------------
# Generation


def test_basic_generation():
    provider = FakeProvider(reply="a reply")
    result = provider.generate(PROMPT)

    assert result.text == "a reply"
    assert result.provider == "fake"
    assert result.model == "fake-model"
    assert result.finish_reason is FinishReason.STOP


def test_a_provider_satisfies_the_protocol():
    assert isinstance(FakeProvider(), LLMProvider)


def test_scripted_responses_are_consumed_in_order():
    provider = FakeProvider(responses=["first", "second"], reply="afterwards")
    assert [provider.generate(PROMPT).text for _ in range(3)] == [
        "first",
        "second",
        "afterwards",
    ]


def test_a_provider_records_what_it_was_asked():
    """So a later phase can assert on the prompt it built, with no network."""
    provider = FakeProvider()
    provider.generate(GenerationRequest.of(ChatMessage.system("be brief"), ChatMessage.user("hi")))

    (request,) = provider.requests
    assert [m.role for m in request.messages] == [ChatRole.SYSTEM, ChatRole.USER]


def test_a_request_needs_at_least_one_message():
    with pytest.raises(ValueError):
        GenerationRequest(messages=())


def test_provider_reported_usage_is_carried_through():
    provider = FakeProvider(usage=TokenUsage(input_tokens=10, output_tokens=4))
    usage = provider.generate(PROMPT).usage

    assert usage is not None
    assert usage.total_tokens == 14


def test_usage_is_absent_rather_than_invented():
    """A provider that reports nothing produces None, never a tokenizer estimate."""
    assert FakeProvider().generate(PROMPT).usage is None


def test_partial_usage_has_no_total():
    assert TokenUsage(input_tokens=10).total_tokens is None


def test_a_truncated_answer_is_visible():
    provider = FakeProvider(reply="cut off mid-", finish_reason=FinishReason.LENGTH)
    assert provider.generate(PROMPT).finish_reason is FinishReason.LENGTH


def test_the_provider_response_is_preserved():
    """Same discipline as the import layer: keep what the provider actually sent."""
    assert FakeProvider(reply="x").generate(PROMPT).raw == {"text": "x"}


# ----------------------------------------------------------------------
# Structured output


class Extracted(BaseModel):
    name: str
    count: int


SCHEMA = schema_for(Extracted)


def test_structured_output_returns_parsed_data():
    provider = FakeProvider(reply='{"name": "postgres", "count": 2}')
    result = provider.structured_output(PROMPT, SCHEMA)

    assert result.data == {"name": "postgres", "count": 2}
    assert result.text == '{"name": "postgres", "count": 2}', "the raw text is kept too"


def test_a_schema_comes_from_a_pydantic_model():
    assert SCHEMA["type"] == "object"
    assert set(SCHEMA["required"]) == {"name", "count"}


def test_schema_for_refuses_a_non_model():
    with pytest.raises(TypeError, match="not a pydantic model"):
        schema_for(dict)


def test_a_fenced_response_is_still_parsed():
    """Models wrap JSON in markdown unprompted; that is not a malformed response."""
    provider = FakeProvider(reply='```json\n{"name": "x", "count": 1}\n```')
    assert provider.structured_output(PROMPT, SCHEMA).data["name"] == "x"


def test_prose_instead_of_json_is_malformed():
    provider = FakeProvider(reply="Sure! Here is the answer you wanted.")
    with pytest.raises(MalformedStructuredOutputError, match="not valid JSON") as info:
        provider.structured_output(PROMPT, SCHEMA)
    assert "Sure!" in info.value.text, "the error carries what the model actually said"


def test_an_empty_response_is_malformed():
    with pytest.raises(MalformedStructuredOutputError, match="no content"):
        FakeProvider(reply="   ").structured_output(PROMPT, SCHEMA)


def test_the_wrong_json_type_is_malformed():
    with pytest.raises(MalformedStructuredOutputError, match="should be a JSON object"):
        FakeProvider(reply="[1, 2, 3]").structured_output(PROMPT, SCHEMA)


def test_missing_required_fields_are_named():
    with pytest.raises(MalformedStructuredOutputError, match="missing required fields count, name"):
        FakeProvider(reply='{"other": 1}').structured_output(PROMPT, SCHEMA)


def test_a_single_missing_field_reads_correctly():
    with pytest.raises(MalformedStructuredOutputError, match="missing required field count"):
        FakeProvider(reply='{"name": "x"}').structured_output(PROMPT, SCHEMA)


def test_shape_checking_stops_where_it_says_it_does():
    """Documented limit: top-level shape only, not full JSON Schema validation.

    Pinned so the boundary of the guarantee is a test rather than a hope. A
    caller needing real validation validates the data itself.
    """
    data = parse_structured_output('{"name": 42, "count": "two"}', SCHEMA)
    assert data == {"name": 42, "count": "two"}, "value types are not checked here"

    with pytest.raises(ValueError):
        Extracted.model_validate(data)


def test_an_integer_schema_does_not_accept_a_boolean():
    """bool subclasses int in Python; JSON does not agree."""
    with pytest.raises(MalformedStructuredOutputError):
        parse_structured_output("true", {"type": "integer"})


# ----------------------------------------------------------------------
# Model information and capabilities


def test_model_metadata():
    info = FakeProvider(model="m-1", context_window=32_000).model_info()

    assert info.provider == "fake"
    assert info.model == "m-1"
    assert info.context_window == 32_000
    assert info.max_output_tokens == 1_024


def test_capability_detection():
    info = FakeProvider().model_info()

    assert info.supports(Capability.TEXT_GENERATION)
    assert info.supports(Capability.STRUCTURED_OUTPUT)
    assert info.counts_exactly
    assert not info.supports(Capability.VISION)


def test_there_is_no_capability_the_interface_cannot_act_on():
    """Every flag must answer a question that changes what a caller does.

    Streaming had a flag and no method, which a caller could read and do nothing
    with. It returns when there is a streaming call to guard.
    """
    assert not hasattr(Capability, "STREAMING")
    assert {c.value for c in Capability} == {
        "text_generation",
        "structured_output",
        "tool_calling",
        "vision",
        "exact_token_count",
    }


def test_an_unsupported_capability_raises_rather_than_degrading():
    """Falling back would make the flag a suggestion and mislead anyone who checked."""
    provider = FakeProvider(capabilities=frozenset({Capability.TEXT_GENERATION}))

    with pytest.raises(UnsupportedCapabilityError) as info:
        provider.structured_output(PROMPT, SCHEMA)
    assert info.value.capability == "structured_output"
    assert "fake/fake-model" in str(info.value)


def test_an_unknown_context_window_is_none_not_a_guess():
    """An invented window produces a budget that looks authoritative and is wrong."""
    info = ModelInfo(provider="p", model="m")
    assert info.context_window is None
    assert info.max_output_tokens is None


def test_a_context_window_must_be_positive_if_stated():
    with pytest.raises(ValueError):
        ModelInfo(provider="p", model="m", context_window=0)


def test_model_info_is_frozen():
    with pytest.raises(ValueError):
        FakeProvider().model_info().model = "something else"


# ----------------------------------------------------------------------
# Errors


def test_provider_errors_share_one_base():
    """So a caller can catch the boundary without enumerating vendors."""
    for error in (
        ContextLimitExceededError("too long"),
        RateLimitError("slow down"),
        UnsupportedCapabilityError("vision"),
        MalformedStructuredOutputError("bad json"),
    ):
        assert isinstance(error, LLMError)


def test_a_provider_failure_reaches_the_caller_intact():
    provider = FailingProvider(ContextLimitExceededError("too long", requested_tokens=200_000))

    with pytest.raises(ContextLimitExceededError) as info:
        provider.generate(PROMPT)
    assert info.value.requested_tokens == 200_000


def test_a_rate_limit_carries_what_the_provider_said():
    error = RateLimitError("slow down", retry_after=30.0)
    assert error.retry_after == 30.0, "retained for a future retry policy, not acted on here"


def test_a_context_limit_error_does_not_invent_numbers():
    error = ContextLimitExceededError("too long")
    assert error.requested_tokens is None
    assert error.context_window is None
