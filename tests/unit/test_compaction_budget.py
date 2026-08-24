"""Budget allocation and rendering. No archive and no model."""

import pytest

from open_context.compaction import (
    MINIMUM_TARGET_TOKENS,
    BaselineConfig,
    InvalidConfigurationError,
    TargetTooSmallError,
    allocate,
    render_event,
    render_events,
    to_message,
)
from open_context.importers import EventType, RawEvent
from open_context.llm import ChatRole

# ----------------------------------------------------------------------
# Allocation


def test_the_default_split_is_sixty_forty():
    allocation = allocate(10_000, BaselineConfig())
    assert allocation.recent_tokens == 4_000
    assert allocation.summary_tokens == 6_000
    assert allocation.target_tokens == 10_000


def test_the_split_is_configurable():
    allocation = allocate(10_000, BaselineConfig(recent_fraction=0.25))
    assert allocation.recent_tokens == 2_500
    assert allocation.summary_tokens == 7_500


def test_an_allocation_never_exceeds_its_target():
    for target in (50, 51, 99, 100, 333, 1_000, 9_999):
        for fraction in (0.1, 0.4, 0.5, 0.9):
            allocation = allocate(target, BaselineConfig(recent_fraction=fraction))
            assert allocation.recent_tokens + allocation.summary_tokens <= target
            assert allocation.recent_tokens >= 1
            assert allocation.summary_tokens >= 1


def test_small_but_viable_targets_still_allocate():
    """The 60/40 split cannot be assumed to survive rounding at these sizes."""
    for target in (50, 100, 500, 1_000):
        allocation = allocate(target, BaselineConfig())
        assert allocation.recent_tokens >= 1
        assert allocation.summary_tokens >= 1


def test_a_target_below_the_minimum_is_refused():
    """Neither half is worth producing, so a result would misrepresent the work."""
    with pytest.raises(TargetTooSmallError) as info:
        allocate(10, BaselineConfig())
    assert info.value.target_tokens == 10
    assert info.value.minimum == MINIMUM_TARGET_TOKENS
    assert "no useful representation" in str(info.value)


def test_the_minimum_is_configurable():
    allocation = allocate(10, BaselineConfig(minimum_target_tokens=5))
    assert allocation.target_tokens == 10


def test_nonsense_configuration_is_refused():
    for fraction in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(InvalidConfigurationError, match="recent_fraction"):
            BaselineConfig(recent_fraction=fraction)
    with pytest.raises(InvalidConfigurationError, match="context_fill_fraction"):
        BaselineConfig(context_fill_fraction=0.0)
    with pytest.raises(InvalidConfigurationError, match="summary_retry_fraction"):
        BaselineConfig(summary_retry_fraction=1.0)
    with pytest.raises(InvalidConfigurationError, match="minimum_target_tokens"):
        BaselineConfig(minimum_target_tokens=0)
    with pytest.raises(InvalidConfigurationError, match="prompt_reserve_tokens"):
        BaselineConfig(prompt_reserve_tokens=-1)


def test_allocation_is_deterministic():
    config = BaselineConfig()
    assert allocate(7_777, config) == allocate(7_777, config)


# ----------------------------------------------------------------------
# Rendering


def event(**kwargs):
    return RawEvent(provider="test", **kwargs)


def test_a_message_renders_with_its_kind():
    rendered = render_event(event(type=EventType.USER_MESSAGE, text="hello"))
    assert rendered == "[user_message] hello"


def test_a_tool_call_renders_with_its_name_and_arguments():
    """Arguments live in raw, so an event with no text must still say what it did."""
    rendered = render_event(
        event(
            type=EventType.TOOL_CALL,
            tool_name="search",
            raw={"tool_name": "search", "arguments": {"q": "postgres"}},
        )
    )
    assert "tool_call" in rendered
    assert "name=search" in rendered
    assert "postgres" in rendered


def test_an_unrecognised_event_keeps_the_providers_own_word():
    rendered = render_event(
        event(type=EventType.OTHER, source_type="checkpoint", raw={"branch": "main"})
    )
    assert "type=checkpoint" in rendered
    assert "main" in rendered


def test_roles_map_to_the_wire_vocabulary():
    assert to_message(event(type=EventType.USER_MESSAGE, text="x")).role is ChatRole.USER
    assert to_message(event(type=EventType.ASSISTANT_MESSAGE, text="x")).role is ChatRole.ASSISTANT
    assert to_message(event(type=EventType.SYSTEM_MESSAGE, text="x")).role is ChatRole.SYSTEM
    assert to_message(event(type=EventType.TOOL_RESULT, text="x")).role is ChatRole.TOOL


def test_a_tool_call_is_attributed_to_the_assistant():
    """The assistant made the call; the result is what comes back as tool."""
    assert to_message(event(type=EventType.TOOL_CALL, tool_name="t")).role is ChatRole.ASSISTANT


def test_an_unknown_kind_falls_back_without_losing_its_name():
    message = to_message(event(type=EventType.OTHER, source_type="critic", text="needs work"))
    assert message.role is ChatRole.USER
    assert "type=critic" in message.content


def test_rendering_preserves_order():
    events = [event(type=EventType.USER_MESSAGE, text=str(i)) for i in range(5)]
    assert render_events(events).splitlines() == [f"[user_message] {i}" for i in range(5)]


def test_rendering_is_deterministic():
    one = event(type=EventType.OTHER, raw={"b": 2, "a": 1})
    assert render_event(one) == render_event(one)
    assert render_event(one).index('"a"') < render_event(one).index('"b"'), "keys are sorted"


def test_an_event_with_nothing_in_it_still_renders():
    assert render_event(event(type=EventType.OTHER)) == "[other] "
