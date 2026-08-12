"""Session and Message."""

import pytest
from pydantic import ValidationError

from open_context.models import Message, Role, Session, TrustLevel


def make_message(**overrides):
    defaults = {"session_id": Session().id, "seq": 0, "role": Role.USER, "content": "hello"}
    return Message(**{**defaults, **overrides})


def test_message_round_trips_through_json():
    original = make_message(trust=TrustLevel.USER_AUTHORED, metadata={"k": [1, 2]})
    restored = Message.model_validate_json(original.model_dump_json())
    assert restored == original


def test_message_requires_valid_session_id():
    with pytest.raises(ValidationError, match="malformed identifier"):
        Message(session_id="not-an-id", seq=0, role=Role.USER, content="hi")


def test_message_rejects_message_id_in_session_field():
    other = make_message()
    with pytest.raises(ValidationError, match="expected an id with prefix"):
        Message(session_id=other.id, seq=0, role=Role.USER, content="hi")


def test_negative_seq_rejected():
    with pytest.raises(ValidationError):
        make_message(seq=-1)


def test_tool_message_requires_tool_name():
    with pytest.raises(ValidationError, match="must carry tool_name"):
        make_message(role=Role.TOOL)


def test_tool_name_forbidden_on_non_tool_message():
    with pytest.raises(ValidationError, match="only valid on tool messages"):
        make_message(role=Role.ASSISTANT, tool_name="grep")


def test_tool_message_accepts_tool_name():
    message = make_message(role=Role.TOOL, tool_name="grep", content="3 matches")
    assert message.tool_name == "grep"


def test_message_cannot_be_its_own_parent():
    message = make_message()
    with pytest.raises(ValidationError, match="cannot be its own parent"):
        Message(
            id=message.id,
            session_id=message.session_id,
            seq=1,
            role=Role.USER,
            content="x",
            parent_id=message.id,
        )


def test_empty_content_allowed():
    """Provider exports contain empty assistant turns. Rejecting them would make
    a real transcript unimportable."""
    assert make_message(role=Role.ASSISTANT, content="").content == ""


def test_default_trust_is_agent_generated():
    assert make_message().trust is TrustLevel.AGENT_GENERATED


def test_seq_orders_messages_sharing_a_timestamp():
    session = Session()
    stamp = session.created_at
    messages = [
        make_message(session_id=session.id, seq=i, timestamp=stamp, content=str(i))
        for i in (2, 0, 1)
    ]
    assert [m.content for m in sorted(messages, key=lambda m: m.seq)] == ["0", "1", "2"]


def test_session_defaults():
    session = Session()
    assert session.id.startswith("ses_")
    assert session.metadata == {}
    assert session.title is None
