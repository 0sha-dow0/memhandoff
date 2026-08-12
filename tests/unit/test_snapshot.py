"""Context snapshots."""

import pytest
from pydantic import ValidationError

from open_context.models import ContextSnapshot, Session, TokenEstimate, ids

SESSION_ID = Session().id


def test_snapshot_defaults():
    snapshot = ContextSnapshot(session_id=SESSION_ID)
    assert snapshot.id.startswith("snap_")
    assert snapshot.parent_snapshot_id is None
    assert snapshot.token_estimate is None


def test_snapshot_chain_round_trips():
    first = ContextSnapshot(session_id=SESSION_ID)
    second = ContextSnapshot(session_id=SESSION_ID, parent_snapshot_id=first.id)
    restored = ContextSnapshot.model_validate_json(second.model_dump_json())
    assert restored.parent_snapshot_id == first.id


def test_snapshot_cannot_be_its_own_parent():
    snapshot = ContextSnapshot(session_id=SESSION_ID)
    with pytest.raises(ValidationError, match="cannot be its own parent"):
        ContextSnapshot(id=snapshot.id, session_id=SESSION_ID, parent_snapshot_id=snapshot.id)


def test_message_cannot_be_active_and_archived():
    message = ids.new_id(ids.MESSAGE)
    with pytest.raises(ValidationError, match="cannot be both active and archived"):
        ContextSnapshot(
            session_id=SESSION_ID,
            active_message_ids=[message],
            archived_message_ids=[message],
        )


def test_duplicate_message_ids_rejected():
    message = ids.new_id(ids.MESSAGE)
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        ContextSnapshot(session_id=SESSION_ID, active_message_ids=[message, message])


def test_state_ids_must_be_state_prefixes():
    with pytest.raises(ValidationError, match="expected one of"):
        ContextSnapshot(session_id=SESSION_ID, state_item_ids=[ids.new_id(ids.MESSAGE)])
    snapshot = ContextSnapshot(session_id=SESSION_ID, state_item_ids=[ids.new_id(ids.DECISION)])
    assert len(snapshot.state_item_ids) == 1


def test_token_estimate_carries_its_method():
    """A count is never stored without how it was produced. A heuristic and a
    provider tokeniser disagree, and a budget spent on the wrong one overflows."""
    estimate = TokenEstimate(value=8000, method="chars/4")
    assert estimate.exact is False
    exact = TokenEstimate(value=8000, method="anthropic:claude", exact=True)
    assert exact.exact is True


def test_token_estimate_requires_a_method():
    with pytest.raises(ValidationError):
        TokenEstimate(value=100, method="")


def test_negative_token_count_rejected():
    with pytest.raises(ValidationError):
        TokenEstimate(value=-1, method="chars/4")
