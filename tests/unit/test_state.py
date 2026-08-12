"""The discriminated state item union."""

import pytest
from pydantic import ValidationError

from open_context.models import (
    Constraint,
    Decision,
    Fact,
    Goal,
    RetentionClass,
    Session,
    StateItemAdapter,
    StateStatus,
    StateType,
    Task,
    TaskStatus,
    ids,
)

SESSION_ID = Session().id


def test_decision_requires_rationale():
    """The point of the union. A decision without its reason is a claim, and the
    reason is what people come back to a long session to find."""
    with pytest.raises(ValidationError):
        Decision(session_id=SESSION_ID, content="Use RocksDB")


def test_decision_with_rationale_accepted():
    decision = Decision(
        session_id=SESSION_ID,
        content="Use RocksDB",
        rationale="Local persistent key-value workload",
        alternatives=["SQLite", "LMDB"],
    )
    assert decision.type is StateType.DECISION
    assert decision.id.startswith("dec_")


def test_rejected_decision_is_a_decision_with_status():
    """There is no separate rejected_decision type. Keeping the rationale and
    alternatives attached to the decision they explain is the whole reason."""
    rejected = Decision(
        session_id=SESSION_ID,
        content="Use Kafka for the event log",
        rationale="Operational cost is not justified at this scale",
        status=StateStatus.REJECTED,
    )
    assert rejected.status is StateStatus.REJECTED
    assert rejected.rationale


def test_adapter_resolves_type_from_json():
    original = Decision(session_id=SESSION_ID, content="Use SQLite", rationale="Local first")
    restored = StateItemAdapter.validate_json(original.model_dump_json())
    assert isinstance(restored, Decision)
    assert restored == original


def test_adapter_rejects_unknown_type():
    with pytest.raises(ValidationError):
        StateItemAdapter.validate_python({"type": "vibe", "session_id": SESSION_ID, "content": "x"})


def test_adapter_rejects_decision_missing_rationale():
    """A malformed extraction is caught at parse time, not at compile time."""
    with pytest.raises(ValidationError):
        StateItemAdapter.validate_python(
            {"type": "decision", "session_id": SESSION_ID, "content": "Use RocksDB"}
        )


def test_each_type_gets_its_own_id_prefix():
    goal = Goal(session_id=SESSION_ID, content="Ship the compiler")
    fact = Fact(session_id=SESSION_ID, content="Storage is SQLite")
    assert goal.id.startswith("goal_")
    assert fact.id.startswith("fact_")


def test_critical_item_requires_a_source():
    with pytest.raises(ValidationError, match="requires at least one source"):
        Constraint(
            session_id=SESSION_ID,
            content="Must run offline",
            retention=RetentionClass.CRITICAL,
        )


def test_critical_item_with_source_accepted():
    constraint = Constraint(
        session_id=SESSION_ID,
        content="Must run offline",
        retention=RetentionClass.CRITICAL,
        sources=[ids.new_id(ids.MESSAGE)],
    )
    assert constraint.hard is True


def test_sources_accept_messages_and_evidence_only():
    Fact(
        session_id=SESSION_ID,
        content="x",
        sources=[ids.new_id(ids.MESSAGE), ids.new_id(ids.EVIDENCE)],
    )
    with pytest.raises(ValidationError, match="expected one of"):
        Fact(session_id=SESSION_ID, content="x", sources=[ids.new_id(ids.SNAPSHOT)])


def test_item_cannot_supersede_itself():
    item = Goal(session_id=SESSION_ID, content="x")
    with pytest.raises(ValidationError, match="cannot supersede itself"):
        Goal(id=item.id, session_id=SESSION_ID, content="x", supersedes=item.id)


def test_item_cannot_relate_to_itself():
    item = Goal(session_id=SESSION_ID, content="x")
    with pytest.raises(ValidationError, match="cannot be related to itself"):
        Goal(id=item.id, session_id=SESSION_ID, content="x", related_ids=[item.id])


def test_supersession_keeps_the_original_reachable():
    """A changing fact produces a second record pointing at the first. Both survive."""
    january = Fact(session_id=SESSION_ID, content="Database is PostgreSQL", subject="database")
    march = Fact(
        session_id=SESSION_ID,
        content="Database is MongoDB",
        subject="database",
        supersedes=january.id,
    )
    assert march.supersedes == january.id
    assert january.status is StateStatus.ACTIVE


def test_blocked_task_must_record_what_blocks_it():
    with pytest.raises(ValidationError, match="must record what blocks it"):
        Task(session_id=SESSION_ID, content="Ship it", task_status=TaskStatus.BLOCKED)


def test_blocked_task_with_blocker_accepted():
    blocker = Task(session_id=SESSION_ID, content="Fix the importer")
    task = Task(
        session_id=SESSION_ID,
        content="Ship it",
        task_status=TaskStatus.BLOCKED,
        blocked_by=[blocker.id],
    )
    assert task.blocked_by == [blocker.id]


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_importance_and_confidence_bounded(value):
    with pytest.raises(ValidationError):
        Goal(session_id=SESSION_ID, content="x", importance=value)
    with pytest.raises(ValidationError):
        Goal(session_id=SESSION_ID, content="x", confidence=value)


def test_empty_content_rejected():
    with pytest.raises(ValidationError):
        Goal(session_id=SESSION_ID, content="")


def test_union_covers_every_state_type():
    """Guards against a type being added to the enum but left out of the union.

    Without this, an extractor could emit a valid StateType that the adapter
    cannot parse, and the failure would surface as a dropped state item several
    phases later.
    """
    from typing import get_args

    from open_context.models.state import StateItem

    members = get_args(get_args(StateItem)[0])
    covered = {member.model_fields["type"].default for member in members}
    assert covered == set(StateType)


def test_every_state_type_round_trips_through_the_adapter():
    from typing import get_args

    from open_context.models.state import StateItem

    required = {
        "decision": {"rationale": "because"},
        "entity": {"name": "RocksDB"},
    }
    for member in get_args(get_args(StateItem)[0]):
        kind = member.model_fields["type"].default
        item = member(session_id=SESSION_ID, content="x", **required.get(kind, {}))
        assert StateItemAdapter.validate_json(item.model_dump_json()) == item
