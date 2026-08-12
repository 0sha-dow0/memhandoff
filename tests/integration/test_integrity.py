"""Referential integrity, the thing Phase 1 could not enforce."""

import pytest

from open_context.models import (
    ContextSnapshot,
    Decision,
    Evidence,
    Fact,
    Goal,
    Message,
    Role,
    Session,
    SourceType,
    Task,
    TaskStatus,
    ids,
)
from open_context.storage import (
    DanglingReferenceError,
    DuplicateRecordError,
    RecordNotFoundError,
    SequenceConflictError,
)

pytestmark = pytest.mark.integration


def test_message_needs_an_existing_session(repo):
    orphan = Message(session_id=Session().id, seq=0, role=Role.USER, content="x")
    with pytest.raises(DanglingReferenceError) as info:
        repo.add_message(orphan)
    assert info.value.reference == orphan.session_id


def test_message_parent_must_exist(repo, session):
    with pytest.raises(DanglingReferenceError):
        repo.add_message(
            Message(
                session_id=session.id,
                seq=0,
                role=Role.USER,
                content="x",
                parent_id=ids.new_id(ids.MESSAGE),
            )
        )


def test_parent_may_appear_later_in_the_same_batch(repo, session):
    """Imports arrive in file order, which is not dependency order."""
    parent = Message(session_id=session.id, seq=0, role=Role.USER, content="parent")
    child = Message(
        session_id=session.id, seq=1, role=Role.ASSISTANT, content="child", parent_id=parent.id
    )
    assert repo.add_messages([child, parent]) == 2
    assert repo.get_message(child.id).parent_id == parent.id


def test_evidence_source_must_exist(repo, session):
    with pytest.raises(DanglingReferenceError):
        repo.add_evidence(
            Evidence(
                session_id=session.id,
                source_ids=[ids.new_id(ids.MESSAGE)],
                source_type=SourceType.MESSAGE,
                content="x",
            )
        )


def test_evidence_may_cite_a_state_item(repo, session):
    """Evidence accepts artifact and event references, which are state items.

    That crosses tables, so the reference is routed by prefix.
    """
    from open_context.models import Artifact

    artifact = Artifact(session_id=session.id, content="the report", path="/tmp/r.md")
    repo.add_state_item(artifact)
    evidence = Evidence(
        session_id=session.id,
        source_ids=[artifact.id],
        source_type=SourceType.ARTIFACT,
        content="excerpt",
    )
    repo.add_evidence(evidence)
    assert repo.get_evidence(evidence.id).source_ids == [artifact.id]


def test_state_provenance_must_resolve(repo, session):
    with pytest.raises(DanglingReferenceError):
        repo.add_state_item(
            Fact(session_id=session.id, content="x", sources=[ids.new_id(ids.EVIDENCE)])
        )


def test_state_supersedes_must_resolve(repo, session):
    with pytest.raises(DanglingReferenceError):
        repo.add_state_item(
            Fact(session_id=session.id, content="x", supersedes=ids.new_id(ids.FACT))
        )


def test_state_related_must_resolve(repo, session):
    with pytest.raises(DanglingReferenceError):
        repo.add_state_item(
            Goal(session_id=session.id, content="x", related_ids=[ids.new_id(ids.DECISION)])
        )


def test_task_blocker_must_resolve(repo, session):
    with pytest.raises(DanglingReferenceError):
        repo.add_state_item(
            Task(
                session_id=session.id,
                content="ship",
                task_status=TaskStatus.BLOCKED,
                blocked_by=[ids.new_id(ids.TASK)],
            )
        )


def test_snapshot_references_must_resolve(repo, session):
    with pytest.raises(DanglingReferenceError):
        repo.add_snapshot(
            ContextSnapshot(session_id=session.id, active_message_ids=[ids.new_id(ids.MESSAGE)])
        )
    with pytest.raises(DanglingReferenceError):
        repo.add_snapshot(
            ContextSnapshot(session_id=session.id, state_item_ids=[ids.new_id(ids.DECISION)])
        )
    with pytest.raises(DanglingReferenceError):
        repo.add_snapshot(
            ContextSnapshot(session_id=session.id, parent_snapshot_id=ids.new_id(ids.SNAPSHOT))
        )


def test_duplicate_id_is_rejected_not_overwritten(repo, session):
    """Records are immutable, so writing over one is never the intent."""
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="first")
    repo.add_message(message)
    clash = Message(id=message.id, session_id=session.id, seq=1, role=Role.USER, content="second")
    with pytest.raises(DuplicateRecordError):
        repo.add_message(clash)
    assert repo.get_message(message.id).content == "first"


def test_seq_is_unique_within_a_session(repo, session):
    repo.add_message(Message(session_id=session.id, seq=0, role=Role.USER, content="a"))
    with pytest.raises(SequenceConflictError):
        repo.add_message(Message(session_id=session.id, seq=0, role=Role.USER, content="b"))


def test_same_seq_in_different_sessions_is_fine(repo, session):
    other = repo.add_session(Session())
    repo.add_message(Message(session_id=session.id, seq=0, role=Role.USER, content="a"))
    repo.add_message(Message(session_id=other.id, seq=0, role=Role.USER, content="b"))
    assert repo.counts()["messages"] == 2


def test_missing_record_raises_not_found(repo):
    with pytest.raises(RecordNotFoundError):
        repo.get_message(ids.new_id(ids.MESSAGE))
    with pytest.raises(RecordNotFoundError):
        repo.get_state_item(ids.new_id(ids.DECISION))


def test_evidence_and_state_may_reference_each_other(repo, session):
    """Evidence cites a state item and a state item cites evidence, so no static
    insert order works. Deferred foreign keys make the cycle storable."""
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="x")
    repo.add_message(message)
    evidence = Evidence(
        session_id=session.id,
        source_ids=[message.id],
        source_type=SourceType.MESSAGE,
        content="x",
    )
    repo.add_evidence(evidence)
    decision = Decision(session_id=session.id, content="d", rationale="r", sources=[evidence.id])
    repo.add_state_item(decision)
    assert repo.get_state_item(decision.id).sources == [evidence.id]
