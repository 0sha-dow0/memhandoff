"""The session state graph is session-local.

Knowledge meant to outlive one session belongs in the memory layer, which is a
later phase. It does not belong in a reference reaching sideways out of one
session's history into another's, because that makes a session unreadable
without loading whatever else it happens to point at.
"""

import pytest

from open_context.models import (
    Artifact,
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
)
from open_context.storage import CrossSessionReferenceError

pytestmark = pytest.mark.integration


@pytest.fixture
def other(repo):
    return repo.add_session(Session(title="the other session"))


def test_message_parent_must_be_in_the_same_session(repo, session, other):
    parent = Message(session_id=session.id, seq=0, role=Role.USER, content="A")
    repo.add_message(parent)
    with pytest.raises(CrossSessionReferenceError) as info:
        repo.add_message(
            Message(session_id=other.id, seq=0, role=Role.USER, content="B", parent_id=parent.id)
        )
    assert info.value.expected_session == other.id
    assert info.value.found_session == session.id


def test_cross_session_parent_in_a_batch_is_rejected(repo, session, other):
    """The batch path resolves pending ids, and must not lose the session check."""
    parent = Message(session_id=session.id, seq=0, role=Role.USER, content="A")
    child = Message(session_id=other.id, seq=0, role=Role.USER, content="B", parent_id=parent.id)
    with pytest.raises(CrossSessionReferenceError):
        repo.add_messages([parent, child])
    assert repo.counts()["messages"] == 0


def test_evidence_cannot_cite_another_sessions_message(repo, session, other):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="x")
    repo.add_message(message)
    with pytest.raises(CrossSessionReferenceError):
        repo.add_evidence(
            Evidence(
                session_id=other.id,
                source_ids=[message.id],
                source_type=SourceType.MESSAGE,
                content="x",
            )
        )


def test_evidence_cannot_cite_another_sessions_artifact(repo, session, other):
    artifact = Artifact(session_id=session.id, content="the report")
    repo.add_state_item(artifact)
    with pytest.raises(CrossSessionReferenceError):
        repo.add_evidence(
            Evidence(
                session_id=other.id,
                source_ids=[artifact.id],
                source_type=SourceType.ARTIFACT,
                content="excerpt",
            )
        )


def test_state_provenance_must_be_session_local(repo, session, other):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="x")
    repo.add_message(message)
    with pytest.raises(CrossSessionReferenceError):
        repo.add_state_item(
            Decision(session_id=other.id, content="d", rationale="r", sources=[message.id])
        )


def test_state_provenance_via_evidence_must_be_session_local(repo, session, other):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="x")
    repo.add_message(message)
    evidence = Evidence(
        session_id=session.id,
        source_ids=[message.id],
        source_type=SourceType.MESSAGE,
        content="x",
    )
    repo.add_evidence(evidence)
    with pytest.raises(CrossSessionReferenceError):
        repo.add_state_item(
            Decision(session_id=other.id, content="d", rationale="r", sources=[evidence.id])
        )


def test_supersedes_must_be_session_local(repo, session, other):
    january = Fact(session_id=session.id, content="PostgreSQL", subject="database")
    repo.add_state_item(january)
    with pytest.raises(CrossSessionReferenceError):
        repo.add_state_item(
            Fact(session_id=other.id, content="MongoDB", subject="database", supersedes=january.id)
        )


def test_related_ids_must_be_session_local(repo, session, other):
    goal = Goal(session_id=session.id, content="ship")
    repo.add_state_item(goal)
    with pytest.raises(CrossSessionReferenceError):
        repo.add_state_item(Goal(session_id=other.id, content="also ship", related_ids=[goal.id]))


def test_task_blockers_must_be_session_local(repo, session, other):
    blocker = Task(session_id=session.id, content="blocker")
    repo.add_state_item(blocker)
    with pytest.raises(CrossSessionReferenceError):
        repo.add_state_item(
            Task(
                session_id=other.id,
                content="ship",
                task_status=TaskStatus.BLOCKED,
                blocked_by=[blocker.id],
            )
        )


def test_snapshot_references_must_be_session_local(repo, session, other):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="x")
    repo.add_message(message)
    item = Fact(session_id=session.id, content="f")
    repo.add_state_item(item)
    parent = ContextSnapshot(session_id=session.id)
    repo.add_snapshot(parent)

    with pytest.raises(CrossSessionReferenceError):
        repo.add_snapshot(ContextSnapshot(session_id=other.id, active_message_ids=[message.id]))
    with pytest.raises(CrossSessionReferenceError):
        repo.add_snapshot(ContextSnapshot(session_id=other.id, archived_message_ids=[message.id]))
    with pytest.raises(CrossSessionReferenceError):
        repo.add_snapshot(ContextSnapshot(session_id=other.id, state_item_ids=[item.id]))
    with pytest.raises(CrossSessionReferenceError):
        repo.add_snapshot(ContextSnapshot(session_id=other.id, parent_snapshot_id=parent.id))


def test_a_rejected_cross_session_write_leaves_nothing_behind(repo, session, other):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="x")
    repo.add_message(message)
    good = Goal(session_id=other.id, content="fine")
    bad = Decision(session_id=other.id, content="d", rationale="r", sources=[message.id])
    with pytest.raises(CrossSessionReferenceError):
        repo.add_state_items([good, bad])
    assert repo.counts()["state_items"] == 0


def test_same_session_references_are_unaffected(repo, session):
    """The guard must not break the normal case."""
    parent = Message(session_id=session.id, seq=0, role=Role.USER, content="A")
    child = Message(
        session_id=session.id, seq=1, role=Role.ASSISTANT, content="B", parent_id=parent.id
    )
    repo.add_messages([child, parent])
    assert repo.get_message(child.id).parent_id == parent.id
