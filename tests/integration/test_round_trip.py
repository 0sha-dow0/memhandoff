"""Every model must come back exactly as it went in."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from open_context.models import (
    Artifact,
    Constraint,
    ContextSnapshot,
    Decision,
    Entity,
    Event,
    Evidence,
    Fact,
    Goal,
    Message,
    OpenQuestion,
    Preference,
    RetentionClass,
    Role,
    Session,
    SourceType,
    StateStatus,
    Task,
    TaskStatus,
    TokenEstimate,
    TrustLevel,
)

pytestmark = pytest.mark.integration


def test_session_round_trip(repo):
    original = Session(title="Project", source="chatgpt", metadata={"tags": ["a", "b"]})
    repo.add_session(original)
    assert repo.get_session(original.id) == original


def test_message_round_trip(repo, session):
    original = Message(
        session_id=session.id,
        seq=7,
        role=Role.TOOL,
        content="3 matches",
        trust=TrustLevel.UNTRUSTED,
        tool_name="grep",
        tool_call_id="call_1",
        metadata={"nested": {"k": 1}},
    )
    repo.add_message(original)
    assert repo.get_message(original.id) == original


def test_ids_are_preserved_exactly(repo, session):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="x")
    repo.add_message(message)
    assert repo.get_message(message.id).id == message.id
    assert repo.get_message(message.id).session_id == session.id


def test_timezone_is_normalised_and_preserved(repo):
    ist = timezone(timedelta(hours=5, minutes=30))
    original = Session(created_at=datetime(2026, 1, 1, 17, 30, tzinfo=ist))
    repo.add_session(original)
    restored = repo.get_session(original.id)
    assert restored.created_at == original.created_at
    assert restored.created_at.utcoffset() == timedelta(0)


def test_evidence_round_trip_with_multiple_sources(repo, session):
    first = Message(session_id=session.id, seq=0, role=Role.USER, content="a")
    second = Message(session_id=session.id, seq=1, role=Role.ASSISTANT, content="b")
    repo.add_messages([first, second])
    original = Evidence(
        session_id=session.id,
        source_ids=[second.id, first.id],
        source_type=SourceType.MESSAGE,
        content="the exact words",
    )
    repo.add_evidence(original)
    restored = repo.get_evidence(original.id)
    assert restored == original
    assert restored.source_ids == [second.id, first.id], "source order must survive"
    assert restored.verify()


def test_every_state_type_round_trips(repo, session):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="src")
    repo.add_message(message)
    blocker = Task(session_id=session.id, content="blocker")
    repo.add_state_item(blocker)

    items = [
        Goal(session_id=session.id, content="ship"),
        Constraint(session_id=session.id, content="offline only", hard=False),
        Fact(session_id=session.id, content="db is sqlite", subject="database"),
        Decision(
            session_id=session.id,
            content="RocksDB",
            rationale="local kv workload",
            alternatives=["LMDB", "SQLite"],
        ),
        Task(
            session_id=session.id,
            content="ship it",
            task_status=TaskStatus.BLOCKED,
            blocked_by=[blocker.id],
        ),
        OpenQuestion(session_id=session.id, content="which tokeniser?"),
        Preference(session_id=session.id, content="terse output"),
        Entity(session_id=session.id, content="the store", name="RocksDB", kind="library"),
        Event(
            session_id=session.id,
            content="migrated",
            occurred_at=datetime(2026, 3, 1, tzinfo=UTC),
        ),
        Artifact(session_id=session.id, content="the doc", path="/tmp/a.md", media_type="text/md"),
    ]
    repo.add_state_items(items)
    for item in items:
        assert repo.get_state_item(item.id) == item, f"{type(item).__name__} did not round trip"


def test_state_item_round_trip_with_full_provenance(repo, session):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="use rocksdb")
    repo.add_message(message)
    evidence = Evidence(
        session_id=session.id,
        source_ids=[message.id],
        source_type=SourceType.MESSAGE,
        content="use rocksdb",
    )
    repo.add_evidence(evidence)
    related = Goal(session_id=session.id, content="pick a store")
    repo.add_state_item(related)

    original = Decision(
        session_id=session.id,
        content="RocksDB",
        rationale="local persistent kv",
        alternatives=["LMDB"],
        sources=[message.id, evidence.id],
        related_ids=[related.id],
        retention=RetentionClass.CRITICAL,
        importance=0.9,
        confidence=0.8,
        metadata={"extracted_by": "test"},
    )
    repo.add_state_item(original)
    restored = repo.get_state_item(original.id)
    assert restored == original
    assert restored.sources == [message.id, evidence.id], "source order must survive"


def test_snapshot_round_trip(repo, session):
    messages = [
        Message(session_id=session.id, seq=i, role=Role.USER, content=str(i)) for i in range(4)
    ]
    repo.add_messages(messages)
    item = Fact(session_id=session.id, content="a fact")
    repo.add_state_item(item)

    parent = ContextSnapshot(session_id=session.id, reason="manual")
    repo.add_snapshot(parent)
    original = ContextSnapshot(
        session_id=session.id,
        parent_snapshot_id=parent.id,
        active_message_ids=[messages[3].id, messages[2].id],
        archived_message_ids=[messages[0].id, messages[1].id],
        state_item_ids=[item.id],
        token_estimate=TokenEstimate(value=1200, method="chars/4"),
        compiler_version="v0",
        reason="threshold:0.80",
    )
    repo.add_snapshot(original)
    restored = repo.get_snapshot(original.id)
    assert restored == original
    assert restored.active_message_ids == [messages[3].id, messages[2].id]


def test_snapshot_without_token_estimate_round_trips(repo, session):
    original = ContextSnapshot(session_id=session.id)
    repo.add_snapshot(original)
    assert repo.get_snapshot(original.id).token_estimate is None


def test_status_and_enums_survive(repo, session):
    original = Decision(
        session_id=session.id,
        content="Kafka",
        rationale="not worth the ops cost",
        status=StateStatus.REJECTED,
        trust=TrustLevel.USER_AUTHORED,
    )
    repo.add_state_item(original)
    restored = repo.get_state_item(original.id)
    assert restored.status is StateStatus.REJECTED
    assert restored.trust is TrustLevel.USER_AUTHORED
