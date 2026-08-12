"""Durability across connections, and query behaviour."""

import pytest

from open_context.models import (
    Constraint,
    Decision,
    Evidence,
    Fact,
    Goal,
    Message,
    RetentionClass,
    Role,
    Session,
    SourceType,
    StateStatus,
)
from open_context.storage import SCHEMA_VERSION, Database, Repository, SchemaVersionError

pytestmark = pytest.mark.integration


def test_data_survives_reopening_the_file(tmp_path):
    path = tmp_path / "nested" / "context.db"
    session = Session(title="persisted")

    with Database(path) as db:
        repo = Repository(db)
        repo.add_session(session)
        message = Message(session_id=session.id, seq=0, role=Role.USER, content="remember me")
        repo.add_message(message)
        decision = Decision(
            session_id=session.id,
            content="RocksDB",
            rationale="local kv",
            sources=[message.id],
            retention=RetentionClass.CRITICAL,
        )
        repo.add_state_item(decision)

    with Database(path) as db:
        repo = Repository(db)
        assert repo.get_session(session.id).title == "persisted"
        assert repo.get_state_item(decision.id) == decision
        assert repo.counts() == {
            "sessions": 1,
            "messages": 1,
            "evidence": 0,
            "state_items": 1,
            "snapshots": 0,
        }


def test_parent_directory_is_created(tmp_path):
    path = tmp_path / "a" / "b" / "c.db"
    with Database(path):
        pass
    assert path.exists()


def test_opening_without_migrating_requires_the_current_version(tmp_path):
    path = tmp_path / "context.db"
    with pytest.raises(SchemaVersionError):
        Database(path, migrate_on_open=False)
    with Database(path):
        pass
    with Database(path, migrate_on_open=False) as db:
        assert db.schema_version == SCHEMA_VERSION


def test_messages_are_ordered_by_seq_not_insertion(repo, session):
    for seq in (4, 0, 2, 1, 3):
        repo.add_message(Message(session_id=session.id, seq=seq, role=Role.USER, content=str(seq)))
    assert [m.content for m in repo.list_messages(session.id)] == ["0", "1", "2", "3", "4"]


def test_messages_sharing_a_timestamp_still_order_correctly(repo, session):
    stamp = session.created_at
    for seq in (2, 0, 1):
        repo.add_message(
            Message(
                session_id=session.id,
                seq=seq,
                role=Role.USER,
                content=str(seq),
                timestamp=stamp,
            )
        )
    assert [m.seq for m in repo.list_messages(session.id)] == [0, 1, 2]


def test_next_seq_reports_the_free_position(repo, session):
    assert repo.next_seq(session.id) == 0
    repo.add_message(Message(session_id=session.id, seq=0, role=Role.USER, content="a"))
    repo.add_message(Message(session_id=session.id, seq=5, role=Role.USER, content="b"))
    assert repo.next_seq(session.id) == 6


def test_sessions_are_isolated(repo, session):
    other = repo.add_session(Session(title="other"))
    repo.add_message(Message(session_id=session.id, seq=0, role=Role.USER, content="mine"))
    repo.add_message(Message(session_id=other.id, seq=0, role=Role.USER, content="theirs"))
    assert [m.content for m in repo.list_messages(session.id)] == ["mine"]
    assert [m.content for m in repo.list_messages(other.id)] == ["theirs"]


def test_list_state_items_filters_by_type_and_status(repo, session):
    goal = Goal(session_id=session.id, content="ship")
    constraint = Constraint(session_id=session.id, content="offline")
    rejected = Decision(
        session_id=session.id, content="Kafka", rationale="cost", status=StateStatus.REJECTED
    )
    repo.add_state_items([goal, constraint, rejected])

    assert [i.id for i in repo.list_state_items(session.id, types=["goal"])] == [goal.id]
    assert len(repo.list_state_items(session.id, types=["goal", "constraint"])) == 2
    assert repo.list_state_items(session.id, types=[]) == []
    assert [i.id for i in repo.list_state_items(session.id, status=StateStatus.REJECTED)] == [
        rejected.id
    ]


def test_evidence_lookup_by_hash_finds_duplicates(repo, session):
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="x")
    repo.add_message(message)
    first = Evidence(
        session_id=session.id,
        source_ids=[message.id],
        source_type=SourceType.MESSAGE,
        content="identical text",
    )
    second = Evidence(
        session_id=session.id,
        source_ids=[message.id],
        source_type=SourceType.MESSAGE,
        content="identical text",
    )
    repo.add_evidence(first)
    repo.add_evidence(second)
    found = repo.find_evidence_by_hash(first.content_hash)
    assert {e.id for e in found} == {first.id, second.id}


def test_empty_batches_are_no_ops(repo):
    assert repo.add_messages([]) == 0
    assert repo.add_state_items([]) == 0


def test_superseded_by_returns_the_replacement(repo, session):
    old = Fact(session_id=session.id, content="old")
    repo.add_state_item(old)
    assert repo.superseded_by(old.id) is None
    new = Fact(session_id=session.id, content="new", supersedes=old.id)
    repo.add_state_item(new)
    assert repo.superseded_by(old.id).id == new.id
