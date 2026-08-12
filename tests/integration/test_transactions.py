"""Atomicity and the immutability contract."""

import pytest

from open_context.models import (
    Decision,
    Fact,
    Goal,
    Message,
    Role,
    StateStatus,
    ids,
)
from open_context.storage import DanglingReferenceError, SerializationError

pytestmark = pytest.mark.integration


def test_a_failed_batch_writes_nothing(repo, session):
    """Partial writes are the failure mode that corrupts an archive quietly."""
    good = Message(session_id=session.id, seq=0, role=Role.USER, content="good")
    bad = Message(
        session_id=session.id,
        seq=1,
        role=Role.USER,
        content="bad",
        parent_id=ids.new_id(ids.MESSAGE),
    )
    with pytest.raises(DanglingReferenceError):
        repo.add_messages([good, bad])
    assert repo.counts()["messages"] == 0
    assert repo.list_messages(session.id) == []


def test_a_failed_state_batch_leaves_no_junction_rows(repo, session):
    """The junction tables must roll back with the parent row."""
    good = Goal(session_id=session.id, content="ok")
    bad = Fact(session_id=session.id, content="bad", sources=[ids.new_id(ids.MESSAGE)])
    with pytest.raises(DanglingReferenceError):
        repo.add_state_items([good, bad])
    assert repo.counts()["state_items"] == 0
    leftover = repo.database.connection.execute("SELECT COUNT(*) AS n FROM state_sources")
    assert leftover.fetchone()["n"] == 0


def test_unserialisable_metadata_fails_cleanly_and_writes_nothing(repo, session):
    """metadata is typed dict[str, Any], so a set passes model validation and
    only fails at write time. It must fail with a useful message, not a partial
    write and a bare TypeError."""
    message = Message(
        session_id=session.id, seq=0, role=Role.USER, content="x", metadata={"bad": {1, 2}}
    )
    with pytest.raises(SerializationError, match="metadata"):
        repo.add_message(message)
    assert repo.counts()["messages"] == 0


def test_repository_exposes_no_update_or_delete(repo):
    """Immutability is enforced by the absence of the operation, not by a check
    inside one."""
    surface = {name for name in dir(repo) if not name.startswith("_")}
    forbidden = {"update", "delete", "remove", "save", "upsert", "set_status", "merge"}
    assert not (surface & forbidden)


def test_supersession_does_not_touch_the_original(repo, session):
    """A changing fact writes a new record. The old row is untouched, including
    its status."""
    january = Fact(session_id=session.id, content="Database is PostgreSQL", subject="database")
    repo.add_state_item(january)
    march = Fact(
        session_id=session.id,
        content="Database is MongoDB",
        subject="database",
        supersedes=january.id,
    )
    repo.add_state_item(march)

    stored = repo.get_state_item(january.id)
    assert stored == january
    assert stored.status is StateStatus.ACTIVE, "the historical row must not be rewritten"
    assert repo.is_superseded(january.id)
    assert not repo.is_superseded(march.id)


def test_supersession_chain_resolves_current_state(repo, session):
    """January PostgreSQL, March MongoDB, June PostgreSQL. All three survive and
    the current one is derived from the edges."""
    january = Fact(session_id=session.id, content="PostgreSQL", subject="database")
    repo.add_state_item(january)
    march = Fact(
        session_id=session.id, content="MongoDB", subject="database", supersedes=january.id
    )
    repo.add_state_item(march)
    june = Fact(
        session_id=session.id, content="PostgreSQL", subject="database", supersedes=march.id
    )
    repo.add_state_item(june)

    chain = repo.supersession_chain(january.id)
    assert [item.content for item in chain] == ["PostgreSQL", "MongoDB", "PostgreSQL"]
    current = repo.current_state_items(session.id)
    assert [item.id for item in current] == [june.id]
    assert repo.counts()["state_items"] == 3, "history is kept, not overwritten"


def test_supersession_chain_terminates_on_a_cycle(repo, session):
    first = Fact(session_id=session.id, content="a")
    second = Fact(session_id=session.id, content="b", supersedes=first.id)
    repo.add_state_items([first, second])
    repo.database.connection.execute(
        "UPDATE state_items SET supersedes = ? WHERE id = ?", (second.id, first.id)
    )
    assert len(repo.supersession_chain(first.id)) == 2


def test_rejected_items_are_kept_but_not_current(repo, session):
    rejected = Decision(
        session_id=session.id,
        content="Kafka",
        rationale="ops cost not justified",
        status=StateStatus.REJECTED,
    )
    active = Decision(session_id=session.id, content="SQLite", rationale="local first")
    repo.add_state_items([rejected, active])

    current_ids = {item.id for item in repo.current_state_items(session.id)}
    assert current_ids == {active.id}
    assert repo.get_state_item(rejected.id) == rejected, "why it was rejected stays answerable"


def test_snapshot_lineage_is_oldest_first(repo, session):
    from open_context.models import ContextSnapshot

    first = ContextSnapshot(session_id=session.id, reason="one")
    repo.add_snapshot(first)
    second = ContextSnapshot(session_id=session.id, parent_snapshot_id=first.id, reason="two")
    repo.add_snapshot(second)
    third = ContextSnapshot(session_id=session.id, parent_snapshot_id=second.id, reason="three")
    repo.add_snapshot(third)

    assert [s.reason for s in repo.snapshot_lineage(third.id)] == ["one", "two", "three"]
    assert repo.latest_snapshot(session.id).id == third.id


def test_supersession_cycle_is_rejected(repo, session):
    """A batch can contain items that supersede each other. Supersession says
    which record replaced which, so a loop leaves no current state to resolve
    to."""
    from open_context.storage import SupersessionCycleError

    first = Fact(session_id=session.id, content="a")
    second = Fact(session_id=session.id, content="b", supersedes=first.id)
    looping_first = Fact(id=first.id, session_id=session.id, content="a", supersedes=second.id)
    with pytest.raises(SupersessionCycleError):
        repo.add_state_items([looping_first, second])
    assert repo.counts()["state_items"] == 0


def test_a_normal_supersession_chain_is_still_accepted(repo, session):
    first = Fact(session_id=session.id, content="a")
    second = Fact(session_id=session.id, content="b", supersedes=first.id)
    third = Fact(session_id=session.id, content="c", supersedes=second.id)
    repo.add_state_items([first, second, third])
    assert len(repo.supersession_chain(first.id)) == 3
