"""The Evidence to State provenance invariant.

The intended layering is:

    raw source  ->  evidence  ->  state

A message is what was said. Evidence is an immutable excerpt of it, hashed. A
state item is an interpretation of that evidence and cites it.

No schema change was needed for this. The property that keeps the graph clean
already follows from how writes work: a provenance reference must resolve to a
record that is already stored, so provenance can only ever point backwards in
time. That makes the graph acyclic by construction, and the unbounded
State -> Evidence -> State -> Evidence chain cannot be built.

These tests pin that invariant so a later change cannot quietly remove it.
"""

import pytest

from open_context.models import (
    Artifact,
    Decision,
    Evidence,
    Message,
    Role,
    SourceType,
)
from open_context.storage import DanglingReferenceError

pytestmark = pytest.mark.integration


def test_the_canonical_chain(repo, session):
    """message -> evidence -> decision, and the decision cites the evidence."""
    message = Message(
        session_id=session.id, seq=183, role=Role.USER, content="We should use RocksDB."
    )
    repo.add_message(message)

    evidence = Evidence(
        session_id=session.id,
        source_ids=[message.id],
        source_type=SourceType.MESSAGE,
        content="We should use RocksDB.",
    )
    repo.add_evidence(evidence)

    decision = Decision(
        session_id=session.id,
        content="Use RocksDB",
        rationale="Local persistent key-value workload",
        sources=[evidence.id],
    )
    repo.add_state_item(decision)

    stored = repo.get_state_item(decision.id)
    assert stored.sources == [evidence.id]
    supporting = repo.get_evidence(stored.sources[0])
    assert supporting.source_ids == [message.id]
    assert supporting.verify(), "the excerpt still hashes to its stored content"
    assert repo.get_message(supporting.source_ids[0]).content == message.content


def test_state_may_cite_a_message_directly(repo, session):
    """Evidence is not mandatory. A state item can point straight at the message
    when there is no excerpt worth materialising."""
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="Use SQLite.")
    repo.add_message(message)
    decision = Decision(
        session_id=session.id, content="SQLite", rationale="local first", sources=[message.id]
    )
    repo.add_state_item(decision)
    assert repo.get_state_item(decision.id).sources == [message.id]


def test_provenance_can_only_point_at_already_stored_records(repo, session):
    """The property the whole invariant rests on.

    Unlike supersedes and related_ids, a provenance reference is not satisfied
    by a record appearing later in the same batch. It must already exist.
    """
    evidence = Evidence(
        session_id=session.id,
        source_ids=[Message(session_id=session.id, seq=0, role=Role.USER, content="x").id],
        source_type=SourceType.MESSAGE,
        content="x",
    )
    with pytest.raises(DanglingReferenceError):
        repo.add_evidence(evidence)


def test_a_provenance_cycle_cannot_be_built(repo, session):
    """Evidence citing an artifact, and that artifact citing the same evidence,
    is the shape that would produce an unbounded chain. Whichever is written
    first has nothing to point at."""
    artifact = Artifact(session_id=session.id, content="the report")
    evidence = Evidence(
        session_id=session.id,
        source_ids=[artifact.id],
        source_type=SourceType.ARTIFACT,
        content="an excerpt of the report",
    )
    circular_artifact = Artifact(
        id=artifact.id, session_id=session.id, content="the report", sources=[evidence.id]
    )

    with pytest.raises(DanglingReferenceError):
        repo.add_evidence(evidence)
    with pytest.raises(DanglingReferenceError):
        repo.add_state_item(circular_artifact)
    assert repo.counts()["evidence"] == 0
    assert repo.counts()["state_items"] == 0


def test_evidence_over_an_artifact_terminates(repo, session):
    """Evidence may cite an artifact or event state item, which is the one place
    the chain passes through state a second time. It terminates, because the
    artifact was stored first and its own provenance is fixed."""
    artifact = Artifact(session_id=session.id, content="design notes", path="/tmp/notes.md")
    repo.add_state_item(artifact)
    evidence = Evidence(
        session_id=session.id,
        source_ids=[artifact.id],
        source_type=SourceType.ARTIFACT,
        content="the relevant paragraph",
    )
    repo.add_evidence(evidence)
    decision = Decision(
        session_id=session.id,
        content="Adopt the layout in the notes",
        rationale="already agreed",
        sources=[evidence.id],
    )
    repo.add_state_item(decision)

    hops = 0
    frontier = list(repo.get_state_item(decision.id).sources)
    while frontier and hops < 10:
        reference = frontier.pop()
        hops += 1
        if reference.startswith("ev_"):
            frontier.extend(repo.get_evidence(reference).source_ids)
        elif reference.startswith(("art_", "evt_")):
            frontier.extend(repo.get_state_item(reference).sources)
    assert not frontier, "the chain terminated rather than looping"
    assert hops < 10


def test_evidence_content_is_never_rewritten_by_later_state(repo, session):
    """State is an interpretation. Adding one must not touch the material it
    interprets."""
    message = Message(session_id=session.id, seq=0, role=Role.USER, content="exact words")
    repo.add_message(message)
    evidence = Evidence(
        session_id=session.id,
        source_ids=[message.id],
        source_type=SourceType.MESSAGE,
        content="exact words",
    )
    repo.add_evidence(evidence)
    before = repo.get_evidence(evidence.id)

    repo.add_state_item(
        Decision(
            session_id=session.id,
            content="an interpretation",
            rationale="because",
            sources=[evidence.id],
        )
    )
    assert repo.get_evidence(evidence.id) == before
