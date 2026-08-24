"""Folding a new extraction into existing state.

The properties that matter are conservation ones: nothing is deleted, a re-run
does not double the state, and an ambiguous change is refused rather than
guessed at.
"""

from open_context.extraction.result import ExtractedItem
from open_context.models.enums import StateStatus
from open_context.models.state import Constraint, Decision, Fact, Goal
from open_context.state import (
    ConflictKind,
    apply_supersession,
    fingerprint,
    reconcile,
)

SESSION = "ses_" + "a" * 24


def entry(item, records=("m0",), seqs=(0,)):
    return ExtractedItem(item=item, archive_record_ids=records, archive_seqs=seqs)


def goal(content="ship the thing", **kwargs):
    return Goal(session_id=SESSION, content=content, **kwargs)


def decision(content="use SQLite", why="no server", **kwargs):
    return Decision(session_id=SESSION, content=content, rationale=why, **kwargs)


# ----------------------------------------------------------------------
# Identity


def test_the_same_item_extracted_twice_is_not_added_twice():
    """Re-extracting a session must not double its state."""
    existing = [goal()]
    plan = reconcile(existing, [entry(goal())])
    assert plan.added == []
    assert len(plan.unchanged) == 1


def test_identity_ignores_whitespace_and_case():
    plan = reconcile([goal("Ship The Thing")], [entry(goal("ship  the   thing"))])
    assert plan.unchanged and not plan.added


def test_identity_ignores_the_id():
    """Ids are minted per extraction; matching on them would make every re-run new."""
    first, second = goal(), goal()
    assert first.id != second.id
    assert fingerprint(first) == fingerprint(second)


def test_a_different_type_with_the_same_words_is_a_different_item():
    plan = reconcile(
        [goal("cache the read path")],
        [entry(Constraint(session_id=SESSION, content="cache the read path"))],
    )
    assert len(plan.added) == 1


def test_genuinely_new_content_is_added():
    plan = reconcile([goal()], [entry(goal("something else entirely"))])
    assert len(plan.added) == 1


# ----------------------------------------------------------------------
# Supersession


def test_a_supersession_claim_resolves_to_the_stored_item():
    old = decision("use PostgreSQL", why="matches the stack")
    new = decision("use SQLite", why="no server available")
    new = new.model_copy(update={"metadata": {"supersedes_content": "use PostgreSQL"}})

    plan = reconcile([old], [entry(new)])
    assert plan.supersedes == [(old.id, plan.supersedes[0][1])]
    assert plan.supersedes[0][1].item.content == "use SQLite"


def test_applying_a_supersession_keeps_both_records():
    """The record that a decision changed is part of the state."""
    old = decision("use PostgreSQL", why="matches the stack")
    new = decision("use SQLite", why="no server available")
    retired, successor = apply_supersession(old, new)

    assert retired.status is StateStatus.SUPERSEDED
    assert retired.content == "use PostgreSQL"
    assert isinstance(retired, Decision)
    assert retired.rationale == "matches the stack", "a retired decision keeps its reason"
    assert successor.supersedes == old.id
    assert successor.status is StateStatus.ACTIVE


def test_superseding_something_absent_is_a_conflict_and_the_item_is_still_kept():
    """Both failure modes are worth avoiding: silently marking nothing, or losing
    the fact that the model saw a reversal."""
    new = decision("use SQLite").model_copy(
        update={"metadata": {"supersedes_content": "use a thing nobody recorded"}}
    )
    plan = reconcile([], [entry(new)])

    assert [c.kind for c in plan.conflicts] == [ConflictKind.SUPERSEDES_MISSING]
    assert len(plan.added) == 1, "the item is kept, not dropped"


def test_a_predecessor_arriving_in_the_same_extraction_is_not_missing():
    """A reversal found entirely inside one new range is coherent."""
    old = decision("use PostgreSQL", why="matches the stack")
    new = decision("use SQLite", why="no server").model_copy(
        update={"metadata": {"supersedes_content": "use PostgreSQL"}}
    )
    plan = reconcile([], [entry(old), entry(new)])
    assert not [c for c in plan.conflicts if c.kind is ConflictKind.SUPERSEDES_MISSING]


def test_two_items_superseding_the_same_predecessor_is_a_blocking_conflict():
    """One decision cannot be overturned by two successors without a choice."""
    old = decision("use PostgreSQL", why="matches the stack")
    first = decision("use SQLite", why="a").model_copy(
        update={"metadata": {"supersedes_content": "use PostgreSQL"}}
    )
    second = decision("use DuckDB", why="b").model_copy(
        update={"metadata": {"supersedes_content": "use PostgreSQL"}}
    )
    plan = reconcile([old], [entry(first), entry(second)])

    forks = [c for c in plan.conflicts if c.kind is ConflictKind.SUPERSESSION_FORK]
    assert forks
    assert not plan.safe


def test_an_item_claiming_a_change_is_never_filed_as_unchanged():
    """Filing it as a duplicate would discard the change it asserts."""
    stored = decision("use SQLite", why="no server")
    incoming = decision("use SQLite", why="no server").model_copy(
        update={"metadata": {"supersedes_content": "use PostgreSQL"}}
    )
    plan = reconcile([stored], [entry(incoming)])
    assert plan.unchanged == []


# ----------------------------------------------------------------------
# Conflicts and safety


def test_a_clean_update_is_safe():
    plan = reconcile([goal()], [entry(goal("a new goal"))])
    assert plan.safe
    assert plan.blocked == []


def test_contradictory_facts_about_one_subject_block():
    existing = [Fact(session_id=SESSION, content="the port is 8080", subject="metrics port")]
    incoming = entry(Fact(session_id=SESSION, content="the port is 8082", subject="metrics port"))
    plan = reconcile(existing, [incoming])

    kinds = [c.kind for c in plan.conflicts]
    assert ConflictKind.CONTRADICTORY_FACT in kinds
    assert not plan.safe


def test_facts_without_a_subject_are_not_compared():
    """Comparing arbitrary sentences is the language question this refuses."""
    existing = [Fact(session_id=SESSION, content="the port is 8080")]
    plan = reconcile(existing, [entry(Fact(session_id=SESSION, content="the port is 8082"))])
    assert not [c for c in plan.conflicts if c.kind is ConflictKind.CONTRADICTORY_FACT]


def test_a_duplicate_is_the_only_auto_resolvable_conflict():
    """Everything else is a choice somebody has to make."""
    existing = [
        Fact(session_id=SESSION, content="the port is 8080", subject="p"),
        Fact(session_id=SESSION, content="the port is 8082", subject="p"),
    ]
    plan = reconcile(existing, [])
    for conflict in plan.conflicts:
        if conflict.kind is not ConflictKind.DUPLICATE_ACTIVE:
            assert not conflict.auto_resolvable


def test_a_superseded_item_does_not_conflict_with_the_current_one():
    """That is the record of a change, which supersession exists to preserve."""
    old = Fact(
        session_id=SESSION,
        content="the port is 8080",
        subject="metrics port",
        status=StateStatus.SUPERSEDED,
    )
    new = Fact(session_id=SESSION, content="the port is 8082", subject="metrics port")
    plan = reconcile([old, new], [])
    assert not [c for c in plan.conflicts if c.kind is ConflictKind.CONTRADICTORY_FACT]


def test_reconcile_changes_nothing():
    """It returns a plan; an update carrying a conflict must not be half-applied."""
    existing = [goal()]
    before = [item.model_dump() for item in existing]
    reconcile(existing, [entry(goal("new"))])
    assert [item.model_dump() for item in existing] == before


def test_the_description_reports_what_would_happen():
    plan = reconcile([goal()], [entry(goal()), entry(goal("new"))])
    assert "1 new" in plan.describe()
    assert "1 unchanged" in plan.describe()
