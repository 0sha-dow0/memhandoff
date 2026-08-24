"""The `.ctx` package: generate, validate, inspect.

Offline throughout. The format is the product artifact, so what these pin is
less "the code runs" than "the package cannot quietly claim more than it knows" —
provenance it does not have, an archive it does not describe, or a version it
cannot read.
"""

import json

import pytest

from open_context.archive.records import ArchiveRecord
from open_context.models import ids
from open_context.models.enums import StateStatus
from open_context.models.state import Constraint, Decision, Fact, Goal
from open_context.package import (
    FORMAT,
    FORMAT_VERSION,
    CtxPackage,
    Depth,
    Severity,
    build,
    render,
    summarize,
    validate,
)

SESSION = "ses_" + "a" * 24
MSG_A = "msg_" + "1" * 24
MSG_B = "msg_" + "2" * 24


def goal(session: str = SESSION, content: str = "Ship the export writer") -> Goal:
    return Goal(id=ids.new_id(ids.GOAL), session_id=session, content=content)


def fact(session: str = SESSION, content: str = "The metrics port is 8082") -> Fact:
    return Fact(id=ids.new_id(ids.FACT), session_id=session, content=content)


def records() -> list[ArchiveRecord]:
    return [
        ArchiveRecord(seq=0, id=MSG_A, kind="message", payload={"content": "ship the writer"}),
        ArchiveRecord(seq=1, id=MSG_B, kind="message", payload={"content": "metrics is on 8082"}),
    ]


def packaged(**kwargs):
    items = kwargs.pop("state", None) or [goal(), fact()]
    return build(session_id=SESSION, state=items, **kwargs)


# ----------------------------------------------------------------------
# Generate — the first exit criterion


def test_a_package_can_be_generated():
    package, report = packaged()
    assert package.manifest.format == FORMAT
    assert package.manifest.format_version == FORMAT_VERSION
    assert len(package.state) == 2
    assert report.state_items == 2


def test_a_package_id_says_it_is_a_package():
    """Identifiers are self-describing; one saying `ses_` would be a lie."""
    package, _ = packaged()
    assert package.manifest.package_id.startswith("ctx_")


def test_a_package_always_carries_a_hash():
    package, _ = packaged()
    assert package.content_hash
    assert package.intact()


def test_the_hash_does_not_depend_on_key_order():
    """Canonical serialisation, so a package that survived a reformatting tool
    still verifies. A hash sensitive to key order would call every reserialised
    package tampered with.
    """
    package, _ = packaged(archive_records=records())
    shuffled = json.loads(json.dumps(json.loads(package.to_json()), sort_keys=False))
    assert CtxPackage.model_validate(shuffled).expected_hash() == package.expected_hash()


def test_two_builds_of_the_same_state_agree_when_the_clock_is_pinned():
    """Creation time is the one field that legitimately varies between builds.

    Pin it and the packages are byte-identical, which is what makes a
    reproducibility check possible at all.
    """
    from datetime import UTC, datetime

    when = datetime(2026, 8, 17, tzinfo=UTC)
    items = [goal(), fact()]
    a, _ = build(session_id=SESSION, state=items, package_id="ctx_fixed", created_at=when)
    b, _ = build(session_id=SESSION, state=items, package_id="ctx_fixed", created_at=when)
    assert a.body() == b.body()
    assert a.content_hash == b.content_hash


def test_two_builds_at_different_times_are_different_packages():
    """Not a defect: they were built at different times, and the format says so."""
    items = [goal(), fact()]
    a, _ = build(session_id=SESSION, state=items, package_id="ctx_fixed")
    b, _ = build(session_id=SESSION, state=items, package_id="ctx_fixed")
    assert a.manifest.created_at != b.manifest.created_at or a.content_hash == b.content_hash


def test_a_package_survives_a_round_trip_through_json():
    package, _ = packaged(archive_records=records())
    restored = CtxPackage.from_json(package.to_json())
    assert restored.content_hash == package.content_hash
    assert restored.intact()


def test_the_json_is_written_to_be_read_by_a_person():
    """Human-inspectable is a requirement of the format, not a nicety."""
    package, _ = packaged()
    text = package.to_json()
    assert "\n" in text and "  " in text, "a single-line document is not inspectable"
    assert json.loads(text)["manifest"]["format"] == FORMAT


def test_state_from_another_session_is_refused():
    """A package mixing sessions would present one agent's work as another's."""
    with pytest.raises(ValueError, match="different session"):
        build(session_id=SESSION, state=[goal(), goal(session="ses_" + "b" * 24)])


def test_superseded_items_are_excluded_and_the_exclusion_is_recorded():
    """A reader given both sides of a reversal cannot tell which won.

    The count survives so the reader knows history existed.
    """
    retired = Decision(
        id=ids.new_id(ids.DECISION),
        session_id=SESSION,
        content="Use Postgres",
        rationale="it is already deployed",
        status=StateStatus.SUPERSEDED,
    )
    package, report = build(session_id=SESSION, state=[goal(), retired])
    assert len(package.state) == 1
    assert report.superseded_excluded == 1
    assert package.manifest.metadata["superseded_excluded"] == 1


# ----------------------------------------------------------------------
# Provenance is resolved where possible and admitted where not


def test_provenance_is_resolved_against_the_archive():
    items = [goal(), fact()]
    package, report = build(
        session_id=SESSION,
        state=items,
        archive_records=records(),
        provenance={items[0].id: [0], items[1].id: [1]},
    )
    assert report.evidence_resolved == 2
    assert report.evidence_unresolved == 0
    assert report.fully_traceable
    by_state = {e.state_id: e for e in package.evidence}
    assert by_state[items[1].id].archive_record_ids == (MSG_B,)
    assert "8082" in by_state[items[1].id].excerpt


def test_provenance_resolves_through_source_ids_when_the_archive_holds_them():
    """`ArchiveRecord.id` is "identity of the underlying item, preserved as given",
    so an archive fed the same messages the state came from already holds the
    very ids `sources` names.

    Phase 6 concluded no correspondence existed between the two id schemes. That
    is true of sequence numbers and not of ids, and matching on an identifier
    both sides agreed on is a lookup rather than a guess.
    """
    item = Goal(
        id=ids.new_id(ids.GOAL),
        session_id=SESSION,
        content="Ship the export writer",
        sources=[MSG_B],
    )
    package, report = build(session_id=SESSION, state=[item], archive_records=records())
    assert report.evidence_resolved == 1
    assert package.evidence[0].archive_seqs == (1,)
    assert package.evidence[0].archive_record_ids == (MSG_B,)


def test_inferred_provenance_still_hashes_the_records_it_cites():
    """Otherwise the deep check has nothing to verify and passes vacuously.

    A package that cites a record without hashing it cannot be told apart from a
    package describing a different archive — which is the one thing the
    provenance check exists to catch.
    """
    item = Goal(id=ids.new_id(ids.GOAL), session_id=SESSION, content="Ship it", sources=[MSG_B])
    package, _ = build(session_id=SESSION, state=[item], archive_records=records())
    assert package.archive is not None
    assert set(package.archive.record_hashes) == {1}

    rewritten = ArchiveRecord(seq=1, id=MSG_B, kind="message", payload={"content": "changed"})
    report = validate(package, archive_records={1: rewritten})
    assert any(f.code == "cited-record-changed" for f in report.errors)


def test_a_source_the_archive_does_not_hold_simply_does_not_resolve():
    item = Goal(
        id=ids.new_id(ids.GOAL),
        session_id=SESSION,
        content="Ship it",
        sources=["msg_" + "9" * 24],
    )
    _, report = build(session_id=SESSION, state=[item], archive_records=records())
    assert report.evidence_unresolved == 1


def test_an_explicit_provenance_mapping_wins_over_the_inference():
    """A caller who tracked the derivation knows more than a lookup does."""
    item = Goal(id=ids.new_id(ids.GOAL), session_id=SESSION, content="Ship it", sources=[MSG_B])
    package, _ = build(
        session_id=SESSION, state=[item], archive_records=records(), provenance={item.id: [0]}
    )
    assert package.evidence[0].archive_seqs == (0,)


def test_an_item_with_no_provenance_is_admitted_rather_than_invented():
    """`sources` speaks the SQLite id scheme and the archive numbers its own
    records; no correspondence has ever been established between them.

    Minting a plausible archive id here would create a reference that resolves
    to nothing and looks exactly like one that resolves.
    """
    items = [goal(), fact()]
    package, report = build(
        session_id=SESSION, state=items, archive_records=records(), provenance={items[0].id: [0]}
    )
    assert report.evidence_resolved == 1
    assert report.evidence_unresolved == 1
    assert not report.fully_traceable
    assert any("no archive provenance" in w for w in report.warnings)
    unresolved = [e for e in package.evidence if not e.resolvable]
    assert [e.state_id for e in unresolved] == [items[1].id]


def test_only_cited_records_are_hashed():
    """A reference that hashed every record would grow into a copy of the history."""
    items = [goal(), fact()]
    package, _ = build(
        session_id=SESSION, state=items, archive_records=records(), provenance={items[0].id: [0]}
    )
    assert package.archive is not None
    assert package.archive.record_count == 2
    assert set(package.archive.record_hashes) == {0}


def test_the_archive_is_referenced_not_embedded():
    """Embedding would make the package a second copy that begins to diverge."""
    package, _ = packaged(archive_records=records())
    text = package.to_json()
    assert "metrics is on 8082" not in text, "a full record payload was inlined"
    assert package.archive is not None and package.archive.record_count == 2


def test_an_excerpt_is_short_enough_to_stay_an_excerpt():
    long_record = ArchiveRecord(seq=0, id=MSG_A, kind="message", payload={"content": "x " * 900})
    item = goal()
    package, _ = build(
        session_id=SESSION, state=[item], archive_records=[long_record], provenance={item.id: [0]}
    )
    assert len(package.evidence[0].excerpt) <= 240


# ----------------------------------------------------------------------
# Validate — the second exit criterion


def test_a_well_formed_package_validates():
    items = [goal(), fact()]
    package, _ = build(
        session_id=SESSION,
        state=items,
        archive_records=records(),
        provenance={items[0].id: [0], items[1].id: [1]},
    )
    report = validate(package, archive_records={r.seq: r for r in records()})
    assert report.valid, report
    assert report.depth is Depth.PROVENANCE


def test_validation_says_how_deep_it_went():
    """A structure pass says nothing about whether the package describes the
    archive it names, and a verdict that hid the difference would conceal the
    one dangerous case: internally impeccable, and about something else.
    """
    package, _ = packaged(archive_records=records())
    assert validate(package).depth is Depth.STRUCTURE
    assert validate(package, archive_records={}).depth is Depth.PROVENANCE


def test_an_altered_package_fails_its_hash():
    package, _ = packaged()
    document = json.loads(package.to_json())
    document["state"][0]["content"] = "Ship something else entirely"
    tampered = CtxPackage.model_validate(document)
    report = validate(tampered)
    assert not report.valid
    assert any(f.code == "hash-mismatch" for f in report.errors)


def test_a_package_describing_a_different_archive_is_caught():
    """The check that separates *this* archive from a similar one."""
    item = goal()
    package, _ = build(
        session_id=SESSION, state=[item], archive_records=records(), provenance={item.id: [0]}
    )
    rewritten = ArchiveRecord(seq=0, id=MSG_A, kind="message", payload={"content": "different"})
    report = validate(package, archive_records={0: rewritten})
    assert not report.valid
    assert any(f.code == "cited-record-changed" for f in report.errors)


def test_a_cited_record_the_archive_does_not_have_is_an_error():
    item = goal()
    package, _ = build(
        session_id=SESSION, state=[item], archive_records=records(), provenance={item.id: [0]}
    )
    report = validate(package, archive_records={})
    assert any(f.code == "cited-record-missing" for f in report.errors)


def test_a_package_with_no_traceable_provenance_is_usable_but_says_so():
    """Warning, not error. A package without provenance is still a package;
    a reader relying on it should simply know what it cannot prove.
    """
    package, _ = packaged()
    report = validate(package)
    assert report.valid
    assert {f.code for f in report.warnings} >= {"unresolvable-provenance", "no-archive-reference"}


def test_a_superseded_item_that_reached_the_package_is_an_error():
    """The builder excludes these; a hand-assembled package might not."""
    retired = Constraint(
        id=ids.new_id(ids.CONSTRAINT),
        session_id=SESSION,
        content="Never write to NFS",
        status=StateStatus.SUPERSEDED,
    )
    package, _ = packaged(state=[goal()])
    hand_made = package.model_copy(update={"state": (*package.state, retired), "content_hash": ""})
    hand_made = CtxPackage.model_validate(hand_made.model_dump(mode="json") | {"content_hash": ""})
    report = validate(hand_made)
    assert any(f.code == "superseded-included" for f in report.errors)


def test_evidence_for_an_item_the_package_does_not_contain_is_an_error():
    package, _ = packaged()
    document = json.loads(package.to_json())
    document["evidence"][0]["state_id"] = "goal_" + "z" * 24
    document["content_hash"] = ""
    report = validate(CtxPackage.model_validate(document))
    assert any(f.code == "orphan-evidence" for f in report.errors)


# ----------------------------------------------------------------------
# Versioning


def test_a_package_from_a_future_version_is_refused_rather_than_guessed_at():
    """A newer package may parse cleanly while a field has changed meaning.

    Reading it anyway makes that a silent failure.
    """
    package, _ = packaged()
    document = json.loads(package.to_json())
    document["manifest"]["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="Refusing to parse"):
        CtxPackage.from_json(json.dumps(document))


def test_a_json_file_that_is_not_a_package_is_refused():
    with pytest.raises(ValueError, match="not a open-context-ctx package"):
        CtxPackage.from_json('{"manifest": {"format": "something-else"}}')
    with pytest.raises(ValueError, match="must be a JSON object"):
        CtxPackage.from_json("[]")
    with pytest.raises(ValueError, match="must have a manifest"):
        CtxPackage.from_json("{}")


# ----------------------------------------------------------------------
# Inspect — the third exit criterion


def test_a_package_can_be_inspected():
    items = [goal(), fact()]
    package, _ = build(
        session_id=SESSION,
        state=items,
        archive_records=records(),
        provenance={items[0].id: [0], items[1].id: [1]},
        title="Export writer",
    )
    text = render(package)
    assert "Ship the export writer" in text
    assert "The metrics port is 8082" in text
    assert "GOAL" in text and "FACT" in text
    assert "from archive 1" in text


def test_inspection_shows_what_is_missing_as_plainly_as_what_is_there():
    """Forty confident items with no traceable source is the most misleading
    possible view of such a package.
    """
    package, _ = packaged()
    text = render(package)
    assert "no traceable source" in text
    assert "nothing here can be traced to a source" in text


def test_the_summary_reports_a_broken_hash():
    package, _ = packaged()
    document = json.loads(package.to_json())
    document["state"][0]["content"] = "tampered"
    assert "MISMATCH" in summarize(CtxPackage.model_validate(document))


def test_state_is_rendered_in_reading_order():
    """What the work is for, what limits it, what is true, what was decided."""
    constraint = Constraint(
        id=ids.new_id(ids.CONSTRAINT), session_id=SESSION, content="Never write to NFS"
    )
    package, _ = packaged(state=[fact(), constraint, goal()])
    text = render(package)
    assert text.index("GOAL") < text.index("CONSTRAINT") < text.index("FACT")


def test_severity_reads_as_its_own_value():
    assert str(Severity.ERROR) == "error"
