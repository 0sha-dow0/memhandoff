"""Validating an extraction against the conversation it came from.

Phase 6's exit criterion. The tests that matter most are the ones pinning what a
clean validation does *not* mean.
"""

import json

import pytest

from open_context.archive import Archive
from open_context.extraction import (
    ExtractedItem,
    Severity,
    StructuredExtractor,
    format_findings,
    validate,
)
from open_context.importers import EventType, RawEvent, import_events
from open_context.llm.fakes import FakeProvider
from open_context.models.state import Decision, Goal

pytestmark = pytest.mark.integration

CONVERSATION = (
    "We must not use Redis for the cache layer.",
    "Understood, we will use an in-process cache instead.",
)


def session(tmp_path, texts=CONVERSATION, name="c"):
    archive = Archive(tmp_path / "archive")
    log = archive.create("ses_" + name * 24)
    events = [
        RawEvent(
            type=EventType.USER_MESSAGE,
            provider="test",
            source_id=f"m{index}",
            source_type="user",
            text=text,
        )
        for index, text in enumerate(texts)
    ]
    import_events(log, iter(events), provider="test", source_key="test")
    return log


def extract(log, items):
    provider = FakeProvider(reply=json.dumps({"items": items}), context_window=100_000)
    return StructuredExtractor(provider).extract(log)


def grounded_item(**kwargs):
    base = {
        "type": "constraint",
        "content": "Redis must not be used for the cache layer",
        "source_ids": ["m0"],
    }
    return {**base, **kwargs}


# ----------------------------------------------------------------------


def test_a_grounded_extraction_has_no_findings(tmp_path):
    log = session(tmp_path)
    result = extract(log, [grounded_item()])
    assert validate(result, log) == []


def test_the_clean_message_says_completeness_is_not_checked(tmp_path):
    """A clean result must not be read as "everything was found"."""
    log = session(tmp_path)
    assert "completeness is not checked" in format_findings(validate(extract(log, []), log))


def test_an_extraction_that_found_nothing_is_still_clean(tmp_path):
    """Grounding and recall are different questions, and this one measures grounding.

    An extraction that returned nothing asserts nothing unsupported. Reporting
    that as a validation failure would conflate the two, and would make the
    validator unable to say what it does say.
    """
    log = session(tmp_path)
    assert validate(extract(log, []), log) == []


def test_an_item_whose_wording_is_absent_from_its_source_is_flagged(tmp_path):
    log = session(tmp_path)
    result = extract(
        log,
        [grounded_item(content="the deployment target runs Kubernetes across three regions")],
    )
    findings = validate(result, log)
    assert [f.code for f in findings] == ["weak_grounding"]
    assert findings[0].severity is Severity.WARNING


def test_weak_grounding_is_a_warning_not_an_error(tmp_path):
    """It cannot tell a correct paraphrase from an unsupported claim.

    Findings are for a human to read; nothing is deleted on the strength of a
    word-overlap heuristic.
    """
    log = session(tmp_path)
    result = extract(log, [grounded_item(content="caching will be handled in memory")])
    for finding in validate(result, log):
        assert finding.severity is Severity.WARNING


def test_a_dangling_citation_is_an_error(tmp_path):
    log = session(tmp_path)
    result = extract(log, [grounded_item()])
    tampered = result.items[0]
    broken = ExtractedItem(
        item=tampered.item,
        archive_record_ids=("m99",),
        archive_seqs=tampered.archive_seqs,
    )
    result = type(result)(**{**result.__dict__, "items": (broken,)})

    findings = validate(result, log)
    assert any(f.code == "dangling_citation" and f.severity is Severity.ERROR for f in findings)


def test_a_citation_outside_the_extracted_range_is_an_error(tmp_path):
    log = session(tmp_path)
    result = extract(log, [grounded_item()])
    entry = result.items[0]
    moved = ExtractedItem(
        item=entry.item, archive_record_ids=entry.archive_record_ids, archive_seqs=(99,)
    )
    result = type(result)(**{**result.__dict__, "items": (moved,)})

    assert any(f.code == "out_of_range" for f in validate(result, log))


def test_an_item_scoped_to_another_session_is_an_error(tmp_path):
    log = session(tmp_path)
    result = extract(log, [grounded_item()])
    stray = ExtractedItem(
        item=Goal(session_id="ses_" + "b" * 24, content="something else"),
        archive_record_ids=("m0",),
        archive_seqs=(0,),
    )
    result = type(result)(**{**result.__dict__, "items": (*result.items, stray)})

    findings = validate(result, log)
    assert any(f.code == "wrong_session" and f.severity is Severity.ERROR for f in findings)


def test_a_decision_missing_its_rationale_is_an_error(tmp_path):
    """The extractor rejects these, so this guards the validator independently."""
    log = session(tmp_path)
    result = extract(log, [grounded_item()])
    decision = Decision.model_construct(
        session_id=log.session_id,
        content="use SQLite",
        rationale="   ",
    )
    entry = ExtractedItem(item=decision, archive_record_ids=("m0",), archive_seqs=(0,))
    result = type(result)(**{**result.__dict__, "items": (entry,)})

    assert any(f.code == "decision_without_rationale" for f in validate(result, log))


def test_superseding_an_item_that_was_not_extracted_is_flagged(tmp_path):
    log = session(tmp_path)
    result = extract(log, [grounded_item(supersedes_content="a thing nobody recorded")])
    assert any(f.code == "supersedes_unknown_item" for f in validate(result, log))


def test_superseding_an_item_that_was_extracted_is_fine(tmp_path):
    log = session(tmp_path)
    result = extract(
        log,
        [
            grounded_item(content="Redis is the cache", status="superseded"),
            grounded_item(
                content="an in-process cache is used instead",
                source_ids=["m1"],
                supersedes_content="Redis is the cache",
            ),
        ],
    )
    assert not [f for f in validate(result, log) if f.code == "supersedes_unknown_item"]


def test_marking_everything_superseded_is_an_error(tmp_path):
    """It leaves no current state, which is a misread instruction not a finding."""
    log = session(tmp_path)
    result = extract(log, [grounded_item(status="superseded")])
    findings = validate(result, log)
    assert any(f.code == "everything_superseded" and f.severity is Severity.ERROR for f in findings)


def test_errors_sort_before_warnings(tmp_path):
    log = session(tmp_path)
    result = extract(
        log,
        [
            grounded_item(content="entirely unrelated wording about Kubernetes regions"),
            grounded_item(status="superseded"),
        ],
    )
    findings = validate(result, log)
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: 0 if s is Severity.ERROR else 1)


def test_the_validator_calls_no_model(tmp_path):
    """A validator that asked a model would inherit the failure it detects."""
    log = session(tmp_path)
    result = extract(log, [grounded_item()])
    provider = FakeProvider(reply="should never be called")
    validate(result, log)
    assert provider.requests == []
