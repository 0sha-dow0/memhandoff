"""Structured extraction: parsing, rejecting, and refusing to invent.

Extraction is the first layer that interprets, so most of what matters is what
it declines to do with a model's output. Offline throughout, against the
deterministic fakes.
"""

import json

import pytest

from open_context.archive import Archive
from open_context.extraction import (
    EXTRACTION_PROMPT_V1,
    ExtractionConfig,
    StructuredExtractor,
    parse_json_object,
)
from open_context.importers import EventType, RawEvent, import_events
from open_context.llm.fakes import FakeProvider
from open_context.models.enums import StateStatus, StateType

pytestmark = pytest.mark.integration


def session(tmp_path, texts=("We must not use Redis.", "Agreed, no Redis.")):
    archive = Archive(tmp_path / "archive")
    log = archive.create("ses_" + "a" * 24)
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


def reply(items):
    return json.dumps({"items": items})


def item(**kwargs):
    base = {
        "type": "constraint",
        "content": "Redis must not be used",
        "source_ids": ["m0"],
        "status": "active",
        "confidence": 0.9,
    }
    return {**base, **kwargs}


def extract(tmp_path, text, **kwargs):
    log = session(tmp_path, **kwargs)
    provider = FakeProvider(reply=text, context_window=100_000)
    return StructuredExtractor(provider).extract(log), log


# ----------------------------------------------------------------------
# The happy path


def test_a_well_formed_item_becomes_a_record(tmp_path):
    result, _ = extract(tmp_path, reply([item()]))
    assert result.report.items_accepted == 1
    entry = result.items[0]
    assert entry.item.type is StateType.CONSTRAINT
    assert entry.item.content == "Redis must not be used"
    assert entry.archive_record_ids == ("m0",)


def test_provenance_records_where_it_came_from(tmp_path):
    """The citation must resolve to something a reader can open."""
    result, _ = extract(tmp_path, reply([item(source_ids=["m0", "m1"])]))
    entry = result.items[0]
    assert entry.archive_record_ids == ("m0", "m1")
    assert entry.archive_seqs == (0, 1)


def test_the_result_records_the_range_it_read(tmp_path):
    result, _ = extract(tmp_path, reply([item()]))
    assert (result.first_seq, result.last_seq) == (0, 1)


def test_the_result_records_the_prompt_that_produced_it(tmp_path):
    result, _ = extract(tmp_path, reply([item()]))
    assert result.prompt_id == EXTRACTION_PROMPT_V1.identifier
    assert result.prompt_hash == EXTRACTION_PROMPT_V1.content_hash


def test_every_record_is_scoped_to_the_session(tmp_path):
    result, log = extract(tmp_path, reply([item()]))
    assert result.items[0].item.session_id == log.session_id


# ----------------------------------------------------------------------
# What it refuses


def test_an_item_citing_nothing_is_rejected(tmp_path):
    """Provenance is the point; an unattributed claim is not state."""
    result, _ = extract(tmp_path, reply([item(source_ids=[])]))
    assert result.items == ()
    assert result.report.unattributed == 1
    assert not result.report.clean


def test_an_item_citing_a_record_from_another_session_is_rejected(tmp_path):
    result, _ = extract(tmp_path, reply([item(source_ids=["not-in-this-session"])]))
    assert result.items == ()
    assert result.report.unattributed == 1


def test_a_decision_without_a_rationale_is_rejected(tmp_path):
    """A decision recorded without its reason cannot be revisited later."""
    result, _ = extract(
        tmp_path, reply([item(type="decision", content="Use SQLite", rationale="")])
    )
    assert result.items == ()
    assert any("rationale" in r.reason for r in result.report.rejected)


def test_a_decision_with_a_rationale_is_kept(tmp_path):
    result, _ = extract(
        tmp_path,
        reply([item(type="decision", content="Use SQLite", rationale="no server available")]),
    )
    assert result.items[0].item.rationale == "no server available"


def test_an_unknown_type_is_counted_not_guessed(tmp_path):
    result, _ = extract(tmp_path, reply([item(type="vibe")]))
    assert result.items == ()
    assert result.report.unknown_types == {"vibe": 1}


def test_a_type_the_model_knows_but_extraction_does_not_emit_is_refused(tmp_path):
    """``preference`` is in the data model and deliberately not extracted."""
    result, _ = extract(tmp_path, reply([item(type="preference")]))
    assert result.items == ()


def test_an_item_without_content_is_rejected(tmp_path):
    result, _ = extract(tmp_path, reply([item(content="  ")]))
    assert result.items == ()


def test_a_non_object_item_is_rejected(tmp_path):
    result, _ = extract(tmp_path, reply(["just a string"]))
    assert result.items == ()
    assert result.report.items_emitted == 1


# ----------------------------------------------------------------------
# Malformed replies


def test_a_reply_that_is_not_json_is_a_recorded_failure(tmp_path):
    """Never a silent empty extraction, which reads as 'nothing to find'."""
    result, _ = extract(tmp_path, "I could not do that.")
    assert result.items == ()
    assert result.report.malformed_response
    assert "not usable JSON" in result.report.describe()


def test_json_wrapped_in_prose_is_still_read(tmp_path):
    """Models fence and preamble however firmly they are asked not to."""
    text = "Here is the state:\n```json\n" + reply([item()]) + "\n```\nHope that helps."
    result, _ = extract(tmp_path, text)
    assert result.report.items_accepted == 1


def test_truncated_json_is_not_repaired(tmp_path):
    """Guessing what a half-written object meant is how state gets invented."""
    result, _ = extract(tmp_path, '{"items": [{"type": "goal", "content": "half')
    assert result.report.malformed_response


def test_parse_json_object_returns_none_rather_than_raising():
    assert parse_json_object("no braces here") is None
    assert parse_json_object("{not json}") is None
    assert parse_json_object("[1, 2, 3]") is None
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_an_empty_session_calls_no_model(tmp_path):
    archive = Archive(tmp_path / "archive")
    log = archive.create("ses_" + "b" * 24)
    provider = FakeProvider(reply="unused")
    result = StructuredExtractor(provider).extract(log)
    assert result.items == ()
    assert provider.requests == []
    assert result.report.llm_calls == 0


# ----------------------------------------------------------------------
# Field handling


def test_status_falls_back_to_active_rather_than_failing(tmp_path):
    result, _ = extract(tmp_path, reply([item(status="whatever")]))
    assert result.items[0].item.status is StateStatus.ACTIVE


def test_a_superseded_item_keeps_its_status(tmp_path):
    result, _ = extract(tmp_path, reply([item(status="superseded")]))
    assert result.items[0].item.status is StateStatus.SUPERSEDED


def test_supersedes_is_carried_by_content_not_by_id(tmp_path):
    """Ids are minted here, after the model answered, so it could not cite one."""
    result, _ = extract(tmp_path, reply([item(supersedes_content="use Redis")]))
    assert result.items[0].item.metadata["supersedes_content"] == "use Redis"


def test_a_boolean_confidence_does_not_become_certainty(tmp_path):
    """``bool`` subclasses ``int``, so True would arrive as the most confident value."""
    result, _ = extract(tmp_path, reply([item(confidence=True)]))
    assert result.items[0].item.confidence == 0.5


def test_an_unparseable_confidence_does_not_discard_the_item(tmp_path):
    result, _ = extract(tmp_path, reply([item(confidence="high")]))
    assert result.items[0].item.confidence == 0.5


def test_confidence_is_clamped_into_range(tmp_path):
    result, _ = extract(tmp_path, reply([item(confidence=5)]))
    assert result.items[0].item.confidence == 1.0


def test_a_hard_constraint_is_the_default(tmp_path):
    result, _ = extract(tmp_path, reply([item()]))
    assert result.items[0].item.hard is True


# ----------------------------------------------------------------------
# The prompt and the request


def test_the_model_is_shown_the_record_ids_it_must_cite(tmp_path):
    """It cannot cite what it was never shown."""
    log = session(tmp_path)
    provider = FakeProvider(reply=reply([item()]), context_window=100_000)
    StructuredExtractor(provider).extract(log)
    sent = provider.requests[0].messages[-1].content
    assert "[m0]" in sent
    assert "[m1]" in sent


def test_extraction_asks_at_temperature_zero(tmp_path):
    log = session(tmp_path)
    provider = FakeProvider(reply=reply([item()]), context_window=100_000)
    StructuredExtractor(provider).extract(log)
    assert provider.requests[0].temperature == 0.0


def test_the_prompt_demands_provenance_and_negative_constraints():
    text = EXTRACTION_PROMPT_V1.text.lower()
    assert "source_ids" in text
    assert "must not" in text, "the prompt has to name negative constraints explicitly"


def test_the_configuration_is_respected(tmp_path):
    log = session(tmp_path)
    provider = FakeProvider(reply=reply([item()]), context_window=100_000)
    config = ExtractionConfig(max_output_tokens=99)
    StructuredExtractor(provider, config).extract(log)
    assert provider.requests[0].max_output_tokens == 99


def test_the_archive_is_not_modified(tmp_path):
    log = session(tmp_path)
    before = [r.hash for r in log.range(0, None)]
    provider = FakeProvider(reply=reply([item()]), context_window=100_000)
    StructuredExtractor(provider).extract(log)
    assert [r.hash for r in log.range(0, None)] == before


def test_an_entity_is_extracted_rather_than_silently_rejected(tmp_path):
    """`Entity` requires a name the prompt does not ask for.

    Without the extractor supplying one, every extracted entity failed model
    validation and was counted as a rejection — silently, since a rejection is
    an ordinary counted outcome rather than an error. Found by Phase 8, which
    was the first code to construct an `Entity` directly.
    """
    result, _ = extract(
        tmp_path, reply([item(type="entity", content="the billing-worker service")])
    )
    assert result.report.items_accepted == 1
    assert result.items[0].item.name


def test_an_explicit_entity_name_is_preferred_over_the_content(tmp_path):
    result, _ = extract(
        tmp_path,
        reply([item(type="entity", content="the billing-worker service", name="billing-worker")]),
    )
    assert result.items[0].item.name == "billing-worker"
