"""Hybrid compaction: state, recent, and history under one budget.

The phase makes a narrow falsifiable bet — that structured state carries exact
values prose drops — so the tests pin the mechanism that would make that true,
and the honesty of the cost accounting that says what it charged for it.
"""

import json

import pytest

from open_context.archive import Archive
from open_context.compaction import CompactionRequest
from open_context.hybrid import (
    STATE_ORDER,
    HybridCompaction,
    HybridConfig,
    fit_state,
    render_state,
)
from open_context.importers import EventType, RawEvent, import_events
from open_context.llm.fakes import FakeProvider, WordTokenizer
from open_context.models.enums import StateStatus, StateType
from open_context.models.state import Constraint, Decision, Entity, Fact, Goal

pytestmark = pytest.mark.integration

SESSION = "ses_" + "a" * 24


def session(tmp_path, texts):
    archive = Archive(tmp_path / "archive")
    log = archive.create(SESSION)
    events = [
        RawEvent(
            type=EventType.USER_MESSAGE,
            provider="t",
            source_id=f"m{i}",
            source_type="user",
            text=text,
        )
        for i, text in enumerate(texts)
    ]
    import_events(log, iter(events), provider="t", source_key="k")
    return log


def extraction_reply(items):
    return json.dumps({"items": items})


def item(**kwargs):
    base = {"type": "fact", "content": "the metrics port is 8082", "source_ids": ["m0"]}
    return {**base, **kwargs}


def goal(content="ship it"):
    return Goal(session_id=SESSION, content=content)


def fact(content="the port is 8082", **kwargs):
    return Fact(session_id=SESSION, content=content, **kwargs)


# ----------------------------------------------------------------------
# Rendering


def test_state_is_grouped_under_headings():
    text = render_state([goal(), Constraint(session_id=SESSION, content="no Redis")])
    assert "GOALS" in text and "CONSTRAINTS" in text
    assert "ship it" in text and "no Redis" in text


def test_a_decision_carries_its_rationale():
    """The reason is what people reopen a long session to recover."""
    text = render_state(
        [Decision(session_id=SESSION, content="use SQLite", rationale="no server available")]
    )
    assert "no server available" in text


def test_superseded_items_are_not_rendered():
    """Both sides of a reversal, without saying which won, is worse than neither."""
    text = render_state(
        [
            fact("the port is 8080", status=StateStatus.SUPERSEDED),
            fact("the port is 8082"),
        ]
    )
    assert "8082" in text
    assert "8080" not in text


def test_empty_state_renders_to_nothing():
    assert render_state([]) == ""
    assert render_state([goal().model_copy(update={"status": StateStatus.SUPERSEDED})]) == ""


def test_headings_follow_the_priority_order():
    text = render_state(
        [
            Entity(session_id=SESSION, content="billing-worker", name="billing-worker"),
            Constraint(session_id=SESSION, content="no Redis"),
            goal(),
        ]
    )
    assert text.index("GOALS") < text.index("CONSTRAINTS") < text.index("ENTITIES")


# ----------------------------------------------------------------------
# Fitting to a budget


def test_everything_fits_when_the_budget_allows():
    items = [goal(), fact()]
    text, kept = fit_state(items, WordTokenizer(exact=True), budget=1000)
    assert kept == 2
    assert text


def test_a_tight_budget_drops_from_the_end_of_the_priority_order():
    """Entities go before constraints, not the other way round."""
    items = [
        goal("ship the thing"),
        Constraint(session_id=SESSION, content="do not use Redis at all"),
        Entity(
            session_id=SESSION, content="billing-worker service component", name="billing-worker"
        ),
    ]
    text, kept = fit_state(items, WordTokenizer(exact=True), budget=12)
    assert kept < 3
    assert "billing-worker" not in text


def test_a_budget_of_zero_keeps_nothing():
    text, kept = fit_state([goal()], WordTokenizer(exact=True), budget=0)
    assert (text, kept) == ("", 0)


def test_fitting_never_cuts_mid_sentence():
    """A truncated constraint can read as its own opposite."""
    items = [Constraint(session_id=SESSION, content="do not delete the production database")]
    text, _ = fit_state(items, WordTokenizer(exact=True), budget=3)
    assert text == "" or "do not delete the production database" in text


def test_facts_outrank_decisions_and_tasks():
    """Exact values live in facts, and exact-value loss is the measured failure.

    The first ordering put facts second from last, on the reasoning that a fact
    without its supporting decision is untrustworthy. Against a real model that
    dropped the port numbers from `confusable-numbers` while keeping a goal, a
    formatting decision, and a task — losing exactly what the phase exists to
    preserve.
    """
    assert STATE_ORDER.index(StateType.FACT) < STATE_ORDER.index(StateType.DECISION)
    assert STATE_ORDER.index(StateType.FACT) < STATE_ORDER.index(StateType.TASK)
    assert STATE_ORDER.index(StateType.CONSTRAINT) < STATE_ORDER.index(StateType.FACT)


def test_the_priority_order_covers_every_extracted_type():
    """A type missing from the order would raise when it appeared."""
    extracted = {
        StateType.GOAL,
        StateType.CONSTRAINT,
        StateType.FACT,
        StateType.DECISION,
        StateType.TASK,
        StateType.OPEN_QUESTION,
        StateType.ENTITY,
    }
    assert extracted == set(STATE_ORDER)


# ----------------------------------------------------------------------
# Compaction


def compactor(reply, **kwargs):
    provider = FakeProvider(reply=reply, context_window=100_000)
    return HybridCompaction(provider, WordTokenizer(exact=True), **kwargs), provider


def test_the_result_carries_a_state_section(tmp_path):
    log = session(tmp_path, ["the metrics port is 8082", "and the api port is 8080"] * 6)
    hybrid, _ = compactor(extraction_reply([item()]))
    result = hybrid.compact(CompactionRequest(log=log, target_tokens=200))

    assert "8082" in result.state_section
    assert result.strategy == "hybrid_v1"


def test_the_state_section_appears_in_the_rendered_output(tmp_path):
    log = session(tmp_path, ["the metrics port is 8082"] * 8)
    hybrid, _ = compactor(extraction_reply([item()]))
    result = hybrid.compact(CompactionRequest(log=log, target_tokens=200))
    assert result.state_section in result.rendered()


def test_the_extraction_call_is_added_to_the_bill(tmp_path):
    """A hybrid result reporting only the summarization call would understate
    its cost by exactly the thing this phase added."""
    log = session(tmp_path, ["a message about the cache"] * 20)
    hybrid, provider = compactor(extraction_reply([item()]))
    result = hybrid.compact(CompactionRequest(log=log, target_tokens=60))

    assert result.model.llm_calls == len(provider.requests)
    assert result.model.llm_calls >= 2, "extraction plus summarization"


def test_the_configuration_records_what_was_extracted_and_kept(tmp_path):
    log = session(tmp_path, ["the metrics port is 8082"] * 8)
    hybrid, _ = compactor(extraction_reply([item(), item(content="a second fact", type="goal")]))
    result = hybrid.compact(CompactionRequest(log=log, target_tokens=200))

    assert result.configuration["state_items_extracted"] == 2
    assert result.configuration["state_fraction"] == 0.35
    assert result.configuration["extraction_prompt_id"]


def test_a_failed_extraction_says_so_rather_than_looking_like_hybrid(tmp_path):
    """Silently degrading to the baseline would make a benchmark row a lie."""
    log = session(tmp_path, ["a message"] * 8)
    hybrid, _ = compactor("not json at all")
    result = hybrid.compact(CompactionRequest(log=log, target_tokens=200))

    assert result.state_section == ""
    assert any("should not be read as hybrid" in w for w in result.warnings)


def test_dropped_state_is_warned_about(tmp_path):
    log = session(tmp_path, ["the metrics port is 8082"] * 8)
    many = [item(content=f"fact number {n} with several words in it") for n in range(12)]
    hybrid, _ = compactor(extraction_reply(many))
    result = hybrid.compact(CompactionRequest(log=log, target_tokens=40))

    assert any("were dropped" in w for w in result.warnings)


def test_the_state_budget_is_a_fraction_of_the_target(tmp_path):
    log = session(tmp_path, ["a message"] * 8)
    hybrid, _ = compactor(extraction_reply([item()]))
    result = hybrid.compact(CompactionRequest(log=log, target_tokens=200))
    assert result.allocated_state_tokens == 70


def test_an_invalid_state_fraction_is_rejected():
    for value in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="state_fraction"):
            HybridConfig(state_fraction=value)


def test_the_archive_is_not_modified(tmp_path):
    log = session(tmp_path, ["the metrics port is 8082"] * 8)
    before = [r.hash for r in log.range(0, None)]
    hybrid, _ = compactor(extraction_reply([item()]))
    hybrid.compact(CompactionRequest(log=log, target_tokens=200))
    assert [r.hash for r in log.range(0, None)] == before
