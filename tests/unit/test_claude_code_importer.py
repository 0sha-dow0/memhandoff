"""Reading Claude Code sessions.

Every fixture here mirrors a shape measured in real transcripts — 6 files,
8,890 conversation records — because the phase this belongs to says to inspect
the actual mechanism rather than assume it, and because each of these tests
corresponds to something the inspection contradicted.
"""

import io
import json

from open_context.importers import detect_importer
from open_context.importers.claude_code import (
    CONVERSATION_PARENT,
    ClaudeCodeImporter,
    abandoned,
    active_thread,
    segments,
)
from open_context.importers.events import EventType
from open_context.importers.report import MalformedRecord

SESSION = "5fd9884a-bde1-4a4c-80af-ad0befd8eaa4"


def record(uuid, parent, kind="user", content="hello", **extra):
    row = {
        "type": kind,
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION,
        "timestamp": "2026-08-16T08:36:46.316Z",
        "cwd": "/work",
        "gitBranch": "main",
        "version": "2.1.0",
        "userType": "external",
        "isSidechain": False,
        "entrypoint": "cli",
        "message": {"role": "assistant" if kind == "assistant" else "user", "content": content},
    }
    row.update(extra)
    return row


def read(rows, **kwargs):
    text = "\n".join(json.dumps(r) for r in rows)
    return list(ClaudeCodeImporter(**kwargs).read(io.StringIO(text)))


def events(rows, **kwargs):
    return [r for r in read(rows, **kwargs) if not isinstance(r, MalformedRecord)]


# ----------------------------------------------------------------------
# Detection


def test_a_session_file_is_recognised():
    sample = json.dumps(record("a", None))
    assert ClaudeCodeImporter().detect(sample)


def test_detection_does_not_depend_on_the_first_record_carrying_a_uuid():
    """One of six real files opens with a `custom-title` record that has a
    `sessionId` and no `uuid`. A check anchored on the first record's uuid
    rejected that file outright."""
    sample = "\n".join(
        [
            json.dumps({"type": "custom-title", "customTitle": "mlh", "sessionId": SESSION}),
            json.dumps(record("a", None)),
        ]
    )
    assert ClaudeCodeImporter().detect(sample)


def test_generic_json_lines_are_not_claimed():
    """`JsonLinesImporter.detect` accepts any JSON object, so this one has to be
    the narrow side of that pair or it would claim ordinary exports."""
    assert not ClaudeCodeImporter().detect('{"role": "user", "content": "hi"}')
    assert not ClaudeCodeImporter().detect('{"type": "user", "content": "hi"}')


def test_a_session_file_is_routed_to_this_importer_not_the_generic_one():
    """Order in the registry is the whole point of it: generic-first would claim
    every transcript and import it as anonymous records that look fine."""
    chosen = detect_importer(json.dumps(record("a", None)))
    assert chosen.provider == "claude-code"


# ----------------------------------------------------------------------
# Content shapes


def test_content_may_be_a_plain_string():
    """225 of 3,323 real user records carry a string rather than blocks."""
    (event,) = events([record("a", None, content="just text")])
    assert event.type is EventType.USER_MESSAGE
    assert event.text == "just text"


def test_content_may_be_a_list_of_typed_blocks():
    rows = [
        record(
            "a",
            None,
            kind="assistant",
            content=[
                {"type": "text", "text": "I will check the port"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ss -ltn"}},
            ],
        )
    ]
    text, call = events(rows)
    assert text.type is EventType.ASSISTANT_MESSAGE
    assert call.type is EventType.TOOL_CALL
    assert call.tool_name == "Bash"
    assert call.tool_call_id == "t1"


def test_one_record_can_produce_several_events():
    """A turn with text and three tool calls is four things that happened.
    Collapsing them would lose the tool calls, which Phase 8.5 measured to be
    where answers hide."""
    rows = [
        record(
            "a",
            None,
            kind="assistant",
            content=[
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "Read", "input": {}},
                {"type": "tool_use", "id": "t3", "name": "Grep", "input": {}},
            ],
        )
    ]
    assert len(events(rows)) == 4


def test_a_tool_result_keeps_its_correlation_id():
    rows = [
        record(
            "a",
            None,
            content=[{"type": "tool_result", "tool_use_id": "t1", "content": "8082 listening"}],
        )
    ]
    (event,) = events(rows)
    assert event.type is EventType.TOOL_RESULT
    assert event.tool_call_id == "t1"
    assert "8082" in (event.text or "")


def test_a_tool_result_holding_blocks_is_flattened_not_serialised():
    """Serialising the structure would fill the archive with braces a model then
    has to parse back out."""
    rows = [
        record(
            "a",
            None,
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "port 8082"}],
                }
            ],
        )
    ]
    (event,) = events(rows)
    assert event.text == "port 8082"


# ----------------------------------------------------------------------
# Thinking


def test_thinking_is_skipped_by_default():
    """This project has already stored deliberation as a conclusion once.
    `strip_inline_reasoning` exists because of it."""
    rows = [
        record(
            "a",
            None,
            kind="assistant",
            content=[
                {"type": "thinking", "thinking": "maybe the port is 8080"},
                {"type": "text", "text": "The port is 8082."},
            ],
        )
    ]
    (event,) = events(rows)
    assert event.text == "The port is 8082."


def test_thinking_can_be_asked_for():
    rows = [
        record(
            "a", None, kind="assistant", content=[{"type": "thinking", "thinking": "deliberating"}]
        )
    ]
    (event,) = events(rows, include_thinking=True)
    assert event.text == "deliberating"


def test_a_turn_that_was_only_thinking_still_produces_an_event():
    """Dropping it would leave a hole the parent chain points into."""
    rows = [
        record("a", None, kind="assistant", content=[{"type": "thinking", "thinking": "hmm"}]),
    ]
    (event,) = events(rows)
    assert event.text == ""
    assert event.metadata["skipped_thinking_blocks"] == 1


# ----------------------------------------------------------------------
# The parent chain runs through records this importer does not import


def test_the_conversation_parent_skips_attachments():
    """Measured: 804 attachment and 62 system records sit between turns in one
    real session. Indexing only imported records breaks the chain at the first
    attachment."""
    rows = [
        record("a", None, content="first"),
        {"type": "attachment", "uuid": "att", "parentUuid": "a", "sessionId": SESSION},
        record("b", "att", content="second"),
    ]
    first, second = events(rows)
    assert second.parent_id == "att", "the provider's own reference stays verbatim"
    assert second.metadata[CONVERSATION_PARENT] == "a"
    assert first.metadata.get(CONVERSATION_PARENT) is None


def test_bookkeeping_records_are_skipped():
    rows = [
        {"type": "ai-title", "aiTitle": "x", "sessionId": SESSION},
        {"type": "permission-mode", "permissionMode": "default", "sessionId": SESSION},
        record("a", None),
    ]
    assert len(events(rows)) == 1


def test_meta_records_are_skipped():
    """Injected by the tool rather than said by anyone."""
    rows = [record("a", None, isMeta=True), record("b", None)]
    assert len(events(rows)) == 1


# ----------------------------------------------------------------------
# Forks are rewinds; second roots are continuations


def test_the_abandoned_side_of_a_rewind_is_dropped():
    rows = [
        record("a", None, content="original question"),
        record("b", "a", kind="assistant", content="first answer"),
        record("c", "a", kind="assistant", content="retried answer"),
    ]
    live = active_thread(events(rows))
    texts = [e.text for e in live]
    assert "retried answer" in texts
    assert "first answer" not in texts
    assert [e.text for e in abandoned(events(rows))] == ["first answer"]


def test_a_second_root_is_kept_because_it_is_a_continuation():
    """The correction that mattered most. Claude Code starts a *new tree* when it
    compacts, so following only the chain ending at the last record discarded
    every pre-compaction turn — 78% and 86% of two real sessions, and exactly the
    history a handoff exists to carry. One of those files has no forks at all.
    """
    rows = [
        record("a", None, content="early work"),
        record("b", "a", kind="assistant", content="early answer"),
        record("c", None, content="compacted summary", isCompactSummary=True),
        record("d", "c", kind="assistant", content="continued work"),
    ]
    live = active_thread(events(rows))
    assert [e.text for e in live] == [
        "early work",
        "early answer",
        "compacted summary",
        "continued work",
    ]
    assert abandoned(events(rows)) == []


def test_a_fork_inside_a_later_segment_is_still_resolved():
    rows = [
        record("a", None, content="early"),
        record("b", None, content="after compaction", isCompactSummary=True),
        record("c", "b", kind="assistant", content="wrong"),
        record("d", "b", kind="assistant", content="right"),
    ]
    texts = [e.text for e in active_thread(events(rows))]
    assert texts == ["early", "after compaction", "right"]


def test_segments_are_consecutive_and_rejoin_into_the_conversation():
    rows = [
        record("a", None, content="early"),
        record("b", "a", kind="assistant", content="early answer"),
        record("c", None, content="after compaction", isCompactSummary=True),
    ]
    parts = segments(events(rows))
    assert [len(p) for p in parts] == [2, 1]
    assert [e for part in parts for e in part] == active_thread(events(rows))


def test_an_empty_conversation_is_not_an_error():
    assert active_thread([]) == []
    assert segments([]) == []


# ----------------------------------------------------------------------
# Damage


def test_a_broken_line_is_located_rather_than_fatal():
    text = json.dumps(record("a", None)) + "\n{not json\n" + json.dumps(record("b", "a"))
    results = list(ClaudeCodeImporter().read(io.StringIO(text)))
    bad = [r for r in results if isinstance(r, MalformedRecord)]
    assert len(bad) == 1
    assert bad[0].location == "line 2"
    assert len([r for r in results if not isinstance(r, MalformedRecord)]) == 2


def test_a_record_without_a_message_is_reported():
    rows = [{"type": "user", "uuid": "a", "parentUuid": None, "sessionId": SESSION}]
    (bad,) = read(rows)
    assert isinstance(bad, MalformedRecord)
    assert bad.reason == "no_message"


def test_detection_survives_a_first_record_larger_than_the_sample():
    """The detection sample is a fixed number of bytes and a single record can
    be large — 1.36 MB in one measured transcript. Reading a truncated line as
    "not this format" handed every such file to the generic reader, which
    accepts anything shaped like JSON and would import it as anonymous records.
    """
    huge = json.dumps(record("a", None, content="x" * 20_000))
    sample = huge[:8192]
    assert not ClaudeCodeImporter().detect(sample), "a lone truncated line says nothing"

    two = huge + "\n" + json.dumps(record("b", "a"))
    assert ClaudeCodeImporter().detect(two[:30_000])
