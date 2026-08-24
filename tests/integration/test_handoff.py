"""Phase 11's exit criterion: a documented cross-agent handoff.

    Agent A  ->  runtime  ->  project.ctx  ->  Agent B  ->  continue the work

Run through the real CLI against a transcript in Claude Code's actual on-disk
format. The one test that touches this machine's own sessions skips when there
are none, so the suite still runs anywhere.
"""

import json

import pytest

from open_context.__main__ import EXIT_OK, main
from open_context.handoff import handoff, read_session, recent_context
from open_context.importers.claude_code import find_sessions

SESSION_UUID = "912bcf79-7320-42d7-b702-749ceed71e5a"


def row(uuid, parent, kind="user", content="hello", **extra):
    record = {
        "type": kind,
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION_UUID,
        "timestamp": "2026-08-16T08:36:46.316Z",
        "cwd": "/work",
        "gitBranch": "main",
        "version": "2.1.0",
        "userType": "external",
        "isSidechain": False,
        "entrypoint": "cli",
        "message": {"role": "assistant" if kind == "assistant" else "user", "content": content},
    }
    record.update(extra)
    return record


@pytest.fixture
def transcript(tmp_path):
    """A session with the shapes that matter: attachments between turns, a
    rewind, tool traffic, and a compaction break."""
    rows = [
        {"type": "custom-title", "customTitle": "export work", "sessionId": SESSION_UUID},
        row("u1", None, content="Ship the reporting export writer this week."),
        {"type": "attachment", "uuid": "att1", "parentUuid": "u1", "sessionId": SESSION_UUID},
        row(
            "a1",
            "att1",
            kind="assistant",
            content=[
                {"type": "thinking", "thinking": "maybe UTF-8 is fine"},
                {"type": "text", "text": "The loader rejects UTF-8, so I will write UTF-16LE."},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ss -ltn"}},
            ],
        ),
        row("r1", "a1", content=[{"type": "tool_result", "tool_use_id": "t1", "content": "8082"}]),
        row("a2", "r1", kind="assistant", content="An abandoned attempt."),
        row("a3", "r1", kind="assistant", content="Metrics listens on 8082."),
        row("u2", None, content="Continuing after compaction.", isCompactSummary=True),
    ]
    path = tmp_path / f"{SESSION_UUID}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_a_session_is_read_in_the_agents_own_format(transcript):
    events, malformed = read_session(transcript)
    assert not malformed
    assert any("UTF-16LE" in (e.text or "") for e in events)
    assert any(e.tool_name == "Bash" for e in events)


def test_the_handoff_produces_a_package(transcript, tmp_path):
    package, report = handoff(
        source=transcript,
        session_id="ses_" + "a" * 24,
        archive_root=tmp_path / "archive",
        title="Export work",
    )
    assert package.manifest.title == "Export work"
    assert report.events_read > 0
    assert report.events_abandoned == 1, "the rewound attempt"
    assert report.compacted, "the session was broken and resumed"


def test_the_raw_conversation_reaches_the_archive_before_anything_interprets_it(
    transcript, tmp_path
):
    """Extraction can fail and a model can be rate-limited. When that happens the
    conversation is already on disk and the expensive part can be retried."""
    session = "ses_" + "b" * 24
    _, report = handoff(source=transcript, session_id=session, archive_root=tmp_path / "archive")
    assert report.import_report is not None
    assert report.import_report.events_written > 0

    from open_context.archive import Archive

    archive = Archive(tmp_path / "archive")
    assert archive.exists(session)
    with archive.open(session) as log:
        assert len(list(log)) == report.import_report.events_written


def test_the_package_admits_it_carries_no_extracted_state(transcript, tmp_path):
    """A smaller claim honestly made rather than a larger one faked."""
    _, report = handoff(
        source=transcript, session_id="ses_" + "c" * 24, archive_root=tmp_path / "archive"
    )
    assert any("no state was extracted" in w for w in report.warnings)


def test_recent_context_leaves_out_tool_traffic(transcript):
    """Tool calls are 3,082 of 8,890 records in the sessions measured. A recent
    window full of shell invocations tells a receiving agent less than the same
    budget spent on what was said."""
    events, _ = read_session(transcript)
    recent = recent_context(events)
    assert recent.messages
    assert all(m["role"] in {"user", "assistant"} for m in recent.messages)
    assert not any("ss -ltn" in m["content"] for m in recent.messages)


def test_thinking_never_reaches_the_package(transcript, tmp_path):
    package, _ = handoff(
        source=transcript, session_id="ses_" + "d" * 24, archive_root=tmp_path / "archive"
    )
    assert "maybe UTF-8 is fine" not in package.to_json()


# ----------------------------------------------------------------------
# The whole workflow, through the CLI


def test_a_cross_agent_handoff_runs_end_to_end(transcript, tmp_path, capsys):
    """The exit criterion, start to finish: an agent's own session file becomes
    a validated package and then context for a *different* agent."""
    store = tmp_path / "store"
    out = tmp_path / "project.ctx"

    assert main(["handoff", str(transcript), "--store", str(store), "--out", str(out)]) == EXIT_OK
    assert out.exists()
    capsys.readouterr()

    assert main(["validate", str(out), "--archive", str(store)]) == EXIT_OK
    assert "valid" in capsys.readouterr().out

    for target in ("anthropic", "openai", "gemini"):
        assert (
            main(
                [
                    "compile",
                    str(out),
                    "--target",
                    target,
                    "--budget",
                    "400",
                    "--task",
                    "Finish the export writer.",
                ]
            )
            == EXIT_OK
        )
        payload = json.loads(capsys.readouterr().out)
        assert "Finish the export writer." in json.dumps(payload), target


def test_the_handoff_reports_what_it_kept_and_dropped(transcript, tmp_path, capsys):
    main(
        [
            "handoff",
            str(transcript),
            "--store",
            str(tmp_path / "store"),
            "--out",
            str(tmp_path / "p.ctx"),
        ]
    )
    err = capsys.readouterr().err
    assert "abandoned as rewound" in err
    assert "segments" in err


def test_the_session_id_comes_from_the_transcripts_own_name(transcript, tmp_path, capsys):
    main(
        [
            "handoff",
            str(transcript),
            "--store",
            str(tmp_path / "store"),
            "--out",
            str(tmp_path / "p.ctx"),
        ]
    )
    package = json.loads((tmp_path / "p.ctx").read_text())
    assert package["manifest"]["session_id"].startswith("ses_")
    assert "912bcf79" in package["manifest"]["session_id"]


# ----------------------------------------------------------------------
# This machine's real sessions, when it has any


def test_a_real_session_on_this_machine_hands_off(tmp_path):
    """The format was measured on real transcripts; this checks the code still
    reads them. Skipped where there are none, so the suite runs anywhere."""
    sessions = find_sessions()
    if not sessions:
        pytest.skip("no Claude Code sessions on this machine")

    smallest = min(sessions, key=lambda path: path.stat().st_size)
    package, report = handoff(
        source=smallest, session_id="ses_" + "e" * 24, archive_root=tmp_path / "archive"
    )
    assert report.events_read > 0
    assert report.events_kept > 0
    assert package.intact()


# ----------------------------------------------------------------------
# Extraction, which is the only step that needs a model


def test_a_provider_turns_a_transcript_into_actual_state(transcript, tmp_path):
    """Without this, a package carries recent context and nothing a receiving
    agent can act on. It is opt-in because it is the only step needing a model."""
    from open_context.llm.fakes import FakeProvider

    extracted = json.dumps(
        {
            "items": [
                {
                    "type": "goal",
                    "content": "Ship the reporting export writer",
                    "confidence": 0.9,
                    "source_ids": ["u1"],
                },
                {
                    "type": "decision",
                    "content": "Write UTF-16LE with a BOM",
                    "rationale": "the loader rejects UTF-8",
                    "confidence": 0.9,
                    "source_ids": ["a1"],
                },
            ]
        }
    )
    package, report = handoff(
        source=transcript,
        session_id="ses_" + "f" * 24,
        archive_root=tmp_path / "archive",
        provider=FakeProvider(responses=[extracted]),
    )
    assert report.extracted
    assert report.state_items == 2
    contents = [item.content for item in package.state]
    assert "Ship the reporting export writer" in contents


def test_asking_and_getting_nothing_is_not_the_same_as_not_asking(transcript, tmp_path):
    """Only one of those says anything about the conversation."""
    from open_context.llm.fakes import FakeProvider

    _, asked = handoff(
        source=transcript,
        session_id="ses_" + "g" * 24,
        archive_root=tmp_path / "a1",
        provider=FakeProvider(responses=['{"items": []}']),
    )
    _, never = handoff(
        source=transcript, session_id="ses_" + "h" * 24, archive_root=tmp_path / "a2"
    )
    assert asked.extracted and asked.state_items == 0
    assert not never.extracted
    assert any("pass a provider" in w for w in never.warnings)
    assert not any("pass a provider" in w for w in asked.warnings)


def test_a_failing_model_costs_the_model_call_and_not_the_conversation(transcript, tmp_path):
    """The archive is written first, so extraction can be retried without
    re-reading anything."""
    from open_context.llm.errors import RateLimitError
    from open_context.llm.fakes import FailingProvider

    session = "ses_" + "i" * 24
    package, report = handoff(
        source=transcript,
        session_id=session,
        archive_root=tmp_path / "archive",
        provider=FailingProvider(RateLimitError("slow down")),
        # A rate limit is now waited out, so a test that did not say otherwise
        # would spend its time proving the clock works.
        sleep=lambda _seconds: None,
    )
    assert report.extracted
    assert report.state_items == 0
    assert any("extraction failed" in w for w in report.warnings)

    from open_context.archive import Archive

    with Archive(tmp_path / "archive").open(session) as log:
        assert len(list(log)) > 0, "the conversation survived the failure"
    assert package.intact()


def test_one_long_turn_does_not_take_the_whole_recent_window(tmp_path):
    """Measured on a real handoff: a single assistant message spent 476 of a
    500-token budget and pushed ten other turns out. Breadth is the point."""
    from open_context.handoff import MAX_TURN_CHARACTERS, recent_context

    rows = [row(f"u{n}", f"u{n - 1}" if n else None, content="short turn") for n in range(6)]
    rows.append(row("big", "u5", kind="assistant", content="x" * 40_000))
    path = tmp_path / f"{SESSION_UUID}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    events, _ = read_session(path)
    recent = recent_context(events)
    longest = max(len(m["content"]) for m in recent.messages)
    assert longest <= MAX_TURN_CHARACTERS + 20
    assert any("truncated" in m["content"] for m in recent.messages)
    assert len(recent.messages) >= 6, "the other turns survived"


def test_the_cli_can_actually_reach_a_registered_provider(
    monkeypatch, transcript, tmp_path, capsys
):
    """`--extract` was entirely non-functional: providers register themselves on
    import and the CLI never imported them, so `create_provider` saw an empty
    registry and every run failed with "no provider registered as 'groq'".

    The design is right — the abstraction stays network-free, and a caller that
    wants a real model imports the providers on purpose — but the CLI *is* such
    a caller and had not said so.
    """
    from open_context.__main__ import main

    monkeypatch.setenv("OPEN_CONTEXT_PROVIDER", "groq")
    monkeypatch.setenv("OPEN_CONTEXT_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("OPEN_CONTEXT_API_KEY", "not-a-real-key")

    main(
        [
            "handoff",
            str(transcript),
            "--store",
            str(tmp_path / "store"),
            "--out",
            str(tmp_path / "p.ctx"),
            "--extract",
        ]
    )
    err = capsys.readouterr().err
    assert "no provider registered" not in err, err


def test_a_rate_limited_extraction_is_not_reported_as_an_empty_conversation(transcript, tmp_path):
    """ "No state was extracted" reads as a claim about the session. When the
    model was asked and refused, the claim is about the quota, and a first-time
    user needs to know which."""
    from open_context.llm.errors import RateLimitError
    from open_context.llm.fakes import FailingProvider

    _, report = handoff(
        source=transcript,
        session_id="ses_" + "1a" * 12,
        archive_root=tmp_path / "archive",
        provider=FailingProvider(RateLimitError("slow down", retry_after=1.0)),
        sleep=lambda _s: None,
    )
    assert report.extracted
    assert report.state_items == 0
    assert any("rate limit" in w for w in report.warnings)
    assert any("running the same\n" in w or "again resumes" in w for w in report.warnings)
    assert not any("pass a provider" in w for w in report.warnings)


def test_a_genuinely_empty_session_says_that_instead(transcript, tmp_path):
    from open_context.llm.fakes import FakeProvider

    _, report = handoff(
        source=transcript,
        session_id="ses_" + "2b" * 12,
        archive_root=tmp_path / "archive",
        provider=FakeProvider(responses=['{"items": []}']),
    )
    assert any("found no goals" in w for w in report.warnings)
    assert not any("rate limit" in w for w in report.warnings)
