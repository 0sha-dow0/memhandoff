"""V2.1: reacting to an agent's own context pressure.

The roadmap forbids implementing this until real agent hooks are verified. They
were: Claude Code 2.1.195 emits `PreCompact` before compacting, with
`trigger: "auto"` when it decided by itself. These pin the behaviour against
that contract — including the part where this tool declines a power the protocol
offers it.
"""

import json

from open_context.__main__ import EXIT_OK, main
from open_context.package import CtxPackage, validate
from open_context.pressure import (
    AUTOMATIC,
    PRE_COMPACT,
    HookEvent,
    HookResponse,
    PressureConfig,
    handle,
    should_capture,
)

SESSION_UUID = "912bcf79-7320-42d7-b702-749ceed71e5a"


def payload(**overrides):
    event = {
        "hook_event_name": PRE_COMPACT,
        "session_id": SESSION_UUID,
        "transcript_path": "",
        "cwd": "/work/project",
        "trigger": AUTOMATIC,
        "custom_instructions": "",
    }
    event.update(overrides)
    return event


def transcript(tmp_path, turns=40):
    rows = []
    for n in range(turns):
        rows.append(
            {
                "type": "assistant" if n % 2 else "user",
                "uuid": f"u{n}",
                "parentUuid": f"u{n - 1}" if n else None,
                "sessionId": SESSION_UUID,
                "timestamp": "2026-08-16T08:36:46.316Z",
                "cwd": "/work/project",
                "gitBranch": "main",
                "version": "2.1.195",
                "userType": "external",
                "isSidechain": False,
                "entrypoint": "cli",
                "message": {
                    "role": "assistant" if n % 2 else "user",
                    "content": f"turn {n} about the reporting export writer, at some length",
                },
            }
        )
    path = tmp_path / f"{SESSION_UUID}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


# ----------------------------------------------------------------------
# The verified contract


def test_the_real_payload_shape_parses():
    """Fields taken from the shipping binary, not invented."""
    event = HookEvent.parse(json.dumps(payload(transcript_path="/tmp/x.jsonl")))
    assert event.event == PRE_COMPACT
    assert event.session_id == SESSION_UUID
    assert event.transcript_path == "/tmp/x.jsonl"
    assert event.trigger == AUTOMATIC
    assert event.is_pressure


def test_an_automatic_compaction_is_pressure_and_a_manual_one_is_not():
    """`trigger` is the whole signal: "auto" means the agent decided its window
    was full; "manual" means a person typed /compact."""
    assert HookEvent.parse(json.dumps(payload())).is_pressure
    assert not HookEvent.parse(json.dumps(payload(trigger="manual"))).is_pressure


def test_a_payload_this_code_does_not_understand_is_not_an_error():
    """A hook payload is input from another program, which may add or rename
    fields. Raising here would break the user's agent rather than this feature."""
    for text in ("", "not json", "[]", "null", '{"unexpected": true}'):
        event = HookEvent.parse(text)
        assert not event.is_pressure


# ----------------------------------------------------------------------
# What it declines to do


def test_the_response_always_lets_the_agent_continue():
    """The binary supports blocking compaction from this hook — "Compaction
    blocked by PreCompact hook" — and that power is deliberately unused.

    A context tool able to silently stop an agent compacting could run a session
    out of its window, and nobody debugging that would suspect the thing they
    installed to help.
    """
    assert json.loads(HookResponse().to_json())["continue"] is True

    import inspect

    from open_context import pressure

    source = inspect.getsource(pressure)
    assert '"decision"' not in source, "this handler must not vote on the compaction"
    assert "block" not in source.split('"""')[-1], "no blocking in executable code"


def test_a_failure_to_capture_still_lets_the_agent_continue(tmp_path):
    """A failed backup is not a reason to interfere with a running agent."""
    config = PressureConfig(store=tmp_path / "store")
    event = HookEvent.parse(json.dumps(payload(transcript_path=str(tmp_path / "gone.jsonl"))))
    response, note = handle(event, config)
    assert response.continue_ is True
    assert "no transcript" in note


# ----------------------------------------------------------------------
# When it acts


def test_context_pressure_captures_a_package(tmp_path):
    source = transcript(tmp_path)
    config = PressureConfig(store=tmp_path / "store")
    event = HookEvent.parse(json.dumps(payload(transcript_path=str(source))))

    response, note = handle(event, config)
    assert response.continue_ is True
    assert "captured" in note

    packages = list(config.packages.glob("*.ctx"))
    assert len(packages) == 1
    package = CtxPackage.from_json(packages[0].read_text())
    assert package.intact()
    assert package.recent is not None and package.recent.messages


def test_the_captured_package_validates_against_the_archive_it_names(tmp_path):
    source = transcript(tmp_path)
    config = PressureConfig(store=tmp_path / "store")
    handle(HookEvent.parse(json.dumps(payload(transcript_path=str(source)))), config)

    from open_context.archive import Archive

    package = CtxPackage.from_json(next(config.packages.glob("*.ctx")).read_text())
    archive = Archive(config.archives)
    with archive.open(package.manifest.session_id) as log:
        records = {record.seq: record for record in log}
    assert validate(package, archive_records=records).valid


def test_a_user_requested_compaction_is_left_alone(tmp_path):
    """They decided what they wanted. Capturing anyway is the tool asserting it
    knows better, and doubles what a session costs on disk."""
    config = PressureConfig(store=tmp_path / "store")
    event = HookEvent.parse(
        json.dumps(payload(transcript_path=str(transcript(tmp_path)), trigger="manual"))
    )
    wanted, why = should_capture(event, config)
    assert not wanted
    assert "requested by the user" in why


def test_a_tiny_transcript_is_not_worth_a_package(tmp_path):
    """A directory of near-empty packages is how a useful feature gets turned off."""
    source = tmp_path / f"{SESSION_UUID}.jsonl"
    source.write_text('{"type":"user"}\n')
    config = PressureConfig(store=tmp_path / "store")
    wanted, why = should_capture(
        HookEvent.parse(json.dumps(payload(transcript_path=str(source)))), config
    )
    assert not wanted
    assert "below the floor" in why


def test_another_hook_event_is_ignored(tmp_path):
    config = PressureConfig(store=tmp_path / "store")
    wanted, why = should_capture(
        HookEvent.parse(json.dumps(payload(hook_event_name="SessionStart"))), config
    )
    assert not wanted
    assert "not a compaction event" in why


# ----------------------------------------------------------------------
# Through the CLI, the way the agent will actually call it


def test_the_cli_reads_a_hook_event_from_stdin(tmp_path, capsys, monkeypatch):
    import io

    source = transcript(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload(transcript_path=str(source)))))
    assert main(["hook", "--store", str(tmp_path / "store")]) == EXIT_OK

    captured = capsys.readouterr()
    assert json.loads(captured.out)["continue"] is True
    assert "captured" in captured.err


def test_the_cli_exits_zero_even_when_it_could_not_capture(tmp_path, capsys, monkeypatch):
    """A non-zero exit is how a hook tells the agent to interfere."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("not a hook payload"))
    assert main(["hook", "--store", str(tmp_path / "store")]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["continue"] is True
