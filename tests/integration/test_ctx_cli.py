"""Phase 9's exit criterion, end to end.

> A `.ctx` file can be generated, validated, and inspected.

Through the real CLI, against a real SQLite store and a real archive on disk. A
unit test of `build` proves the function works; this proves the artifact can
actually be produced from a session and read back by something that was not
there when it was written — which is the whole claim the format makes.
"""

import json

import pytest

from open_context.__main__ import EXIT_INVALID, EXIT_OK, EXIT_UNUSABLE, main
from open_context.archive import Archive
from open_context.models import ids
from open_context.models.conversation import Message, Session
from open_context.models.enums import Role
from open_context.models.state import Constraint, Decision, Goal
from open_context.storage import Database, Repository

SESSION = "ses_" + "c" * 24


@pytest.fixture
def store(tmp_path):
    """A session with state and an archive, as a real run would leave it."""
    root = tmp_path / "ctx"
    root.mkdir()

    repository = Repository(Database(root / "state.sqlite"))
    repository.add_session(Session(id=SESSION, title="Reporting export", source="test"))
    repository.add_messages(
        [
            Message(
                id=ids.new_id(ids.MESSAGE),
                session_id=SESSION,
                seq=index,
                role=Role.USER,
                content=text,
            )
            for index, text in enumerate(["ship the export writer", "never write to the NFS mount"])
        ]
    )
    repository.add_state_items(
        [
            Goal(id=ids.new_id(ids.GOAL), session_id=SESSION, content="Ship the export writer"),
            Constraint(
                id=ids.new_id(ids.CONSTRAINT),
                session_id=SESSION,
                content="Never write report files to the shared NFS mount",
            ),
            Decision(
                id=ids.new_id(ids.DECISION),
                session_id=SESSION,
                content="Write UTF-16LE with a BOM",
                rationale="the downstream loader rejects UTF-8",
            ),
        ]
    )

    archive = Archive(root / "archive")
    with archive.create(SESSION) as log:
        log.extend(
            [
                ("msg_1", {"content": "ship the export writer"}),
                ("msg_2", {"content": "never write to the NFS mount"}),
            ]
        )
    return root


def test_a_ctx_file_can_be_generated(store, tmp_path, capsys):
    out = tmp_path / "project.ctx"
    assert main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)]) == EXIT_OK
    assert out.exists()

    document = json.loads(out.read_text())
    assert document["manifest"]["format"] == "open-context-ctx"
    assert document["manifest"]["session_id"] == SESSION
    assert len(document["state"]) == 3


def test_a_ctx_file_can_be_validated(store, tmp_path, capsys):
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])
    capsys.readouterr()

    assert main(["validate", str(out)]) == EXIT_OK
    assert "valid" in capsys.readouterr().out


def test_validation_against_the_archive_reaches_the_deeper_check(store, tmp_path, capsys):
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])
    capsys.readouterr()

    assert main(["validate", str(out), "--archive", str(store / "archive")]) == EXIT_OK
    assert "provenance check" in capsys.readouterr().out


def test_validation_without_an_archive_says_what_it_did_not_check(store, tmp_path, capsys):
    """The shallow pass must not be mistaken for the deep one."""
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])
    capsys.readouterr()

    main(["validate", str(out)])
    captured = capsys.readouterr()
    assert "structure check" in captured.out
    assert "provenance was not checked" in captured.err


def test_a_tampered_package_fails_validation_with_a_non_zero_exit(store, tmp_path, capsys):
    """Non-zero so a pipeline can gate on it without parsing any output."""
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])

    document = json.loads(out.read_text())
    document["state"][0]["content"] = "Ship something else"
    out.write_text(json.dumps(document))
    capsys.readouterr()

    assert main(["validate", str(out)]) == EXIT_INVALID
    assert "hash-mismatch" in capsys.readouterr().out


def test_a_file_that_is_not_a_package_exits_differently_from_an_invalid_one(tmp_path, capsys):
    """ "Could not read it" and "read it, and it is wrong" are different problems."""
    bad = tmp_path / "not.ctx"
    bad.write_text('{"nope": true}')
    assert main(["validate", str(bad)]) == EXIT_UNUSABLE


def test_a_ctx_file_can_be_inspected(store, tmp_path, capsys):
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])
    capsys.readouterr()

    assert main(["inspect", str(out)]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Ship the export writer" in text
    assert "Never write report files to the shared NFS mount" in text
    assert "GOAL" in text and "CONSTRAINT" in text and "DECISION" in text


def test_the_package_references_the_archive_without_copying_it(store, tmp_path, capsys):
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])

    package = json.loads(out.read_text())
    assert package["archive"]["record_count"] == 2
    assert package["archive"]["session_id"] == SESSION
    assert "recent" not in json.dumps(package["archive"])


def test_the_same_package_compiles_for_several_provider_families(store, tmp_path, capsys):
    """Phase 10's exit criterion, from a file on disk.

    Same package, same content, each family's own request shape.
    """
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])
    capsys.readouterr()

    shapes = {}
    for target in ("openai", "anthropic", "gemini", "generic"):
        assert (
            main(
                [
                    "compile",
                    str(out),
                    "--target",
                    target,
                    "--budget",
                    "300",
                    "--task",
                    "Write the export writer.",
                ]
            )
            == EXIT_OK
        )
        payload = json.loads(capsys.readouterr().out)
        shapes[target] = set(payload)
        rendered = json.dumps(payload)
        assert "Never write report files to the shared NFS mount" in rendered, target
        assert "Write the export writer." in rendered, target

    assert shapes["openai"] == {"messages"}
    assert shapes["anthropic"] == {"messages", "system"}
    assert shapes["gemini"] == {"contents", "systemInstruction"}
    assert shapes["generic"] == {"text"}


def test_compiling_reports_the_budget_and_what_it_dropped(store, tmp_path, capsys):
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])
    capsys.readouterr()

    main(["compile", str(out), "--target", "generic", "--budget", "30", "--task", "Go."])
    err = capsys.readouterr().err
    assert "of a 30 budget" in err
    assert "estimated" in err, "an estimate presented as exact is how a context overflows"


def test_packing_reports_what_it_could_not_trace(store, tmp_path, capsys):
    """State stored by earlier phases carries no archive provenance.

    `sources` speaks the SQLite id scheme; the archive numbers its own records.
    The gap is real, and the CLI says so rather than producing a package that
    looks fully traceable.
    """
    out = tmp_path / "project.ctx"
    main(["pack", "--session", SESSION, "--store", str(store), "--out", str(out)])
    assert "no archive provenance" in capsys.readouterr().err
