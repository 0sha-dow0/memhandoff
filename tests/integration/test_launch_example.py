"""The public five-minute example must remain a real, offline handoff."""

import json
from pathlib import Path

from open_context.__main__ import EXIT_OK, main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT = PROJECT_ROOT / "examples" / "cross-agent-handoff" / "agent-a.jsonl"


def test_public_example_hands_context_from_agent_a_to_agent_b(tmp_path, capsys):
    archive = tmp_path / "archive"
    package = tmp_path / "export.ctx"

    assert (
        main(
            [
                "handoff",
                str(TRANSCRIPT),
                "--store",
                str(archive),
                "--out",
                str(package),
                "--title",
                "Customer export writer",
            ]
        )
        == EXIT_OK
    )
    capsys.readouterr()

    assert main(["validate", str(package), "--archive", str(archive)]) == EXIT_OK
    assert "valid" in capsys.readouterr().out

    package_payload = json.loads(package.read_text(encoding="utf-8"))
    assert package_payload["manifest"]["source"] == "generic-jsonl"

    assert (
        main(
            [
                "compile",
                str(package),
                "--target",
                "generic",
                "--budget",
                "800",
                "--task",
                "Continue Agent A's export work.",
            ]
        )
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    context = payload["text"]

    for fact in (
        "UTF-16LE",
        "byte-order mark",
        "500 rows",
        "12,418",
        "shared NFS mount",
        "deployment configuration",
        "Kafka",
        "9443",
    ):
        assert fact in context


def test_public_generic_example_is_not_reported_as_a_compacted_claude_session(tmp_path, capsys):
    package = tmp_path / "export.ctx"

    assert (
        main(
            [
                "handoff",
                str(TRANSCRIPT),
                "--store",
                str(tmp_path / "archive"),
                "--out",
                str(package),
            ]
        )
        == EXIT_OK
    )

    assert "segments" not in capsys.readouterr().err
