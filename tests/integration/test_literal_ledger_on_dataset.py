"""The exact-value ledger, against the scenario that motivated it.

`adv-exact-values` is where every compacted arm in the `v4` matrix lost the
write timeout, and it is built to be lost: the value sits among three
deliberately confusable neighbours, stated once, far enough back that a 160
token budget summarises rather than quotes it.

The summariser here is a fake that returns prose with no numbers in it. That is
not a convenient stand-in — it is what the real run did. One failing answer from
that run says so outright: "the original context does not specify the exact
value", after which it invented one.

What this establishes is that the value reaches the compacted context. Whether a
downstream model then *uses* it is a separate question needing a real model, and
is recorded as unmeasured rather than assumed.
"""

import json

import pytest

from open_context.archive import Archive
from open_context.compaction import BaselineRecentPlusSummary, CompactionRequest
from open_context.compaction.budget import BaselineConfig
from open_context.importers import import_path
from open_context.llm.fakes import FakeProvider, WordTokenizer
from open_context_eval.adversarial import ADVERSARIAL_SCENARIOS

SESSION = "ses_ledger"
BUDGET = 160
"""The middle budget of the published matrix, where the failures were recorded."""

NUMBERLESS = (
    "The team discussed pipeline configuration and agreed on timeout settings "
    "for the client, along with batching behaviour for the ingest path."
)
"""A summary of the right shape and the wrong content: no digits at all."""


def scenario():
    return next(s for s in ADVERSARIAL_SCENARIOS if s.scenario_id == "adv-exact-values")


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "adv.jsonl"
    records = [
        {"id": f"turn{i}", "role": turn["role"], "content": turn["content"]}
        for i, turn in enumerate(scenario().conversation)
    ]
    path.write_text("".join(f"{json.dumps(r)}\n" for r in records), encoding="utf-8")

    archive = Archive(tmp_path / "archive")
    import_path(archive, SESSION, path)
    return archive.open(SESSION)


def compacted_text(log, *, preserve: bool) -> str:
    strategy = BaselineRecentPlusSummary(
        FakeProvider(reply=NUMBERLESS, context_window=100_000), WordTokenizer()
    )
    result = strategy.compact(
        CompactionRequest(
            log=log,
            target_tokens=BUDGET,
            config=BaselineConfig(preserve_literals=preserve),
        )
    )
    assert result.compaction_needed, "the scenario must be long enough to force a choice"
    return result.historical_summary


def test_the_scenario_states_the_value_once_among_its_distractors():
    """If this changes, the rest of the file is measuring something else."""
    text = json.dumps(scenario().conversation)
    assert "74 seconds" in text
    for distractor in ("47 seconds", "7 seconds", "4700"):
        assert distractor in text


def test_without_the_ledger_the_summary_loses_the_value(log):
    """The published failure, reproduced offline and without a model."""
    assert "74" not in compacted_text(log, preserve=False)


def test_with_the_ledger_the_value_survives_compaction(log):
    text = compacted_text(log, preserve=True)
    assert "74" in text
    assert "write timeout" in text.lower()


def test_the_ledger_does_not_smuggle_the_answer_in_as_a_distractor(log):
    """Carrying every number would make the confusion worse, not better.

    The scenario's whole design is that 47, 7 and 4700 are there to be mistaken
    for 74. A ledger that listed all four unlabelled would hand the model the
    same problem in a smaller box.
    """
    text = compacted_text(log, preserve=True)
    for line in text.splitlines():
        if "74" in line:
            assert "timeout" in line.lower(), f"the value arrived unlabelled: {line!r}"
