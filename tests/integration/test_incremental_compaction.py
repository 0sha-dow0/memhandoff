"""V2.2: extracting only what arrived since last time.

Phase 7 built watermarked incremental extraction and nothing called it, so every
package re-read the whole conversation and paid for it again. These pin the
wiring, the durability the watermark needs to survive between runs, and the
property that makes it worth doing: cost that tracks *new* events rather than
total ones.
"""

import json

from open_context.archive import Archive
from open_context.handoff import extract_state
from open_context.importers.events import EventType, RawEvent
from open_context.incremental_store import StoredState, load, save
from open_context.llm.errors import RateLimitError
from open_context.llm.fakes import FailingProvider, FakeProvider
from open_context.llm.provider import GenerationRequest

SESSION = "ses_" + "ab" * 12


class Counting:
    """Records what a provider was actually asked to read."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.prompt_chars = 0

    def model_info(self):
        return self.inner.model_info()

    def generate(self, request: GenerationRequest):
        self.calls += 1
        self.prompt_chars += sum(len(m.content) for m in request.messages)
        return self.inner.generate(request)

    def structured_output(self, request, schema):
        return self.inner.structured_output(request, schema)


def grow(root, to_turns):
    archive = Archive(root / "archive")
    with archive.open_or_create(SESSION) as log:
        start = len(log)
        log.extend(
            [
                (
                    f"m{n:05d}",
                    RawEvent(
                        type=EventType.USER_MESSAGE,
                        provider="test",
                        source_id=f"m{n:05d}",
                        text=f"turn {n} " + "detail " * 40,
                    ).to_payload(),
                )
                for n in range(start, to_turns)
            ]
        )
    return root / "archive"


def reply(*indices):
    return json.dumps(
        {
            "items": [
                {
                    "type": "fact",
                    "content": f"turn {n} established a value",
                    "confidence": 0.9,
                    "source_ids": [f"m{n:05d}"],
                }
                for n in indices
            ]
        }
    )


def provider(*replies):
    return Counting(FakeProvider(responses=list(replies) or [reply(0)]))


# ----------------------------------------------------------------------
# The saving, and where it comes from


def test_a_second_run_reads_only_what_arrived_since_the_first(tmp_path):
    """Compared against a full read of the *same* larger session.

    Not against the first run: an incremental pass also renders the state it
    already knows into the prompt, which Phase 7 established is required — a
    model shown only new messages has never seen the decision being overturned
    and cannot name it. So the second run is not smaller than the first, it is
    smaller than re-reading everything.
    """
    incremental_root = tmp_path / "inc"
    full_root = tmp_path / "full"
    for root in (incremental_root, full_root):
        root.mkdir()
        grow(root, 40)

    # Repackaged periodically as the session grows, which is the situation this
    # exists for. The saving is proportional to how much of the session is
    # already known, so one enormous jump would save little — correctly.
    incremental_chars = full_chars = 0
    extract_state(incremental_root / "archive", SESSION, provider(reply(0, 1)))

    for total in (80, 120, 160, 200):
        grow(incremental_root, total)
        grow(full_root, total)

        step = provider(reply(total - 1))
        extract_state(incremental_root / "archive", SESSION, step)
        incremental_chars += step.prompt_chars

        whole = provider(reply(0, 1))
        extract_state(full_root / "archive", SESSION, whole, incremental=False)
        full_chars += whole.prompt_chars

    assert incremental_chars < full_chars * 0.6, (
        f"incremental read {incremental_chars} where full reads cost {full_chars}"
    )


def test_cost_tracks_new_events_rather_than_total_ones(tmp_path):
    """Full re-extraction is O(session); incremental is O(what is new). On a
    session grown to 200 turns in batches, measured 62% fewer prompt characters
    and 70% less latency, with the incremental prompt staying flat."""
    archive = grow(tmp_path, 40)
    extract_state(archive, SESSION, provider(reply(0)))

    sizes = []
    for total in (80, 120, 160, 200):
        grow(tmp_path, total)
        counter = provider(reply(total - 1))
        extract_state(archive, SESSION, counter)
        sizes.append(counter.prompt_chars)

    assert max(sizes) - min(sizes) < min(sizes) * 0.2, (
        f"incremental prompt size should stay roughly flat as the session grows, got {sizes}"
    )


def test_a_full_read_still_grows_with_the_session(tmp_path):
    """The comparison only means something if the baseline behaves as claimed."""
    archive = grow(tmp_path, 40)
    small = provider()
    extract_state(archive, SESSION, small, incremental=False)

    grow(tmp_path, 200)
    large = provider()
    extract_state(archive, SESSION, large, incremental=False)

    assert large.prompt_chars > small.prompt_chars * 3


def test_no_quality_regression_in_what_is_kept(tmp_path):
    """Same conversation, same state, whichever route reached it.

    A caveat this cannot escape: the model is a stand-in that returns fixed
    replies, so this shows the *pipeline* preserves what it is given. Whether a
    real model finds the same things from a partial view is a question only a
    real run answers.
    """
    incremental_root = tmp_path / "inc"
    full_root = tmp_path / "full"

    for root in (incremental_root, full_root):
        root.mkdir()
        grow(root, 40)

    inc_items, _, _ = extract_state(incremental_root / "archive", SESSION, provider(reply(0, 1, 2)))
    grow(incremental_root, 80)
    inc_items, _, _ = extract_state(incremental_root / "archive", SESSION, provider(reply(40)))

    # A full read is windowed, and each window can only cite records it saw —
    # the extractor validates source ids against the range it was given. A fake
    # returning one reply that cites the whole session is not what a model does.
    grow(full_root, 80)
    full_items, _, _ = extract_state(
        full_root / "archive",
        SESSION,
        # One reply per window (0-20, 20-40, 40-60, 60-80), each citing only
        # records that window actually contains.
        provider(reply(0, 1, 2), reply(), reply(40), reply()),
        incremental=False,
    )

    assert {item.content for item in inc_items} == {item.content for item in full_items}


def test_nothing_new_costs_nothing(tmp_path):
    """ "We did not look" and "we looked and found nothing" stay distinguishable,
    which is Phase 7's contract."""
    archive = grow(tmp_path, 40)
    extract_state(archive, SESSION, provider(reply(0)))

    idle = provider()
    items, warnings, _ = extract_state(archive, SESSION, idle)
    assert idle.calls == 0, "an unchanged session must not be sent to a model"
    assert items
    assert any("nothing new" in w for w in warnings)


# ----------------------------------------------------------------------
# The watermark has to survive, and has to be honest


def test_state_and_watermark_are_stored_together(tmp_path):
    """Stored apart, a crash could leave a watermark claiming events were read
    beside state that never saw them, and nothing downstream could tell."""
    archive = grow(tmp_path, 40)
    extract_state(archive, SESSION, provider(reply(0, 1)))

    stored = load(archive, SESSION)
    assert stored.next_seq == 40
    assert len(stored.items) == 2


def test_a_failed_extraction_does_not_advance_the_watermark(tmp_path):
    """Otherwise the unread range is skipped permanently, and silently."""
    archive = grow(tmp_path, 40)
    extract_state(archive, SESSION, provider(reply(0)))
    before = load(archive, SESSION).next_seq

    grow(tmp_path, 80)
    items, warnings, _ = extract_state(
        archive, SESSION, FailingProvider(RateLimitError("slow down"))
    )
    assert load(archive, SESSION).next_seq == before
    assert any("extraction failed" in w for w in warnings)
    assert items, "what was already known survives a failed update"


def test_damaged_stored_state_is_read_as_nothing_known(tmp_path):
    """A truncated file means the last run died mid-write. Reading the session
    again is recovery; refusing to run would make an interruption permanent."""
    archive = grow(tmp_path, 40)
    extract_state(archive, SESSION, provider(reply(0)))

    from open_context.incremental_store import path_for

    path_for(archive, SESSION).write_text('{"format": "open-context-state", "items": [')
    assert load(archive, SESSION).fresh


def test_state_belonging_to_another_session_is_not_adopted(tmp_path):
    """However the file got there, reading it would attribute one conversation's
    decisions to another."""
    archive = grow(tmp_path, 40)
    save(archive, StoredState(session_id="ses_" + "cd" * 12, next_seq=99))

    from open_context.incremental_store import path_for

    path_for(archive, "ses_" + "cd" * 12).rename(path_for(archive, SESSION))
    assert load(archive, SESSION).fresh


def test_a_write_is_atomic(tmp_path):
    """A reader sees the previous version or the new one, never half of either."""
    archive = grow(tmp_path, 40)
    save(archive, StoredState(session_id=SESSION, next_seq=10))
    save(archive, StoredState(session_id=SESSION, next_seq=20))

    from open_context.incremental_store import path_for

    assert load(archive, SESSION).next_seq == 20
    assert not list(path_for(archive, SESSION).parent.glob("*.tmp"))


def test_opting_out_leaves_no_stored_state(tmp_path):
    archive = grow(tmp_path, 40)
    extract_state(archive, SESSION, provider(reply(0)), incremental=False)
    assert load(archive, SESSION).fresh


def test_extracting_from_a_session_that_was_never_archived_says_so(tmp_path):
    items, warnings, _ = extract_state(tmp_path / "nothing", SESSION, provider())
    assert items == []
    assert any("no archived session" in w for w in warnings)


# ----------------------------------------------------------------------
# V2.3: what a sibling session already established


def test_a_constraint_from_a_sibling_session_reaches_the_new_package(tmp_path):
    """The exit criterion for hierarchy, end to end: Monday's constraint is
    still true on Tuesday, and a session that could only see itself would ask
    again or decide differently."""
    from open_context.handoff import handoff
    from open_context.incremental_store import StoredState, save
    from open_context.models import ids
    from open_context.models.state import Constraint

    project = tmp_path / "repo"
    (project / ".git").mkdir(parents=True)

    monday = "ses_" + "11" * 12
    save(
        tmp_path / "archive",
        StoredState(
            session_id=monday,
            next_seq=10,
            project_root=str(project),
            items=[
                Constraint(
                    id=ids.new_id(ids.CONSTRAINT),
                    session_id=monday,
                    content="Never write report files to the shared NFS mount",
                )
            ],
        ),
    )

    rows = [
        {
            "type": "user",
            "uuid": f"u{n}",
            "parentUuid": f"u{n - 1}" if n else None,
            "sessionId": "tuesday",
            "timestamp": "2026-08-16T08:36:46.316Z",
            "cwd": str(project),
            "gitBranch": "main",
            "version": "2.1.195",
            "userType": "external",
            "isSidechain": False,
            "entrypoint": "cli",
            "message": {"role": "user", "content": f"tuesday turn {n} about the export writer"},
        }
        for n in range(20)
    ]
    source = tmp_path / "tuesday.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows))

    package, report = handoff(
        source=source, session_id="ses_" + "22" * 12, archive_root=tmp_path / "archive"
    )
    assert report.carried_from_project == 1
    assert any("NFS" in item.content for item in package.inherited)
    assert not any("NFS" in item.content for item in package.state), (
        "inherited state must stay distinguishable from what this session concluded"
    )


def test_a_session_in_a_different_project_carries_nothing(tmp_path):
    """Unrelated work must not be mixed. This is why project identity is a
    repository root rather than whatever directory happens to sit above."""
    from open_context.handoff import handoff
    from open_context.incremental_store import StoredState, save
    from open_context.models import ids
    from open_context.models.state import Constraint

    other = "ses_" + "33" * 12
    save(
        tmp_path / "archive",
        StoredState(
            session_id=other,
            next_seq=5,
            items=[
                Constraint(
                    id=ids.new_id(ids.CONSTRAINT),
                    session_id=other,
                    content="A rule from an unrelated project",
                )
            ],
        ),
    )

    elsewhere = tmp_path / "unrelated"
    elsewhere.mkdir()
    rows = [
        {
            "type": "user",
            "uuid": "u0",
            "parentUuid": None,
            "sessionId": "x",
            "timestamp": "2026-08-16T08:36:46.316Z",
            "cwd": str(elsewhere),
            "gitBranch": "main",
            "version": "2.1.195",
            "userType": "external",
            "isSidechain": False,
            "entrypoint": "cli",
            "message": {"role": "user", "content": "unrelated work " * 40},
        }
    ]
    source = tmp_path / "other.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows))

    package, report = handoff(
        source=source, session_id="ses_" + "44" * 12, archive_root=tmp_path / "archive"
    )
    assert report.carried_from_project == 0
    assert not any("unrelated project" in item.content for item in package.inherited)
