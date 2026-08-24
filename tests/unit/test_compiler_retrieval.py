"""V2.4: the compiler asking the archive for what the task needs.

    local archive -> lexical retrieval -> relevant evidence -> compiler

The reason this exists is a measurement, not a hunch. The adversarial benchmark
found that what compaction loses is *exact values* — all of its loss sat in the
two scenarios built around them. A summary cannot keep every number and does not
have to: the archive still holds them.
"""

import resource

import pytest

from open_context.archive.records import ArchiveRecord
from open_context.compiler import (
    CompilationRequest,
    Section,
    compile_context,
    render_for,
)
from open_context.llm.fakes import WordTokenizer
from open_context.models import ids
from open_context.models.state import Goal
from open_context.package import build

SESSION = "ses_" + "ab" * 12
TOKENIZER = WordTokenizer(exact=True)

ANSWER = "metrics listens on 8082, the admin panel on 8081"


def package(**kwargs):
    items = kwargs.pop(
        "state",
        [Goal(id=ids.new_id(ids.GOAL), session_id=SESSION, content="Ship the export writer")],
    )
    pkg, _ = build(session_id=SESSION, state=items, **kwargs)
    return pkg


def record(seq, text, **payload):
    return ArchiveRecord(
        seq=seq,
        id=f"m{seq:05d}",
        kind="message",
        payload={"type": "user_message", "text": text, **payload},
    )


def archive(size=60, answer_at=41):
    records = [record(n, f"turn {n}: routine work on the reporting export") for n in range(size)]
    records[answer_at] = record(answer_at, ANSWER)
    return records


def compiled(*, budget=400, task="which port do the metrics use", records=None, **kwargs):
    return compile_context(
        kwargs.pop("package", None) or package(),
        CompilationRequest(budget_tokens=budget, task=task, **kwargs),
        TOKENIZER,
        archive() if records is None else records,
    )


# ----------------------------------------------------------------------
# 1, 2: invoked when configured, disabled otherwise


def test_the_compiler_retrieves_when_asked():
    context = compiled(retrieve=3)
    assert ANSWER in context.text_of(Section.RETRIEVED)


def test_retrieval_is_off_unless_asked_for():
    """A package that arrived from another machine has no archive to search, so
    doing nothing is the right default."""
    assert compiled().text_of(Section.RETRIEVED) == ""


def test_no_archive_means_no_retrieval_even_when_requested():
    context = compile_context(
        package(),
        CompilationRequest(budget_tokens=400, task="metrics port", retrieve=3),
        TOKENIZER,
        None,
    )
    assert context.text_of(Section.RETRIEVED) == ""


def test_compiling_without_retrieval_is_unchanged():
    """The path that existed before must not shift underneath a caller."""
    pkg = package()
    plain = compile_context(pkg, CompilationRequest(budget_tokens=400, task="Go."), TOKENIZER)
    assert [s.section for s in plain.sections] == [
        Section.FRAME,
        Section.STATE,
        Section.TASK,
    ]


# ----------------------------------------------------------------------
# 3, 4: relevant in, irrelevant out


def test_the_relevant_record_is_the_one_returned():
    text = compiled(retrieve=1).text_of(Section.RETRIEVED)
    assert "8082" in text


def test_irrelevant_records_are_excluded():
    text = compiled(retrieve=3).text_of(Section.RETRIEVED)
    assert "routine work" not in text


def test_a_query_can_differ_from_the_task():
    """ "finish the export writer" is a task; "8082" is what finds the record."""
    context = compiled(task="finish the writer", retrieval_query="8082", retrieve=2)
    assert "8082" in context.text_of(Section.RETRIEVED)


# ----------------------------------------------------------------------
# 5: provenance


def test_retrieved_evidence_names_the_record_it_came_from():
    """A record cited without saying which record it was is an assertion, not
    evidence."""
    text = compiled(retrieve=1).text_of(Section.RETRIEVED)
    assert "[m00041]" in text


def test_the_section_says_the_records_came_from_the_archive():
    """A reader must be able to tell a record the task fetched from one the
    package chose to carry — they were selected by different things."""
    text = compiled(retrieve=1).text_of(Section.RETRIEVED)
    assert "From the archive" in text


def test_provenance_reaches_every_target():
    context = compiled(retrieve=1)
    for target in ("openai", "anthropic", "gemini", "generic"):
        import json

        assert "m00041" in json.dumps(render_for(target, context)), target


# ----------------------------------------------------------------------
# 6: budget


def test_the_compiled_context_stays_in_budget_with_retrieval():
    for budget in (60, 120, 400):
        context = compiled(budget=budget, retrieve=5)
        assert context.total_tokens <= budget, budget


def test_retrieval_is_truncated_rather_than_appended_wholesale():
    long_records = [
        record(n, f"record {n} about metrics port " + "detail " * 60) for n in range(40)
    ]
    context = compiled(budget=120, retrieve=20, records=long_records)
    assert context.total_tokens <= 120
    assert context.dropped.get("retrieved", 0) > 0


def test_a_budget_with_no_room_left_retrieves_nothing():
    """State is worth more than a record the task might have wanted.

    The provenance frame is paid for first and is not small, so it is turned off
    here to leave a budget that fits one section and not two — which is the case
    the ordering exists to decide.
    """
    context = compiled(budget=25, retrieve=5, frame_as_data=False)
    assert context.text_of(Section.STATE)
    assert context.text_of(Section.RETRIEVED) == ""


# ----------------------------------------------------------------------
# 7, 8: determinism and duplicates


def test_retrieval_is_deterministic():
    """Same package, same archive, same query — twice."""
    first = compiled(retrieve=3).text_of(Section.RETRIEVED)
    second = compiled(retrieve=3).text_of(Section.RETRIEVED)
    assert first == second


def test_ties_resolve_by_position_not_by_iteration_order():
    identical = [record(n, "metrics port 8082") for n in range(5)]
    ordered = compiled(retrieve=3, records=identical).text_of(Section.RETRIEVED)
    reversed_input = compiled(retrieve=3, records=list(reversed(identical))).text_of(
        Section.RETRIEVED
    )
    assert ordered == reversed_input


def test_a_record_the_package_already_quotes_is_not_repeated():
    """Retrieval searches the same archive the package's evidence came from, so
    it finds them again. Printing one twice spends budget to say nothing."""
    source = "msg_" + "1" * 24
    item = Goal(
        id=ids.new_id(ids.GOAL),
        session_id=SESSION,
        content="Ship the export writer",
        sources=[source],
    )
    quoted = ArchiveRecord(
        seq=41, id=source, kind="message", payload={"type": "user_message", "text": ANSWER}
    )
    pkg, _ = build(session_id=SESSION, state=[item], archive_records=[quoted])
    assert ANSWER in pkg.evidence[0].excerpt

    context = compile_context(
        pkg,
        CompilationRequest(budget_tokens=400, task="metrics port", retrieve=3),
        TOKENIZER,
        [quoted],
    )
    assert context.text_of(Section.RETRIEVED) == ""


# ----------------------------------------------------------------------
# 9, 10: nothing found, and failure


def test_a_query_that_matches_nothing_adds_no_section():
    context = compiled(task="xylophone unicycle", retrieve=3)
    assert context.text_of(Section.RETRIEVED) == ""


def test_an_empty_archive_is_not_an_error():
    assert compiled(retrieve=3, records=[]).text_of(Section.RETRIEVED) == ""


def test_a_failing_archive_does_not_fail_the_compilation():
    """A context without the extra records is smaller, not wrong. An unreadable
    archive must not deny a caller the package they already hold."""

    def broken():
        yield record(0, "metrics port 8082")
        raise OSError("the archive is damaged")

    context = compiled(retrieve=3, records=broken())
    assert context.text_of(Section.STATE), "the package still compiled"
    assert any("retrieval failed" in w for w in context.warnings)


# ----------------------------------------------------------------------
# The compiler must not load the archive


def test_the_archive_is_streamed_not_loaded():
    """An archive that must be read into a list to be searched has given up the
    property that made it worth keeping."""
    consumed = 0

    def stream():
        nonlocal consumed
        for n in range(80_000):
            consumed += 1
            yield record(n, f"turn {n} " + "padding " * 40)

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    context = compiled(budget=400, task="turn 79999", retrieve=3, records=stream())
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert consumed == 80_000, "every record was seen"
    unit = 1024 * 1024 if __import__("os").uname().sysname == "Darwin" else 1024
    assert (after - before) / unit < 150
    assert context.total_tokens <= 400


def test_an_iterator_is_accepted_not_only_a_sequence():
    context = compiled(retrieve=1, records=iter(archive()))
    assert ANSWER in context.text_of(Section.RETRIEVED)


# ----------------------------------------------------------------------
# 12: the package itself is untouched


def test_retrieval_does_not_change_the_package():
    pkg = package()
    before = pkg.to_json()
    compile_context(
        pkg,
        CompilationRequest(budget_tokens=400, task="metrics port", retrieve=3),
        TOKENIZER,
        archive(),
    )
    assert pkg.to_json() == before
    assert pkg.intact()


def test_a_negative_retrieve_count_is_simply_off():
    assert compiled(retrieve=-1).text_of(Section.RETRIEVED) == ""


@pytest.mark.parametrize("count", [1, 2, 5])
def test_retrieval_never_returns_more_than_asked_for(count):
    many = [record(n, f"metrics port 8082 mention {n}") for n in range(30)]
    text = compiled(budget=2000, retrieve=count, records=many).text_of(Section.RETRIEVED)
    assert text.count("- [") <= count
