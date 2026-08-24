"""Phase 10: one `.ctx`, several provider families.

The exit criterion is that the same package produces useful context for
multiple model/provider families. What most of these actually pin is the
*constraint* that makes it true — decisions happen once, in the portable layer,
and a renderer cannot make them even if a later change tries to let it.
"""

import inspect
import json

import pytest

from open_context.compiler import (
    CompilationRequest,
    Section,
    compile_context,
    known_targets,
    render_anthropic,
    render_for,
    render_gemini,
    render_generic,
    render_openai,
)
from open_context.compiler import targets as targets_module
from open_context.llm.fakes import WordTokenizer
from open_context.models import ids
from open_context.models.state import Constraint, Decision, Entity, Fact, Goal, OpenQuestion
from open_context.package import RecentContext, build

SESSION = "ses_" + "a" * 24
TOKENIZER = WordTokenizer(exact=True)


def state():
    return [
        Goal(id=ids.new_id(ids.GOAL), session_id=SESSION, content="Ship the export writer"),
        Constraint(
            id=ids.new_id(ids.CONSTRAINT),
            session_id=SESSION,
            content="Never write report files to the shared NFS mount",
        ),
        Fact(id=ids.new_id(ids.FACT), session_id=SESSION, content="The metrics port is 8082"),
        Decision(
            id=ids.new_id(ids.DECISION),
            session_id=SESSION,
            content="Write UTF-16LE with a BOM",
            rationale="the downstream loader rejects UTF-8",
        ),
        OpenQuestion(
            id=ids.new_id(ids.OPEN_QUESTION),
            session_id=SESSION,
            content="Should the writer stream or buffer?",
        ),
        Entity(
            id=ids.new_id(ids.ENTITY),
            session_id=SESSION,
            content="reporting.export",
            name="reporting.export",
        ),
    ]


def package(**kwargs):
    pkg, _ = build(session_id=SESSION, state=kwargs.pop("state", None) or state(), **kwargs)
    return pkg


def compiled(budget=400, task="Write the export writer.", **kwargs):
    return compile_context(
        kwargs.pop("package", None) or package(),
        CompilationRequest(budget_tokens=budget, task=task, **kwargs),
        TOKENIZER,
    )


# ----------------------------------------------------------------------
# The exit criterion


def test_one_package_compiles_for_every_known_family():
    """The same `.ctx`, useful context in each family's own shape."""
    context = compiled()
    for target in known_targets():
        payload = render_for(target, context)
        rendered = json.dumps(payload)
        assert "Ship the export writer" in rendered, target
        assert "Never write report files" in rendered, target
        assert "Write the export writer." in rendered, target


def test_each_family_gets_its_own_shape_not_a_shared_one():
    context = compiled()
    assert list(render_openai(context)) == ["messages"]
    assert set(render_anthropic(context)) == {"messages", "system"}
    assert set(render_gemini(context)) == {"systemInstruction", "contents"}
    assert list(render_generic(context)) == ["text"]


def test_openai_puts_standing_context_in_a_system_message():
    payload = render_openai(compiled())
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1]["role"] == "user"


def test_anthropic_keeps_system_outside_the_message_list():
    payload = render_anthropic(compiled())
    assert "system" in payload
    assert all(m["role"] != "system" for m in payload["messages"])


def test_anthropic_always_begins_with_a_user_turn():
    """An empty message list is not a valid request, so a context with no task
    still gets one turn rather than producing something that cannot be sent."""
    payload = render_anthropic(compiled(task=""))
    assert payload["messages"]
    assert payload["messages"][0]["role"] == "user"


def test_gemini_wraps_text_in_parts():
    payload = render_gemini(compiled())
    assert payload["contents"][0]["parts"][0]["text"]
    assert payload["contents"][0]["role"] == "user"


def test_an_unknown_family_still_gets_usable_context():
    """Refusing would make the format less portable than the text it is made of."""
    payload = render_for("some-provider-nobody-has-written-an-adapter-for", compiled())
    assert "Ship the export writer" in payload["text"]


# ----------------------------------------------------------------------
# The constraint that makes the above meaningful


def test_a_renderer_is_never_given_a_budget_or_a_tokenizer():
    """Five adapters that could choose would become five slightly different
    compactors, and the same `.ctx` would quietly mean different things
    depending on where it was sent.

    A renderer takes exactly one argument, and that argument carries no budget.
    """
    for name, renderer in targets_module.TARGETS.items():
        parameters = list(inspect.signature(renderer).parameters)
        assert parameters == ["context"], f"{name} takes more than a compiled context"

    from open_context.compiler.portable import CompiledContext

    fields = set(CompiledContext.__dataclass_fields__)
    assert "budget_tokens" not in fields
    assert "tokenizer" not in fields


def test_no_adapter_reimplements_fitting():
    """Compaction logic lives in one place. An adapter that imported the
    tokenizer or the fitter would be the start of a second copy.

    Checked against names the module actually uses, not its text: the docstring
    explains this constraint and so necessarily contains the words.
    """
    import ast

    tree = ast.parse(inspect.getsource(targets_module))
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in ("fit_state", "render_state", "count_text", "Tokenizer", "budget_tokens"):
        assert forbidden not in used, f"targets.py uses {forbidden!r}"

    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    for forbidden in ("open_context.hybrid.compactor", "open_context.llm.tokens"):
        assert forbidden not in modules, f"targets.py imports {forbidden}"


def test_the_portable_layer_reuses_the_compactor_rather_than_copying_it():
    """The priority order that survives a small budget came from a benchmark
    result. A second copy of it would drift from the first."""
    from open_context.compiler import portable

    assert "from open_context.hybrid.compactor import" in inspect.getsource(portable)


# ----------------------------------------------------------------------
# Budgeting happens once, in the portable layer


def test_the_compiled_context_stays_inside_its_budget():
    for budget in (60, 120, 400):
        context = compiled(budget=budget)
        assert context.total_tokens <= budget, budget


def test_a_small_budget_drops_the_least_important_state_first():
    """Entities and open questions go before constraints do."""
    context = compiled(budget=60)
    text = context.text_of(Section.STATE)
    assert "Ship the export writer" in text
    assert "Should the writer stream or buffer?" not in text
    assert context.dropped["state_items"] > 0


def test_what_was_dropped_is_part_of_the_result():
    """A context that silently omitted half the state looks exactly like one
    with half as much state."""
    full = compiled(budget=400)
    small = compiled(budget=60)
    assert full.complete
    assert not small.complete
    assert small.dropped


def test_a_task_that_does_not_fit_is_refused_rather_than_dropped():
    """A context without its task is not a smaller context, it is a different one."""
    with pytest.raises(ValueError, match="task alone"):
        compiled(budget=3, task="a very long task statement " * 20)


def test_state_is_capped_not_reserved():
    """State that does not need its share leaves the rest to everything else."""
    recent = RecentContext(messages=({"role": "user", "content": "one more thing about ports"},))
    pkg = package(state=[state()[0]], recent=None)
    small = pkg.model_copy(update={"recent": recent, "content_hash": ""})
    context = compile_context(
        small, CompilationRequest(budget_tokens=200, task="Go.", state_fraction=0.6), TOKENIZER
    )
    assert "one more thing about ports" in context.text_of(Section.RECENT)


def test_recent_context_is_taken_from_the_end():
    """It exists to show what was just happening; dropping the newest turns to
    keep older ones would defeat its only purpose."""
    messages = tuple(
        {"role": "user", "content": f"message number {n} about the export work"} for n in range(20)
    )
    pkg = package().model_copy(
        update={"recent": RecentContext(messages=messages), "content_hash": ""}
    )
    context = compile_context(pkg, CompilationRequest(budget_tokens=200, task="Go."), TOKENIZER)
    text = context.text_of(Section.RECENT)
    assert "message number 19" in text
    assert "message number 0 " not in text


def test_evidence_can_be_left_out():
    assert compiled(include_evidence=False).text_of(Section.EVIDENCE) == ""


def test_recent_can_be_left_out():
    messages = ({"role": "user", "content": "the last thing said"},)
    pkg = package().model_copy(
        update={"recent": RecentContext(messages=messages), "content_hash": ""}
    )
    context = compile_context(
        pkg, CompilationRequest(budget_tokens=400, task="Go.", include_recent=False), TOKENIZER
    )
    assert context.text_of(Section.RECENT) == ""


def test_the_task_is_placed_last():
    """It is the most recent thing the model reads."""
    context = compiled()
    assert context.sections[-1].section is Section.TASK


# ----------------------------------------------------------------------
# Neutrality


def test_naming_a_target_model_does_not_change_what_is_compiled():
    """A compiler that wrote different content per model would make the
    package's provider-neutrality a fiction."""
    pkg = package()
    a = compile_context(
        pkg, CompilationRequest(budget_tokens=400, task="Go.", target_model="gpt-5"), TOKENIZER
    )
    b = compile_context(
        pkg,
        CompilationRequest(budget_tokens=400, task="Go.", target_model="claude-opus-5"),
        TOKENIZER,
    )
    assert [s.text for s in a.sections] == [s.text for s in b.sections]
    assert a.target_model != b.target_model


def test_a_tampered_package_compiles_but_says_so():
    """Refusing outright would leave a caller with no way to inspect what they
    were given; compiling silently would launder it."""
    pkg = package()
    tampered = pkg.model_copy(update={"task": None, "target_model": ""}, deep=True)
    tampered = type(pkg).model_validate(json.loads(pkg.to_json()) | {"content_hash": "0" * 64})
    context = compile_context(tampered, CompilationRequest(budget_tokens=400), TOKENIZER)
    assert any("content hash" in warning for warning in context.warnings)


def test_a_budget_must_be_positive():
    with pytest.raises(ValueError, match="budget must be positive"):
        CompilationRequest(budget_tokens=0)


def test_the_state_share_must_be_a_fraction():
    with pytest.raises(ValueError, match="state_fraction"):
        CompilationRequest(budget_tokens=100, state_fraction=1.5)


# ----------------------------------------------------------------------
# V2.3: state inherited from other sessions on the same project


def inherited_package():
    from open_context.models.state import Constraint

    pkg = package(state=[state()[0]])
    rule = Constraint(
        id=ids.new_id(ids.CONSTRAINT),
        session_id=SESSION,
        content="Never write report files to the shared NFS mount",
    )
    return type(pkg).model_validate(
        json.loads(pkg.to_json())
        | {"inherited": [rule.model_dump(mode="json")], "content_hash": ""}
    )


def test_inherited_state_reaches_the_compiled_context():
    context = compile_context(
        inherited_package(), CompilationRequest(budget_tokens=400, task="Go."), TOKENIZER
    )
    assert "Never write report files" in context.text_of(Section.INHERITED)


def test_inherited_state_says_it_came_from_elsewhere():
    """ "We decided X" and "another session decided X" are both worth carrying and
    are not the same sentence. A reader who cannot tell them apart cannot judge
    which to revisit."""
    context = compile_context(
        inherited_package(), CompilationRequest(budget_tokens=400, task="Go."), TOKENIZER
    )
    text = context.text_of(Section.INHERITED)
    assert "other sessions" in text
    assert "Never write report files" not in context.text_of(Section.STATE)


def test_this_sessions_own_state_is_read_first():
    context = compile_context(
        inherited_package(), CompilationRequest(budget_tokens=400, task="Go."), TOKENIZER
    )
    order = [section.section for section in context.sections]
    assert order.index(Section.STATE) < order.index(Section.INHERITED)


def test_a_budget_with_no_room_drops_inherited_before_this_sessions_own():
    """What this session concluded outranks what it inherited.

    The frame is paid for first and is not small, so the budget here is chosen
    to leave room for one section and not two — which is exactly the case the
    ordering exists to decide.
    """
    from open_context.models.state import Constraint

    pkg = package(state=[state()[0]])
    bulky = Constraint(
        id=ids.new_id(ids.CONSTRAINT),
        session_id=SESSION,
        content="a rule stated at considerable length " * 40,
    )
    pkg = type(pkg).model_validate(
        json.loads(pkg.to_json())
        | {"inherited": [bulky.model_dump(mode="json")], "content_hash": ""}
    )

    context = compile_context(
        pkg, CompilationRequest(budget_tokens=40, task="Go.", frame_as_data=False), TOKENIZER
    )
    assert context.text_of(Section.STATE), "this session's own state survives"
    assert context.text_of(Section.INHERITED) == ""
    assert context.dropped.get("inherited_items") == 1
