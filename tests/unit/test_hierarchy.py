"""V2.3: project, session, event — the logical model only.

The exit criterion is that hierarchy improves useful context selection without
unnecessary infrastructure. Both halves are pinned here: what a project level
lets a session see that it otherwise could not, and the absence of any storage
engine to make that work.
"""

import inspect

from open_context.hierarchy import (
    MARKERS,
    Level,
    Project,
    group_by_project,
    place,
    project_id,
    project_root,
)
from open_context.models import ids
from open_context.models.enums import RetentionClass
from open_context.models.state import Constraint, Decision, Fact, Goal

SESSION_A = "ses_" + "ab" * 12
SESSION_B = "ses_" + "cd" * 12


def constraint(session=SESSION_A, content="Never write to the shared NFS mount", **kwargs):
    return Constraint(id=ids.new_id(ids.CONSTRAINT), session_id=session, content=content, **kwargs)


def fact(session=SESSION_A, content="The cursor is on line 40", **kwargs):
    return Fact(id=ids.new_id(ids.FACT), session_id=session, content=content, **kwargs)


# ----------------------------------------------------------------------
# Project identity is a repository, not a working directory


def test_a_repository_root_gathers_the_directories_beneath_it(tmp_path):
    """The failure this prevents, seen on real data: two of six sessions ran in
    `open-context` and `open-context/files (4)/open-context-runtime 2`, which is
    one project and a directory inside it. Splitting them hides a session's own
    history from itself.
    """
    root = tmp_path / "myrepo"
    (root / ".git").mkdir(parents=True)
    (root / "backend" / "api").mkdir(parents=True)

    assert project_id(root) == project_id(root / "backend") == project_id(root / "backend" / "api")


def test_every_marker_is_recognised(tmp_path):
    for index, marker in enumerate(MARKERS):
        root = tmp_path / f"r{index}"
        (root / marker).mkdir(parents=True)
        (root / "sub").mkdir()
        assert project_root(root / "sub") == root


def test_a_directory_in_no_repository_is_its_own_project(tmp_path):
    """Attaching it to whatever sits above would put unrelated work together.

    This is the common case on the machine these were measured on: none of the
    six real session directories is a repository, so each is correctly its own
    project.
    """
    lonely = tmp_path / "scratch"
    lonely.mkdir()
    assert project_root(lonely) == lonely


def test_project_identity_is_answered_by_looking_not_by_parsing():
    """A path that merely resembles a project name proves nothing about one."""
    source = inspect.getsource(project_root)
    assert ".exists()" in source
    assert "split" not in source and "endswith" not in source


# ----------------------------------------------------------------------
# Levels, and what they are for


def test_a_constraint_outlives_its_session_and_a_passing_detail_does_not():
    assert place(constraint()).level is Level.PROJECT
    assert place(fact()).level is Level.SESSION


def test_goals_and_decisions_are_project_scoped():
    goal = Goal(id=ids.new_id(ids.GOAL), session_id=SESSION_A, content="Ship the writer")
    decision = Decision(
        id=ids.new_id(ids.DECISION),
        session_id=SESSION_A,
        content="Write UTF-16LE",
        rationale="the loader rejects UTF-8",
    )
    assert place(goal).project_scoped
    assert place(decision).project_scoped


def test_an_item_that_states_its_own_importance_is_believed():
    """A retention set by extraction or by a person is a judgement about *this*
    item; the type-based guess is about its kind in general.

    A critical item must name a source — the data model refuses one that does
    not, on the ground that the most load-bearing claims are exactly the ones
    that have to be checkable.
    """
    urgent = fact(retention=RetentionClass.CRITICAL, sources=["msg_" + "1" * 24])
    assert place(urgent).level is Level.PROJECT

    passing = constraint(retention=RetentionClass.EPHEMERAL)
    assert place(passing).level is Level.EVENT


def test_each_level_has_its_own_retention():
    assert place(constraint()).retention is RetentionClass.CRITICAL
    assert place(fact()).retention is RetentionClass.IMPORTANT


# ----------------------------------------------------------------------
# What the project level actually buys — the exit criterion


def test_a_session_sees_what_a_sibling_session_established():
    """Monday's constraint is still true on Tuesday. A session that could only
    see itself would ask again, or decide differently."""
    project = Project(root="/work/project")
    project.add(SESSION_A, [constraint(SESSION_A), fact(SESSION_A)])
    project.add(SESSION_B, [fact(SESSION_B, content="Tuesday's detail")])

    carried = project.carried_forward(exclude=SESSION_B)
    assert [item.content for item in carried] == ["Never write to the shared NFS mount"]


def test_only_project_scoped_state_is_carried_between_sessions():
    """Carrying everything would make the second session read the first one
    whole, which is the thing this project exists to avoid."""
    project = Project(root="/work/project")
    project.add(SESSION_A, [constraint(SESSION_A), fact(SESSION_A), fact(SESSION_A)])

    carried = project.carried_forward(exclude=SESSION_B)
    assert len(carried) == 1


def test_a_session_is_not_handed_back_its_own_state():
    project = Project(root="/work/project")
    project.add(SESSION_A, [constraint(SESSION_A)])
    assert project.carried_forward(exclude=SESSION_A) == []


def test_the_same_constraint_restated_in_three_sessions_is_carried_once():
    """A receiving agent shown it three times learns nothing the third time."""
    project = Project(root="/work/project")
    for session in (SESSION_A, SESSION_B, "ses_" + "ef" * 12):
        project.add(session, [constraint(session)])

    assert len(project.carried_forward(exclude="ses_" + "99" * 12)) == 1


def test_sessions_group_into_projects_by_where_they_ran(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "web").mkdir()
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()

    projects = group_by_project(
        [
            (SESSION_A, str(root), [constraint(SESSION_A)]),
            (SESSION_B, str(root / "web"), [fact(SESSION_B)]),
            ("ses_" + "ef" * 12, str(elsewhere), [fact("ses_" + "ef" * 12)]),
        ]
    )
    assert len(projects) == 2
    assert set(projects[str(root)].sessions) == {SESSION_A, SESSION_B}


def test_a_project_reports_a_readable_name(tmp_path):
    assert Project(root="/work/reporting-service").name == "reporting-service"


# ----------------------------------------------------------------------
# Without unnecessary infrastructure


def test_the_hierarchy_introduces_no_storage_engine():
    """The roadmap is explicit: a graph-shaped model is not a reason to install
    a graph engine. This is three levels and a parent link, which is a dict."""
    from open_context import hierarchy

    source = inspect.getsource(hierarchy)
    for engine in ("neo4j", "networkx", "sqlite3", "redis", "qdrant", "sqlalchemy"):
        assert engine not in source.lower(), f"hierarchy reaches for {engine}"

    tree = inspect.getmodule(hierarchy)
    assert tree is not None
    imported = {name for name in dir(hierarchy) if inspect.ismodule(getattr(hierarchy, name, None))}
    assert not imported - {"inspect"}, f"unexpected module dependencies: {imported}"


def test_the_task_level_is_absent_on_purpose():
    """Nothing in a real transcript marks where one task ends. A boundary
    invented here would look exactly like one that was observed."""
    assert {level.value for level in Level} == {"project", "session", "event"}
