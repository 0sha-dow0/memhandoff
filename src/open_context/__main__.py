"""Take a long agent session and hand it to a different agent.

    open-context sessions
    open-context handoff  <transcript>.jsonl --store ./ctx --out project.ctx
    open-context inspect  project.ctx
    open-context validate project.ctx --archive ./ctx
    open-context compile  project.ctx --target anthropic --budget 4000 --task "..."

`handoff` reads a session, archives it losslessly, and writes a portable `.ctx`
package. `compile` turns that package into context in one provider family's
request shape. `pack` builds a package from an already-imported session instead
of a transcript.

`validate` exits 0 when a package is valid, 1 when it is not, and 2 when the
file could not be read as a package at all — so a script can tell "this is
wrong" from "this is not one of these".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from open_context.archive.records import ArchiveRecord
from open_context.compiler import GENERIC, known_targets
from open_context.package import CtxPackage, build, render, validate
from open_context.package.validation import Depth

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_UNUSABLE = 2
"""Distinct from ``EXIT_INVALID``: the file could not be read as a package at
all, which is a different problem from a package that read fine and failed."""


def _load(path: Path) -> CtxPackage:
    return CtxPackage.from_json(path.read_text(encoding="utf-8"))


def _read_archive(directory: Path, session_id: str) -> list[ArchiveRecord]:
    """Every record for one session, in order. Empty when the archive has none."""
    from open_context.archive import Archive

    archive = Archive(directory)
    if not archive.exists(session_id):
        return []
    with archive.open(session_id) as log:
        return list(log)


def _archive_records(directory: Path | None, session_id: str) -> dict[int, ArchiveRecord] | None:
    """Records from an archive on disk, or nothing if none was named.

    Returning ``None`` rather than an empty dict is what tells the validator it
    could not check provenance, as opposed to checking it against an archive
    that happens to be empty. The two produce very different verdicts, and
    collapsing them would let "I did not look" be reported as "I looked and it
    was fine".
    """
    if directory is None:
        return None
    return {record.seq: record for record in _read_archive(directory, session_id)}


def command_pack(args: argparse.Namespace) -> int:
    from open_context.storage import Database, Repository

    store = Path(args.store)
    repository = Repository(Database(store / "state.sqlite"))
    state = repository.list_state_items(args.session)

    archive_directory = store / "archive"
    records = _read_archive(archive_directory, args.session) if archive_directory.exists() else []

    package, report = build(
        session_id=args.session,
        state=state,
        archive_records=records or None,
        archive_location=str(archive_directory) if records else "",
        created_by="open_context pack",
    )

    Path(args.out).write_text(package.to_json(), encoding="utf-8")
    print(f"wrote {args.out}: {report.state_items} state items", file=sys.stderr)
    if report.superseded_excluded:
        print(f"  excluded {report.superseded_excluded} superseded items", file=sys.stderr)
    for warning in report.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    return EXIT_OK


def command_validate(args: argparse.Namespace) -> int:
    try:
        package = _load(Path(args.package))
    except (ValueError, OSError) as exc:
        print(f"cannot read {args.package}: {exc}", file=sys.stderr)
        return EXIT_UNUSABLE

    archive = Path(args.archive) if args.archive else None
    report = validate(
        package, archive_records=_archive_records(archive, package.manifest.session_id)
    )
    print(report)
    if report.depth is Depth.STRUCTURE:
        print(
            "\nNo archive was given, so provenance was not checked. This says the "
            "package is\ninternally consistent, not that it describes the archive "
            "it names.",
            file=sys.stderr,
        )
    from open_context.package.security import review

    security = review(package)
    if security.notes:
        print("\nWhat this package carries, for a reader deciding whether to use it:")
        for note in security.notes:
            print(f"  {note}")
        print(
            "  These are remarks, not a verdict. Nothing here detects prompt injection "
            "reliably,\n  and a clean report is not a guarantee of safety."
        )

    return EXIT_OK if report.valid else EXIT_INVALID


def command_inspect(args: argparse.Namespace) -> int:
    try:
        package = _load(Path(args.package))
    except (ValueError, OSError) as exc:
        print(f"cannot read {args.package}: {exc}", file=sys.stderr)
        return EXIT_UNUSABLE
    print(render(package, show_evidence=not args.brief), end="")
    return EXIT_OK


def command_compile(args: argparse.Namespace) -> int:
    from open_context.compiler import CompilationRequest, compile_context, render_for
    from open_context.llm.fakes import WordTokenizer

    try:
        package = _load(Path(args.package))
    except (ValueError, OSError) as exc:
        print(f"cannot read {args.package}: {exc}", file=sys.stderr)
        return EXIT_UNUSABLE

    # A word tokenizer, and the output says so. Counting exactly needs the target
    # model's own tokenizer, which this command has no way to obtain without
    # calling a provider — and compiling a context is not a reason to make a
    # network request. An estimate a caller knows is an estimate is the honest
    # offer; one presented as exact is how a compiled context overflows.
    # Streamed, never listed: retrieval passes records by a bounded heap, so a
    # task can search a session of any size without holding one in memory.
    archive_records = None
    if args.retrieve and args.archive:
        from open_context.archive import Archive

        store = Archive(Path(args.archive))
        session = package.manifest.session_id
        if store.exists(session):
            archive_records = store.open(session)
        else:
            print(
                f"no archived session {session} under {args.archive}; compiling without retrieval",
                file=sys.stderr,
            )

    try:
        context = compile_context(
            package,
            CompilationRequest(
                budget_tokens=args.budget,
                task=args.task or "",
                target_model=args.model or "",
                include_evidence=not args.no_evidence,
                include_recent=not args.no_recent,
                retrieve=args.retrieve,
                retrieval_query=args.query or "",
            ),
            WordTokenizer(exact=False),
            archive_records,
        )
    finally:
        if archive_records is not None:
            archive_records.close()

    print(json.dumps(render_for(args.target, context), indent=2, sort_keys=True))

    print(
        f"{context.total_tokens} tokens (estimated, word-based) of a {args.budget} budget"
        f" for target {args.target!r}",
        file=sys.stderr,
    )
    for name, count in sorted(context.dropped.items()):
        print(f"  dropped {count} {name.replace('_', ' ')}", file=sys.stderr)
    for warning in context.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    return EXIT_OK


def command_handoff(args: argparse.Namespace) -> int:
    """The whole workflow: a session file in, a `.ctx` out."""
    from open_context.handoff import handoff

    provider = None
    if args.extract:
        import open_context.llm.providers  # noqa: F401  registers groq and openrouter
        from open_context.llm import create_provider, from_env

        try:
            provider = create_provider(from_env())
        except Exception as exc:
            print(
                f"--extract needs a configured provider and none is available: {exc}\n"
                f"Set OPEN_CONTEXT_PROVIDER, OPEN_CONTEXT_MODEL, and OPEN_CONTEXT_API_KEY, "
                f"or drop --extract to package recent context and provenance only.",
                file=sys.stderr,
            )
            return EXIT_UNUSABLE

    try:
        package, report = handoff(
            source=args.source,
            session_id=args.session or _session_id_for(args.source),
            archive_root=args.store,
            title=args.title,
            include_recent=not args.no_recent,
            provider=provider,
        )
    except (OSError, ValueError) as exc:
        print(f"cannot hand off {args.source}: {exc}", file=sys.stderr)
        return EXIT_UNUSABLE

    Path(args.out).write_text(package.to_json(), encoding="utf-8")
    print(report.summary(), file=sys.stderr)
    print(f"  wrote     {args.out}", file=sys.stderr)
    return EXIT_OK


def _session_id_for(source: str) -> str:
    """A session id derived from the transcript's own name.

    Claude Code names each file after its session uuid, so the obvious id is
    already there. It is prefixed rather than used raw, because this project's
    identifiers are self-describing and a bare uuid says nothing about what it
    identifies.
    """
    from open_context.models import ids

    stem = Path(source).stem.replace("-", "")[:24]
    return f"{ids.SESSION}_{stem}" if stem else ids.new_id(ids.SESSION)


def command_sessions(args: argparse.Namespace) -> int:
    """List transcripts this machine already has.

    A convenience for a person who wants to hand one off and does not know where
    the agent keeps them. The location is an observed fact about one tool's
    current layout, not an interface, and nothing else depends on it.
    """
    from open_context.importers.claude_code import find_sessions

    found = find_sessions(args.root)
    if not found:
        print("no Claude Code sessions found on this machine", file=sys.stderr)
        return EXIT_OK
    for path in found[: args.limit]:
        size = path.stat().st_size
        print(f"{path}  ({size // 1024} KiB)")
    return EXIT_OK


def command_hook(args: argparse.Namespace) -> int:
    """Receive an agent hook event on stdin and act on it.

    Wired into Claude Code's `PreCompact` hook, this captures a package at the
    moment the agent decides its window is full — the last instant the whole
    conversation exists.

    **Always exits 0.** The hook protocol treats a non-zero exit as a reason to
    interfere with the agent, and a failed backup is not a reason to do that.
    """
    from open_context.pressure import HookEvent, PressureConfig, handle

    event = HookEvent.parse(sys.stdin.read())
    response, note = handle(event, PressureConfig(store=Path(args.store)))
    print(response.to_json())
    print(f"open-context: {note}", file=sys.stderr)
    return EXIT_OK


def command_verify_models(args: argparse.Namespace) -> int:
    """Re-check the free-model allowlist against the live catalogues.

    Exits non-zero when an approved model is now priced, so this can gate a
    release. A model that has *vanished* is reported and does not fail the
    check: it cannot be called, so it cannot cost anything.
    """
    import os

    from open_context.llm.providers.verify import verify

    keys = {
        "openrouter": os.environ.get("OPENROUTER_API_KEY"),
        "groq": os.environ.get("GROQ_API_KEY"),
    }
    report = verify(keys)
    print(report)
    return EXIT_OK if report.sound else EXIT_INVALID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-context",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="build a .ctx package from a stored session")
    pack.add_argument("--session", required=True, help="session id to package")
    pack.add_argument("--store", required=True, help="directory holding state.sqlite and archive/")
    pack.add_argument("--out", required=True, help="path to write the .ctx file to")
    pack.set_defaults(handler=command_pack)

    check = sub.add_parser("validate", help="check a .ctx package; non-zero exit when invalid")
    check.add_argument("package")
    check.add_argument(
        "--archive",
        help="archive directory, to check provenance as well as structure",
    )
    check.set_defaults(handler=command_validate)

    show = sub.add_parser("inspect", help="print a .ctx package for a person to read")
    show.add_argument("package")
    show.add_argument("--brief", action="store_true", help="omit evidence excerpts")
    show.set_defaults(handler=command_inspect)

    build_context = sub.add_parser(
        "compile", help="compile a .ctx package into context for a target agent"
    )
    build_context.add_argument("package")
    build_context.add_argument(
        "--target",
        default=GENERIC,
        help=f"provider family: {', '.join(known_targets())}. Unknown targets fall back to generic",
    )
    build_context.add_argument("--budget", type=int, required=True, help="token budget")
    build_context.add_argument("--task", help="what the receiving agent is being asked to do")
    build_context.add_argument("--model", help="target model, recorded but never acted on")
    build_context.add_argument(
        "--retrieve",
        type=int,
        default=0,
        metavar="N",
        help="pull up to N relevant records back from the archive for this task. "
        "Needs --archive. Off by default: a package received from elsewhere has no "
        "archive to search",
    )
    build_context.add_argument(
        "--archive", help="archive directory to retrieve from, used with --retrieve"
    )
    build_context.add_argument(
        "--query", help="what to search the archive for; defaults to the task"
    )
    build_context.add_argument("--no-evidence", action="store_true")
    build_context.add_argument("--no-recent", action="store_true")
    build_context.set_defaults(handler=command_compile)

    hand = sub.add_parser(
        "handoff", help="read an agent session, archive it, and write a .ctx package"
    )
    hand.add_argument("source", help="a session transcript, such as a Claude Code .jsonl")
    hand.add_argument("--store", required=True, help="archive directory to write into")
    hand.add_argument("--out", required=True, help="path to write the .ctx file to")
    hand.add_argument("--session", help="session id; derived from the file name when omitted")
    hand.add_argument("--title", help="a human-readable name for the work")
    hand.add_argument("--no-recent", action="store_true", help="omit verbatim recent turns")
    hand.add_argument(
        "--extract",
        action="store_true",
        help="ask the configured model for goals, constraints, and decisions. "
        "Needs OPEN_CONTEXT_PROVIDER/_MODEL/_API_KEY; without it the package "
        "carries recent context and provenance only",
    )
    hand.set_defaults(handler=command_handoff)

    hook = sub.add_parser(
        "hook",
        help="receive an agent hook event on stdin; capture a package under context pressure",
    )
    hook.add_argument("--store", required=True, help="directory to keep archives and packages in")
    hook.set_defaults(handler=command_hook)

    checking = sub.add_parser(
        "verify-models",
        help="re-check the free-model allowlist against the providers' catalogues",
    )
    checking.set_defaults(handler=command_verify_models)

    listing = sub.add_parser("sessions", help="list agent transcripts found on this machine")
    listing.add_argument("--root", help="where to look; defaults to the agent's own location")
    listing.add_argument("--limit", type=int, default=20)
    listing.set_defaults(handler=command_sessions)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    assert callable(handler)
    result = handler(args)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
