"""Open Context Runtime.

A local-first context runtime for AI agents.

The runtime treats a model's context window as a working set rather than as
memory. Raw history lives in an archive, durable semantic state lives in a
state graph, and the context compiler assembles a task-specific context for
whichever model is being called.
"""

from __future__ import annotations

from importlib import metadata

#: The name this package is distributed under, which is not the import name.
_DISTRIBUTION = "open-context-runtime"


def _detect_version() -> str:
    """The version of this package, with one authoritative source.

    That source is the ``version`` field in ``pyproject.toml``. The build
    backend stamps it into the distribution metadata, and an installed package
    is read back from there rather than from a second string kept in sync by
    hand — the two drifted apart once already, and the copy in this file went
    on claiming ``0.0.0`` after the project had shipped ``0.1.0``.

    An uninstalled source checkout has no metadata to read, so the same field
    is read straight out of ``pyproject.toml``. That keeps a bare checkout
    honest instead of having it report a version the project never released.
    """
    try:
        return metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        pass

    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        # Neither installed nor beside a readable pyproject: say so, rather
        # than inventing a number that looks like a real release.
        return "0+unknown"


__version__ = _detect_version()

__all__ = ["__version__"]
