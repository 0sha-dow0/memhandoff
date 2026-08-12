"""Phase 0 smoke tests.

These assert only that the package is installed, importable, and typed.
There is no application logic to test yet.
"""

import importlib.metadata

import open_context


def test_package_imports() -> None:
    assert open_context.__name__ == "open_context"


def test_version_is_exposed() -> None:
    assert isinstance(open_context.__version__, str)
    assert open_context.__version__


def test_installed_version_matches_package_version() -> None:
    installed = importlib.metadata.version("open-context-runtime")
    assert installed == open_context.__version__


def test_package_ships_py_typed() -> None:
    from pathlib import Path

    marker = Path(open_context.__file__).parent / "py.typed"
    assert marker.is_file()
