"""Distribution smoke tests."""

import importlib.metadata
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import open_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports() -> None:
    assert open_context.__name__ == "open_context"


def test_version_is_exposed() -> None:
    assert isinstance(open_context.__version__, str)
    assert open_context.__version__


def test_installed_version_matches_package_version() -> None:
    installed = importlib.metadata.version("open-context-runtime")
    assert installed == open_context.__version__


def test_package_ships_py_typed() -> None:
    marker = Path(open_context.__file__).parent / "py.typed"
    assert marker.is_file()


def test_sdist_excludes_internal_benchmark_artifacts(tmp_path: Path) -> None:
    """Ignored local benchmark runs must not leak into a published sdist."""
    source = tmp_path / "source"
    shutil.copytree(
        PROJECT_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "build",
            "dist",
        ),
    )
    private_result = source / "benchmarks" / "reports" / "private-sentinel.txt"
    private_result.parent.mkdir(parents=True, exist_ok=True)
    private_result.write_text("must not ship\n", encoding="utf-8")

    output = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )

    [sdist] = output.glob("*.tar.gz")
    with tarfile.open(sdist, "r:gz") as archive:
        members = [PurePosixPath(member.name) for member in archive.getmembers() if member.isfile()]

    roots = {member.parts[0] for member in members}
    assert len(roots) == 1
    [root] = roots
    shipped = {member.relative_to(root).as_posix() for member in members}

    public_benchmark_files = {
        "benchmarks/.gitignore",
        "benchmarks/README.md",
        "benchmarks/plot.py",
        "benchmarks/results/v4-adversarial-8b.jsonl",
    }
    assert {path for path in shipped if path.startswith("benchmarks/")} == public_benchmark_files
    assert "benchmarks/reports/private-sentinel.txt" not in shipped
    assert not any("__pycache__" in PurePosixPath(path).parts for path in shipped)
    assert {
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "src/open_context/__init__.py",
        "src/open_context_eval/__init__.py",
        "tests/test_package.py",
    } <= shipped
