"""The installed console script must work outside the source tree.

`pramaan` is the entry point every document in this repo tells a reader
to run. It was broken: the wheel shipped only `src/pramaan`, while
installed library code imports `benchmarks` and `eval` at module level,
so `pramaan data` raised ModuleNotFoundError on any machine that had
merely installed the package.

Nothing caught it because every invocation in CI and in development went
through `python -m pramaan.cli` *from the repo root*, and `python -m`
puts the working directory on `sys.path`. The repo checkout was
standing in for the installed distribution.

These tests run from a temporary directory precisely so that the
working directory cannot supply the packages.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _console_script() -> str | None:
    """Find `pramaan` next to the running interpreter, then fall back to PATH.

    `shutil.which` alone is not enough: a venv that has not been
    "activated" in the calling shell installs the script into
    `<venv>/Scripts` (or `bin`) without putting it on PATH, so which()
    returns None and the test skips. A skip here looks identical to a
    pass, on exactly the tests guarding a bug that already shipped once.
    """
    bindir = Path(sys.executable).parent
    for name in ("pramaan.exe", "pramaan"):
        candidate = bindir / name
        if candidate.exists():
            return str(candidate)
    return shutil.which("pramaan")


CONSOLE_SCRIPT = _console_script()


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
    )


@pytest.mark.parametrize(
    "module",
    [
        # Imported at module level by installed code, so a wheel without
        # them cannot even be imported, let alone run.
        "benchmarks.loaders",
        "eval.run_eval",
        "pramaan.fusion.pipeline",
        "pramaan.risk.certify_pipeline",
    ],
)
def test_runtime_imports_resolve_outside_the_repo(module: str, tmp_path: Path) -> None:
    proc = _run([sys.executable, "-c", f"import {module}"], cwd=tmp_path)
    assert proc.returncode == 0, (
        f"`import {module}` failed from {tmp_path}, i.e. for anyone who "
        f"installed this package rather than cloning it:\n{proc.stderr}"
    )


@pytest.mark.skipif(CONSOLE_SCRIPT is None, reason="package not installed")
def test_console_script_help_works_outside_the_repo(tmp_path: Path) -> None:
    assert CONSOLE_SCRIPT is not None
    proc = _run([CONSOLE_SCRIPT, "--help"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "pramaan" in proc.stdout.lower()


@pytest.mark.skipif(CONSOLE_SCRIPT is None, reason="package not installed")
def test_console_script_subcommands_are_wired(tmp_path: Path) -> None:
    """Checks the CLI surface, NOT the packaging bug above.

    Worth being exact about, because the obvious assumption is wrong:
    re-introducing the packaging bug leaves this test **green**. Typer
    renders a subcommand's help without executing its body, and the
    failing imports are inside those bodies, so `pramaan data --help`
    succeeds against a wheel where `pramaan data` cannot run at all.

    The `test_runtime_imports_*` cases above are what actually catch it
    -- verified by reverting the fix and watching exactly those four
    fail while these two passed.
    """
    assert CONSOLE_SCRIPT is not None
    for command in ("data", "eval", "train"):
        proc = _run([CONSOLE_SCRIPT, command, "--help"], cwd=tmp_path)
        assert proc.returncode == 0, (
            f"`pramaan {command} --help` failed outside the repo:\n{proc.stderr}"
        )
        assert "ModuleNotFoundError" not in proc.stderr
