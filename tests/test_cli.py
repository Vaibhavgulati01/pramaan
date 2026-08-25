"""Phase 0 smoke test: the CLI imports and every command is invocable
without crashing, even before its phase has landed (see _pending() in
src/pramaan/cli.py). This is the guard against the "forgotten CLI command
that errors on first invocation" failure mode.
"""

from typer.testing import CliRunner

from pramaan.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PRAMAAN" in result.output


def test_setup_runs() -> None:
    result = runner.invoke(app, ["setup"])
    # Exit code depends on whether every required dep is importable in this
    # environment; either way it must not crash with an unhandled traceback.
    assert result.exit_code in (0, 1)
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_pending_commands_do_not_crash() -> None:
    for args in (
        ["data", "--scale", "dev"],
        ["data-full"],
        ["train"],
        ["eval"],
        ["report"],
        ["serve"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{args} crashed: {result.output}"


def test_all_full_scale_is_rejected() -> None:
    result = runner.invoke(app, ["all", "--scale", "full"])
    assert result.exit_code == 1
