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


def test_commands_are_invocable_and_fail_informatively() -> None:
    """Every command must respond to invocation with either success or a
    clear, actionable error - never an unhandled traceback.

    This deliberately does NOT require exit code 0. Commands that have
    landed (`train`, `certify`) do real work and correctly refuse when no
    corpus has been built; commands still pending exit 0 with a "lands in
    Phase N" notice. An earlier version asserted 0 for all of them and
    passed locally purely because this machine has a dev corpus - CI,
    which has no `data/` (gitignored), failed. The test was encoding local
    state rather than a property of the CLI.
    """
    from benchmarks.loaders import CorpusNotBuiltError
    from pramaan.fusion.model import ModelNotTrainedError

    for args in (["train"], ["certify"], ["eval"], ["report"], ["serve"]):
        result = runner.invoke(app, args)
        if result.exit_code == 0:
            continue
        # A missing corpus is a legitimate, expected state. Anything else
        # is a genuine crash.
        expected = CorpusNotBuiltError | ModelNotTrainedError | SystemExit
        assert isinstance(result.exception, expected), (
            f"{args} raised {type(result.exception).__name__}: {result.exception}"
        )
        if isinstance(result.exception, CorpusNotBuiltError | ModelNotTrainedError):
            message = str(result.exception)
            assert "pramaan " in message, (
                f"{args} failed without telling the user how to fix it: {message}"
            )


def test_data_rejects_full_scale() -> None:
    result = runner.invoke(app, ["data", "--scale", "full"])
    assert result.exit_code == 1
    assert "data-full" in result.output


def test_all_full_scale_is_rejected() -> None:
    result = runner.invoke(app, ["all", "--scale", "full"])
    assert result.exit_code == 1


def test_all_passes_real_values_not_typer_sentinels(monkeypatch) -> None:
    """`all` must call the plain implementations, not the @app.command
    wrappers. Invoking a Typer command as an ordinary function passes its
    `typer.Option(...)` sentinel through as the argument value - which
    reached random.seed() as an OptionInfo and blew up in CI the first
    time `all` did real work. Local runs never caught it because `data`
    was only ever invoked through the CLI.
    """
    from pramaan import cli

    captured: dict[str, object] = {}

    def fake_build(tier: str, seed: int) -> None:
        captured["tier"] = tier
        captured["seed"] = seed

    monkeypatch.setattr(cli, "_build_corpus", fake_build)
    monkeypatch.setattr(cli, "_run_setup", lambda: True)
    monkeypatch.setattr(cli, "_run_train", lambda tier, use_clip: None)

    result = runner.invoke(app, ["all", "--scale", "smoke"])
    assert result.exit_code == 0, result.output
    assert captured["tier"] == "smoke"
    assert isinstance(captured["seed"], int)
    assert captured["seed"] == cli.DEFAULT_SEED


def test_all_forwards_an_explicit_seed(monkeypatch) -> None:
    from pramaan import cli

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli, "_build_corpus", lambda tier, seed: captured.update(tier=tier, seed=seed)
    )
    monkeypatch.setattr(cli, "_run_setup", lambda: True)
    monkeypatch.setattr(cli, "_run_train", lambda tier, use_clip: None)

    result = runner.invoke(app, ["all", "--scale", "smoke", "--seed", "99"])
    assert result.exit_code == 0, result.output
    assert captured["seed"] == 99
