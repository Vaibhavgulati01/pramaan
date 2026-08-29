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


def test_commands_are_invocable_and_fail_informatively(monkeypatch, tmp_path) -> None:
    """Every command must respond to invocation with either success or a
    clear, actionable error - never an unhandled traceback.

    This deliberately does NOT require exit code 0. Commands that have
    landed do real work and correctly refuse when no corpus has been
    built. An earlier version asserted 0 for all of them and passed
    locally purely because this machine has a dev corpus - CI, which has
    no `data/` (gitignored), failed. The test was encoding local state
    rather than a property of the CLI.

    The repo root is redirected to an empty directory so the "no corpus"
    branch is taken *deterministically*. Without that, the outcome
    depended on whether the machine running the tests happened to have a
    dev corpus lying around: on this one it did, so the test quietly ran
    a full dev train + certify + eval + README rewrite -- minutes of real
    work, and a mutation of a tracked file, inside a unit test.
    """
    from benchmarks.loaders import CorpusNotBuiltError
    from pramaan import cli
    from pramaan.fusion.model import ModelNotTrainedError

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)

    for args in (["train"], ["certify"], ["eval"], ["report"]):
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


def test_serve_starts_the_server_without_this_test_starting_one(monkeypatch) -> None:
    """`serve` is excluded from the loop above, on purpose.

    It calls `uvicorn.run()`, which binds a port and blocks forever. The
    previous version of that loop invoked `["serve"]` directly, so the
    whole test suite's runtime depended on whether port 8000 happened to
    be occupied: busy meant a fast SystemExit and a green run, free meant
    the suite hung indefinitely. It passed for weeks and then stopped,
    with no change to the code under test.

    Here uvicorn is stubbed, so what is asserted is the wiring.
    """
    import uvicorn

    called: dict[str, object] = {}

    def fake_run(target: str, **kwargs: object) -> None:
        called["target"] = target
        called.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = runner.invoke(app, ["serve", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert called["target"] == "pramaan.api.app:app"
    assert called["port"] == 9999


def test_data_rejects_full_scale() -> None:
    result = runner.invoke(app, ["data", "--scale", "full"])
    assert result.exit_code == 1
    assert "data-full" in result.output


def test_all_full_scale_is_rejected() -> None:
    result = runner.invoke(app, ["all", "--scale", "full"])
    assert result.exit_code == 1


def _stub_every_stage(monkeypatch, order: list[str], captured: dict[str, object]):
    """Replace every stage `all` runs with a recorder.

    Stubbing *all* of them is load-bearing, not tidiness. These tests
    previously stubbed only setup/data/train, so when `all` was fixed to
    stop skipping certify and eval, the tests silently began running the
    real pipeline -- turning a sub-second unit test into a multi-minute
    one that refits models. A test that does real work by accident is a
    test that will be deleted by whoever is next in a hurry.
    """
    from pramaan import cli

    def record(name: str):
        def stage(*args: object, **kwargs: object) -> None:
            order.append(name)
        return stage

    def fake_build(tier: str, seed: int) -> None:
        order.append("data")
        captured["tier"] = tier
        captured["seed"] = seed

    monkeypatch.setattr(cli, "_run_setup", lambda: order.append("setup") or True)
    monkeypatch.setattr(cli, "_build_corpus", fake_build)
    monkeypatch.setattr(cli, "_run_train", record("train"))
    monkeypatch.setattr(cli, "_run_certify", record("certify"))
    monkeypatch.setattr(cli, "_run_eval", record("eval"))
    monkeypatch.setattr(cli, "_run_report", record("report"))
    return cli


def test_all_passes_real_values_not_typer_sentinels(monkeypatch) -> None:
    """`all` must call the plain implementations, not the @app.command
    wrappers. Invoking a Typer command as an ordinary function passes its
    `typer.Option(...)` sentinel through as the argument value - which
    reached random.seed() as an OptionInfo and blew up in CI the first
    time `all` did real work. Local runs never caught it because `data`
    was only ever invoked through the CLI.
    """
    order: list[str] = []
    captured: dict[str, object] = {}
    cli = _stub_every_stage(monkeypatch, order, captured)

    result = runner.invoke(app, ["all", "--scale", "smoke"])
    assert result.exit_code == 0, result.output
    assert captured["tier"] == "smoke"
    assert isinstance(captured["seed"], int)
    assert captured["seed"] == cli.DEFAULT_SEED


def test_all_runs_every_stage_in_cost_order(monkeypatch) -> None:
    """The composition itself, at runtime.

    `all` shipped for four phases calling a "not implemented yet" stub
    instead of certify/eval/report, and CI stayed green the whole time
    because the stub wrote to stderr and exited zero.
    """
    order: list[str] = []
    _stub_every_stage(monkeypatch, order, {})

    result = runner.invoke(app, ["all", "--scale", "dev"])
    assert result.exit_code == 0, result.output
    assert order == ["setup", "data", "train", "certify", "eval", "report"], (
        f"`pramaan all --scale dev` ran {order}"
    )


def test_all_skips_report_at_smoke_scale(monkeypatch) -> None:
    """smoke is never reportable, and `report` rewrites README.md in place.

    CI runs `pramaan all --scale smoke` on every push, so wiring report in
    unconditionally would have CI overwrite the committed README with
    CI-budget numbers.
    """
    order: list[str] = []
    _stub_every_stage(monkeypatch, order, {})

    result = runner.invoke(app, ["all", "--scale", "smoke"])
    assert result.exit_code == 0, result.output
    assert "report" not in order
    assert order == ["setup", "data", "train", "certify", "eval"]
    assert "smoke is never reportable" in result.output


def test_all_forwards_an_explicit_seed(monkeypatch) -> None:
    order: list[str] = []
    captured: dict[str, object] = {}
    _stub_every_stage(monkeypatch, order, captured)

    result = runner.invoke(app, ["all", "--scale", "smoke", "--seed", "99"])
    assert result.exit_code == 0, result.output
    assert captured["seed"] == 99
