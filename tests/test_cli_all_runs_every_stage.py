"""`pramaan all` must actually run every stage it advertises.

It did not. Long after Phase 6 shipped working `certify`, `eval` and
`report` implementations, `all` still called a `_pending()` stub that
printed "not implemented yet - it lands in Phase 6" for the last two
stages and returned. CI ran `pramaan all --scale smoke` on every push
and stayed green throughout, because the stub wrote to stderr and exited
zero.

So the flagship command was a partial no-op, and the failure mode was
specifically that *nothing failed*. These tests assert the composition
directly rather than trusting an exit code.
"""

from __future__ import annotations

import inspect

from pramaan import cli


def test_all_calls_every_pipeline_stage() -> None:
    source = inspect.getsource(cli.all_)
    for stage in ("_run_setup", "_build_corpus", "_run_train", "_run_certify", "_run_eval"):
        assert stage in source, (
            f"`pramaan all` never calls {stage}. A stage silently missing from the "
            f"flagship command is exactly the bug this test exists for."
        )


def test_all_never_calls_the_typer_wrappers() -> None:
    """Calling an @app.command as a function passes Option sentinels through."""
    source = inspect.getsource(cli.all_)
    for wrapper in ("setup(", "certify(", "eval_(", "report(", "data("):
        assert f" {wrapper}" not in source and f"={wrapper}" not in source


def test_all_does_not_publish_smoke_numbers_to_the_readme() -> None:
    """smoke is never reportable, and `report` rewrites README.md in place.

    CI runs `pramaan all --scale smoke`, so wiring `report` in
    unconditionally would have had CI overwrite the committed README with
    CI-budget numbers on every push.
    """
    source = inspect.getsource(cli.all_)
    assert "_run_report" in source, "dev-scale `all` should still write the README"
    assert "if smoke:" in source, "smoke must take a branch that skips report"

    report_line = next(i for i, ln in enumerate(source.splitlines()) if "_run_report" in ln)
    guard_line = next(i for i, ln in enumerate(source.splitlines()) if "if smoke:" in ln)
    assert guard_line < report_line, "the smoke guard must precede the report call"


def test_no_not_implemented_stubs_remain() -> None:
    """The helper that produced the stub messages should be gone entirely."""
    source = inspect.getsource(cli)
    assert "_pending" not in source
    assert "not implemented yet" not in source
