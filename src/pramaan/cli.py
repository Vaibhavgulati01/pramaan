"""Single source of truth for every PRAMAAN entry point.

The Makefile at the repo root is a thin wrapper that shells out to these
commands (`python -m pramaan.cli <command>` / the installed `pramaan`
script) so that CI, the VM, and local dev all run exactly the same code
path. See docs/EVALUATION_PROTOCOL.md for what --scale smoke/dev/full mean.
"""

from __future__ import annotations

import importlib
import sys
from enum import Enum
from pathlib import Path

import typer


def _repo_root() -> Path:
    """Repo root, resolved from this file rather than the cwd so the CLI
    works when invoked from anywhere (CI, the VM, a subdirectory)."""
    return Path(__file__).resolve().parents[2]

app = typer.Typer(
    name="pramaan",
    # Plain hyphen, not an em dash: some Windows terminals render this
    # help text in a legacy codepage where non-ASCII punctuation mangles.
    help="PRAMAAN - risk-controlled selective adjudication of claim evidence.",
    no_args_is_help=True,
)


class Scale(str, Enum):
    smoke = "smoke"
    dev = "dev"
    full = "full"


DEFAULT_SEED = 1337


def _pending(command: str, phase: str) -> None:
    typer.echo(
        f"`pramaan {command}` is not implemented yet - it lands in {phase}. "
        "See PROGRESS.md / PRAMAAN_v2_architecture.md Sec.9 for the build order.",
        err=True,
    )


# Each command below is a thin @app.command wrapper over a plain `_run_*`
# / `_build_*` function. `all` composes those plain functions, never the
# wrappers: calling a Typer command as an ordinary Python function passes
# its `typer.Option(...)` sentinel objects through as real argument
# values rather than their defaults, which surfaced as a TypeError deep
# inside random.seed() the first time `all` did real work.


def _run_setup() -> bool:
    """Returns True if every required dependency imported."""
    checks: list[tuple[str, bool, str]] = []

    def check(mod: str, optional: bool = False) -> None:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "?")
            checks.append((mod, True, version))
        except ImportError as exc:
            checks.append((mod, False, "optional, skipped" if optional else str(exc)))

    for mod in (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "lightgbm",
        "faiss",
        "torch",
        "open_clip",
        "imagehash",
        "PIL",
        "shap",
        "rapidfuzz",
        "indic_transliteration",
        "networkx",
        "pydantic",
        "hydra",
        "fastapi",
        "typer",
        "piexif",
        "yaml",
    ):
        check(mod)
    check("c2pa", optional=True)

    ok = True
    typer.echo(f"Python: {sys.version.split()[0]} ({sys.executable})")
    typer.echo("")
    for mod, passed, detail in checks:
        mark = "OK  " if passed else "FAIL"
        typer.echo(f"  [{mark}] {mod:<24} {detail}")
        if not passed and "optional" not in detail:
            ok = False

    typer.echo("")
    if ok:
        typer.echo("Environment OK.")
    else:
        typer.echo("Environment has missing required dependencies (see FAIL rows above).")
    return ok


@app.command()
def setup() -> None:
    """Validate the environment: core deps import, versions, optional deps status."""
    if not _run_setup():
        raise typer.Exit(code=1)


def _build_corpus(tier: str, seed: int) -> None:
    """Shared by `data` and `data-full` - the only difference between them
    is which tier's sizing (configs/data.yaml, set by the Phase 0.5 power
    analysis) is used, and that `full` is expected to run on the VM."""
    import logging

    import yaml

    from benchmarks.build_bench import build_bench, summarize

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config_path = _repo_root() / "configs" / "data.yaml"
    config = yaml.safe_load(config_path.read_text())
    tier_config = config[tier]

    n_claims = tier_config["n_claims"]
    if n_claims is None:
        typer.echo(
            f"configs/data.yaml has no n_claims for tier {tier!r}. "
            "Run `python scripts/run_power_analysis.py` to size it.",
            err=True,
        )
        raise typer.Exit(code=1)

    merchants = [f"merchant_{i}" for i in range(tier_config["merchants"])]
    manifest = build_bench(
        n_claims=n_claims,
        merchants=merchants,
        seed=seed,
        tier=tier,
        # Anchored to the repo, not the cwd, so `pramaan data` writes to
        # the same place regardless of where it's invoked from.
        output_root=_repo_root() / "data",
        composition=config.get("composition"),
    )
    typer.echo("")
    typer.echo(summarize(manifest))


@app.command()
def data(
    scale: Scale = typer.Option(Scale.dev, help="smoke (CI) or dev (local mechanism-scale)."),
    seed: int = typer.Option(DEFAULT_SEED, help="Corpus build seed; recorded in the manifest."),
) -> None:
    """Build PRAMAAN-Bench-v1 at the given non-full scale."""
    if scale is Scale.full:
        typer.echo("Use `pramaan data-full` for the full-scale corpus.", err=True)
        raise typer.Exit(code=1)
    _build_corpus(scale.value, seed)


@app.command(name="data-full")
def data_full(
    seed: int = typer.Option(DEFAULT_SEED, help="Corpus build seed; recorded in the manifest."),
) -> None:
    """Build the full-scale (VM) corpus, sized by the Phase 0.5 power analysis."""
    _build_corpus("full", seed)


@app.command()
def train(
    scale: Scale = typer.Option(Scale.dev, help="Which corpus tier to train the fusion model on."),
) -> None:
    """Fit LightGBM fusion + Mondrian isotonic calibration. Lands in Phase 3."""
    _pending(f"train --scale {scale.value}", "Phase 3")


@app.command()
def eval(  # noqa: A001 - matches the spec's `make eval`
    scale: Scale = typer.Option(Scale.dev, help="smoke, dev, or full."),
) -> None:
    """Run baselines, ablations, shift matrix, bootstrap CIs. Lands in Phase 6."""
    _pending(f"eval --scale {scale.value}", "Phase 6")


@app.command()
def report(
    scale: Scale = typer.Option(Scale.dev, help="smoke, dev, or full."),
) -> None:
    """Regenerate README-injected tables from reports/{scale}/metrics.json. Lands in Phase 6."""
    _pending(f"report --scale {scale.value}", "Phase 6")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Launch the FastAPI /adjudicate /explain /healthz service. Lands in Phase 8."""
    _pending(f"serve (host={host}, port={port})", "Phase 8")


@app.command(name="all")
def all_(
    scale: Scale = typer.Option(Scale.dev, help="smoke or dev (full runs on the VM)."),
    seed: int = typer.Option(DEFAULT_SEED, help="Corpus build seed; recorded in the manifest."),
) -> None:
    """setup -> data -> train -> eval -> report, for smoke or dev scale."""
    if scale is Scale.full:
        typer.echo(
            "`pramaan all` does not run --scale full: that step happens on the VM per "
            "docs/REAL_DATA_ONRAMP.md. Use `data-full`, `train`, `eval --scale full` there.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Call the plain implementations, never the @app.command wrappers:
    # invoking a Typer command as an ordinary function passes its
    # `typer.Option(...)` sentinels through as real argument values
    # instead of their defaults.
    _run_setup()
    _build_corpus(scale.value, seed)
    _pending(f"train --scale {scale.value}", "Phase 3")
    _pending(f"eval --scale {scale.value}", "Phase 6")
    _pending(f"report --scale {scale.value}", "Phase 6")


if __name__ == "__main__":
    app()
