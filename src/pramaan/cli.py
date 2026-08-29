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


def _run_train(tier: str, use_clip: bool) -> None:
    import logging

    from pramaan.fusion.calibration import evaluate_calibration
    from pramaan.fusion.pipeline import train_fusion

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = _repo_root()
    model, extracted = train_fusion(
        tier,
        data_root=root / "data",
        model_dir=root / "data" / tier / "model",
        use_clip=use_clip,
    )
    assert model.report is not None

    typer.echo("")
    typer.echo(f"Fusion model [{tier}]")
    typer.echo(f"  train claims:        {model.report.n_train}")
    typer.echo(f"  features:            {model.report.n_features} "
               f"(schema {model.report.schema_version})")
    typer.echo(f"  monotone constraints:{model.report.monotone_constraints_applied:4d}")
    typer.echo(f"  calibrator cells:    {len(model.report.calibrator_cells)}")
    typer.echo(
        f"  OOF Brier:           {model.report.oof_brier_uncalibrated:.4f} raw "
        f"-> {model.report.oof_brier_calibrated:.4f} calibrated"
    )

    # Calibration is reported on the OUT-OF-FOLD predictions, not on
    # in-sample ones. The calibrator was fitted on out-of-fold scores, so
    # pushing in-sample (far sharper) scores through it measures the gap
    # between two distributions rather than calibration quality.
    #
    # These numbers are still optimistic - the calibrator was fitted on
    # exactly these predictions - so they are labelled as a diagnostic,
    # not a result. The clean held-out figure arrives in Phase 6. The
    # calibration split is untouched here: it belongs to Learn-then-Test.
    _, train_labels, train_groups = extracted.for_split("train")
    assert model.oof_calibrated is not None
    report = evaluate_calibration(model.oof_calibrated, train_labels, train_groups)
    typer.echo("")
    typer.echo("  Calibration (TRAIN out-of-fold; calibrator fitted on these,")
    typer.echo("  so optimistic - a diagnostic, not a held-out result):")
    typer.echo(
        f"    Brier {report.overall.brier:.4f} | ECE {report.overall.ece:.4f} "
        f"| MCE {report.overall.mce:.4f} (n={report.overall.n})"
    )
    if report.per_group:
        typer.echo("    worst-calibrated cells by ECE:")
        for name, metrics in report.worst_groups(3):
            typer.echo(f"      {name:28s} ECE {metrics.ece:.4f} (n={metrics.n})")

    from pramaan.fusion.plots import plot_reliability

    written = plot_reliability(
        report, root / "reports" / tier, tier, label="train out-of-fold"
    )
    typer.echo("")
    for path in written:
        typer.echo(f"  wrote {path.relative_to(root)}")

    # Pillar-level gain, which is what makes a misbehaving pillar visible.
    # Inspecting exactly this is what exposed the source/label confound
    # that made Sec.6's central ablation untestable (docs/LIMITATIONS.md).
    importance = model.feature_importance()
    total = sum(importance.values()) or 1.0
    by_pillar: dict[str, float] = {}
    for name, gain in importance.items():
        by_pillar[name.split("_")[0]] = by_pillar.get(name.split("_")[0], 0.0) + gain
    typer.echo("")
    typer.echo("  Gain by pillar (diagnostic - not the Phase 6 ablation):")
    for pillar, gain in sorted(by_pillar.items(), key=lambda kv: -kv[1]):
        typer.echo(f"    {pillar:12s} {gain / total:6.1%}")


@app.command()
def train(
    scale: Scale = typer.Option(Scale.dev, help="Which corpus tier to train the fusion model on."),
    clip: bool = typer.Option(True, help="Use CLIP embeddings in the reuse pillar."),
) -> None:
    """Fit LightGBM fusion + Mondrian isotonic calibration on the train split."""
    _run_train(scale.value, use_clip=clip)


def _run_certify(tier: str, use_clip: bool) -> None:
    import logging

    from pramaan.risk.certify_pipeline import certify, summarise

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    certificate = certify(tier, _repo_root(), use_clip=use_clip)
    typer.echo("")
    typer.echo(summarise(certificate))


@app.command()
def certify(
    scale: Scale = typer.Option(Scale.dev, help="Which corpus tier to certify."),
    clip: bool = typer.Option(True, help="Use CLIP embeddings in the reuse pillar."),
) -> None:
    """Run Learn-then-Test on the calibration split and seal it.

    This is the ONLY place the calibration split is consumed. See
    docs/GUARANTEE.md for the three caveats that qualify the result.
    """
    _run_certify(scale.value, use_clip=clip)


def _run_eval(tier: str, use_clip: bool, n_resamples: int, skip_slow: bool) -> None:
    import logging

    from eval.run_eval import run_eval, summarise

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metrics = run_eval(
        tier,
        _repo_root(),
        use_clip=use_clip,
        n_resamples=n_resamples,
        skip_slow=skip_slow,
    )
    typer.echo("")
    typer.echo(summarise(metrics))


@app.command(name="eval")
def eval_(
    scale: Scale = typer.Option(Scale.dev, help="smoke, dev, or full."),
    clip: bool = typer.Option(True, help="Use CLIP embeddings in the reuse pillar."),
    resamples: int = typer.Option(2000, help="Bootstrap resamples (Sec.6 asks for 2000)."),
    skip_slow: bool = typer.Option(
        False, help="Skip ablations and negative controls (they refit the model repeatedly)."
    ),
) -> None:
    """Run baselines, ablations, negative controls and cost analysis."""
    _run_eval(scale.value, use_clip=clip, n_resamples=resamples, skip_slow=skip_slow)


def _run_report(tier: str, check: bool) -> None:
    import subprocess

    root = _repo_root()
    command = [
        sys.executable,
        str(root / "scripts" / "inject_metrics.py"),
        "--tier", tier,
        "--repo-root", str(root),
    ]
    if check:
        command.append("--check")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@app.command()
def report(
    scale: Scale = typer.Option(Scale.dev, help="smoke, dev, or full."),
    check: bool = typer.Option(
        False, help="Do not write; fail if the committed README is out of date (CI)."
    ),
) -> None:
    """Regenerate the README's generated sections from reports/{scale}/metrics.json."""
    _run_report(scale.value, check=check)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Bind port."),
    scale: Scale = typer.Option(Scale.dev, help="Which tier's model to serve."),
) -> None:
    """Launch the FastAPI /adjudicate /explain /healthz service."""
    import os

    import uvicorn

    os.environ["PRAMAAN_SCALE"] = scale.value
    uvicorn.run("pramaan.api.app:app", host=host, port=port, log_level="info")


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
    smoke = scale is Scale.smoke
    # smoke keeps CLIP off: the embedding dominates runtime and the tier
    # exists to prove the pipeline executes, not to produce numbers.
    use_clip = not smoke

    _run_setup()
    _build_corpus(scale.value, seed)
    _run_train(scale.value, use_clip=use_clip)
    _run_certify(scale.value, use_clip=use_clip)
    _run_eval(
        scale.value,
        use_clip=use_clip,
        # Sec.6 asks for 2000 resamples at reporting scale. smoke exists to
        # prove the pipeline executes inside a CI budget, and refitting the
        # model for every ablation is what makes it slow.
        n_resamples=200 if smoke else 2000,
        skip_slow=smoke,
    )

    # `report` rewrites README.md in place from reports/<tier>/metrics.json.
    # At smoke that would publish CI-budget numbers as the project's
    # headline, which the reporting contract in docs/EVALUATION_PROTOCOL.md
    # forbids outright -- so the stage is skipped rather than run and
    # reverted. Everything upstream of it has already executed.
    if smoke:
        typer.echo(
            "\nSkipping `report`: it would inject smoke-tier numbers into README.md, "
            "and smoke is never reportable (docs/EVALUATION_PROTOCOL.md). "
            "Metrics are in reports/smoke/metrics.json."
        )
    else:
        _run_report(scale.value, check=False)


if __name__ == "__main__":
    app()
