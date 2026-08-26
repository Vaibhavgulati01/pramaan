"""Reliability diagrams, global and per `{category x price-band}` group
(PRAMAAN_v2_architecture.md Sec.4 L2's reporting requirement).

A single global reliability curve is exactly the artifact that hides the
failure this layer exists to catch: a model can sit almost perfectly on
the diagonal overall while being badly miscalibrated on high-value cells.
So the per-group panel is the point, and the global curve is context.

Figures are written to `reports/{tier}/`, and every number on them comes
from `fusion.calibration` rather than being recomputed here - a plot that
computes its own statistics is a second implementation that can silently
disagree with the first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

# Non-interactive backend, chosen before pyplot is imported: these run in
# CI and on a headless VM where no display exists.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pramaan.fusion.calibration import PerGroupReport  # noqa: E402

logger = logging.getLogger(__name__)

# Cells below this are drawn faintly: with few claims per bin their curve
# is mostly sampling noise, and showing it at full weight invites reading
# meaning into wobble.
FAINT_BELOW_N = 100


def plot_reliability(
    report: PerGroupReport,
    output_dir: Path,
    tier: str,
    label: str = "out-of-fold",
) -> list[Path]:
    """Writes a global and a per-group reliability diagram. Returns the
    paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(_plot_global(report, output_dir, tier, label))
    if report.per_group:
        written.append(_plot_per_group(report, output_dir, tier, label))
    return written


def _diagonal(ax: plt.Axes) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.6", label="perfect")


def _plot_global(report: PerGroupReport, output_dir: Path, tier: str, label: str) -> Path:
    curve = report.curves["global"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    _diagonal(ax)
    ax.plot(curve.mean_predicted, curve.observed_rate, marker="o", color="#1f77b4")

    metrics = report.overall
    ax.set_title(
        f"Reliability — {tier} ({label})\n"
        f"Brier {metrics.brier:.4f} · ECE {metrics.ece:.4f} · "
        f"MCE {metrics.mce:.4f} · n={metrics.n}",
        fontsize=10,
    )
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    path = output_dir / f"reliability_global_{tier}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def _plot_per_group(report: PerGroupReport, output_dir: Path, tier: str, label: str) -> Path:
    """Per-cell panel, worst-calibrated first.

    Ordering by ECE rather than alphabetically is deliberate: the reader
    should meet the cells that are actually miscalibrated before the ones
    that are fine.
    """
    ordered = sorted(report.per_group.items(), key=lambda kv: -kv[1].ece)
    n = len(ordered)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)

    for index, (group, metrics) in enumerate(ordered):
        ax = axes[index // cols][index % cols]
        curve = report.curves[group]
        faint = metrics.n < FAINT_BELOW_N

        _diagonal(ax)
        ax.plot(
            curve.mean_predicted,
            curve.observed_rate,
            marker="o",
            markersize=3,
            color="#d62728" if metrics.ece > report.overall.ece * 2 else "#1f77b4",
            alpha=0.45 if faint else 1.0,
        )
        ax.set_title(
            f"{group}\nECE {metrics.ece:.3f} · n={metrics.n}" + ("  (thin)" if faint else ""),
            fontsize=8,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

    for index in range(n, rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.suptitle(
        f"Reliability by category × price band — {tier} ({label})\n"
        f"red = ECE more than 2× the global {report.overall.ece:.4f}",
        fontsize=10,
    )
    path = output_dir / f"reliability_per_group_{tier}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path
