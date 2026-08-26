"""Reliability diagram rendering.

Figures are hard to assert on meaningfully, so these tests check the
things that actually break: that files are written, that a headless
backend is used (these run in CI and on a display-less VM), and that the
per-group panel exists whenever there are groups - since the per-group
panel is the entire point of Sec.4 L2's reporting requirement and
silently emitting only the global curve would hide exactly what it exists
to surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from pramaan.fusion.calibration import evaluate_calibration
from pramaan.fusion.plots import plot_reliability

RNG = np.random.default_rng(3)


def _report(n: int = 900):
    p = RNG.uniform(0, 1, n)
    y = (RNG.uniform(0, 1, n) < p).astype(float)
    groups = np.array(
        [f"{c}|{b}" for c, b in zip(
            RNG.choice(["electronics", "apparel", "home"], n),
            RNG.choice(["low", "mid", "high"], n),
            strict=True,
        )]
    )
    return evaluate_calibration(p, y, groups)


def test_uses_a_headless_backend() -> None:
    """CI and the VM have no display; an interactive backend would hang
    or crash rather than render."""
    assert matplotlib.get_backend().lower() == "agg"


def test_writes_global_and_per_group_figures(tmp_path: Path) -> None:
    written = plot_reliability(_report(), tmp_path, "dev")
    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 1000  # a real PNG, not an empty file


def test_figure_names_include_the_tier(tmp_path: Path) -> None:
    """Reports are written per tier and must never overwrite each other -
    a dev figure landing on top of a full one would be a silent
    scale-mixing error of exactly the kind the tier discipline forbids."""
    plot_reliability(_report(), tmp_path, "dev")
    plot_reliability(_report(), tmp_path, "full")
    names = {p.name for p in tmp_path.glob("*.png")}
    assert "reliability_global_dev.png" in names
    assert "reliability_global_full.png" in names


def test_only_global_figure_when_no_groups_qualify(tmp_path: Path) -> None:
    p = RNG.uniform(0, 1, 40)
    y = (RNG.uniform(0, 1, 40) < p).astype(float)
    groups = np.array([f"g{i}" for i in range(40)])  # every group of size 1
    report = evaluate_calibration(p, y, groups, min_group_size=30)
    assert not report.per_group

    written = plot_reliability(report, tmp_path, "dev")
    assert len(written) == 1
    assert "global" in written[0].name


def test_creates_the_output_directory(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "dev"
    assert not target.exists()
    plot_reliability(_report(), target, "dev")
    assert target.is_dir()
