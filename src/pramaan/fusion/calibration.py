"""Mondrian (group-conditional) isotonic calibration and its metrics
(PRAMAAN_v2_architecture.md Sec.4 L2).

Marginal calibration is not enough when decisions are per-claim and
priced per-rupee. A model that is well calibrated overall but badly
calibrated on high-value electronics will lose money exactly where the
money is, and a single global reliability curve hides that completely.

So a separate isotonic regressor is fitted per `{category x price_band}`
cell, with a **shrinkage fallback to the global calibrator for thin
cells** - a cell with twelve claims cannot support its own monotone step
function, and pretending otherwise produces confident nonsense on the
smallest, often most expensive, segments.

Shrinkage is a size-weighted blend rather than a hard switch:

    w = n_cell / (n_cell + prior_strength)
    p = w * p_cell + (1 - w) * p_global

so a cell earns independence gradually as evidence accumulates, instead
of flipping behaviour the moment it crosses an arbitrary threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression

# Cells smaller than this get essentially the global curve; cells much
# larger get essentially their own. At n = prior_strength the blend is
# exactly half and half.
DEFAULT_PRIOR_STRENGTH = 50.0

# A cell below this many samples is not fitted at all - isotonic on a
# handful of points is a step function memorising noise.
MIN_CELL_SIZE = 20


@dataclass
class CalibrationMetrics:
    """Brier, ECE, MCE (Sec.4 L2's reporting requirement)."""

    brier: float
    ece: float
    mce: float
    n: int
    n_bins: int

    def as_dict(self) -> dict[str, float]:
        return {
            "brier": self.brier,
            "ece": self.ece,
            "mce": self.mce,
            "n": float(self.n),
            "n_bins": float(self.n_bins),
        }


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def _equal_mass_bins(probabilities: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Equal-MASS bins, as Sec.4 L2 specifies, not equal-width.

    Equal-width bins are the common default and are misleading here:
    fraud scores pile up near zero, so most equal-width bins end up empty
    and ECE is dominated by a handful of sparsely-populated high bins.
    Equal-mass bins put the same number of claims in each, so every bin's
    contribution is comparably estimated.
    """
    order = np.argsort(probabilities, kind="stable")
    return [chunk for chunk in np.array_split(order, n_bins) if chunk.size > 0]


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[float, float]:
    """Returns (ECE, MCE) over equal-mass bins."""
    if probabilities.size == 0:
        return 0.0, 0.0

    gaps = []
    weights = []
    for idx in _equal_mass_bins(probabilities, n_bins):
        gap = abs(float(probabilities[idx].mean()) - float(labels[idx].mean()))
        gaps.append(gap)
        weights.append(idx.size)

    if not gaps:
        return 0.0, 0.0
    gaps_arr = np.array(gaps)
    weights_arr = np.array(weights, dtype=float)
    ece = float(np.sum(gaps_arr * weights_arr) / weights_arr.sum())
    return ece, float(gaps_arr.max())


def calibration_metrics(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> CalibrationMetrics:
    ece, mce = expected_calibration_error(probabilities, labels, n_bins)
    return CalibrationMetrics(
        brier=brier_score(probabilities, labels),
        ece=ece,
        mce=mce,
        n=int(probabilities.size),
        n_bins=n_bins,
    )


@dataclass
class ReliabilityCurve:
    """Points for a reliability diagram, per group."""

    group: str
    mean_predicted: list[float]
    observed_rate: list[float]
    bin_counts: list[int]
    n: int


class MondrianIsotonicCalibrator:
    """Group-conditional isotonic calibration with shrinkage.

    Fit on out-of-fold predictions, never on in-sample ones: an isotonic
    regressor fitted to a model's own training scores learns that the
    model is far more accurate than it is, and calibrates in the wrong
    direction. `fusion/model.py` uses cross-fitting to supply honest
    inputs here.
    """

    def __init__(
        self,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        min_cell_size: int = MIN_CELL_SIZE,
    ) -> None:
        self.prior_strength = prior_strength
        self.min_cell_size = min_cell_size
        self._global: IsotonicRegression | None = None
        self._cells: dict[str, IsotonicRegression] = {}
        self._cell_sizes: dict[str, int] = {}
        self._fitted = False

    @staticmethod
    def group_key(category: str, price_band: str) -> str:
        return f"{category}|{price_band}"

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def cell_names(self) -> list[str]:
        return sorted(self._cells)

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> MondrianIsotonicCalibrator:
        probabilities = np.asarray(probabilities, dtype=float)
        labels = np.asarray(labels, dtype=float)
        groups = np.asarray(groups)

        if probabilities.size == 0:
            raise ValueError("cannot fit a calibrator on zero samples")

        self._global = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._global.fit(probabilities, labels)

        self._cells.clear()
        self._cell_sizes.clear()
        for group in np.unique(groups):
            mask = groups == group
            n = int(mask.sum())
            self._cell_sizes[str(group)] = n
            if n < self.min_cell_size:
                continue  # too thin to fit; falls back to global entirely
            # A cell whose labels are all one class gives isotonic nothing
            # to order, and it would emit a constant. The global curve is
            # strictly more informative there.
            if len(np.unique(labels[mask])) < 2:
                continue
            regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            regressor.fit(probabilities[mask], labels[mask])
            self._cells[str(group)] = regressor

        self._fitted = True
        return self

    def predict(self, probabilities: np.ndarray, groups: np.ndarray) -> np.ndarray:
        if not self._fitted or self._global is None:
            raise RuntimeError("calibrator is not fitted")

        probabilities = np.asarray(probabilities, dtype=float)
        groups = np.asarray(groups)
        calibrated = self._global.predict(probabilities)

        for group in np.unique(groups):
            key = str(group)
            regressor = self._cells.get(key)
            if regressor is None:
                continue  # unseen or thin cell: global curve already applied
            mask = groups == group
            n = self._cell_sizes.get(key, 0)
            weight = n / (n + self.prior_strength)
            cell_prediction = regressor.predict(probabilities[mask])
            calibrated[mask] = weight * cell_prediction + (1 - weight) * calibrated[mask]

        return np.clip(calibrated, 0.0, 1.0)

    def shrinkage_weight(self, group: str) -> float:
        """How much this cell's own curve counts, in [0, 1). Exposed for
        reporting - a reader should be able to see which cells are
        effectively global."""
        n = self._cell_sizes.get(group, 0)
        if group not in self._cells:
            return 0.0
        return float(n / (n + self.prior_strength))


def reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    group_name: str = "global",
    n_bins: int = 10,
) -> ReliabilityCurve:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)

    mean_predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for idx in _equal_mass_bins(probabilities, n_bins):
        mean_predicted.append(float(probabilities[idx].mean()))
        observed.append(float(labels[idx].mean()))
        counts.append(int(idx.size))

    return ReliabilityCurve(
        group=group_name,
        mean_predicted=mean_predicted,
        observed_rate=observed,
        bin_counts=counts,
        n=int(probabilities.size),
    )


@dataclass
class PerGroupReport:
    """Calibration quality per `{category x price_band}` cell.

    The point of Sec.4 L2: a system well-calibrated overall and badly
    calibrated on high-value electronics is a system that loses money
    exactly where money is. This is what surfaces that.
    """

    overall: CalibrationMetrics
    per_group: dict[str, CalibrationMetrics] = field(default_factory=dict)
    curves: dict[str, ReliabilityCurve] = field(default_factory=dict)

    def worst_groups(self, k: int = 5) -> list[tuple[str, CalibrationMetrics]]:
        """Cells with the largest ECE, biggest first - the ones worth
        looking at before trusting the headline number."""
        return sorted(self.per_group.items(), key=lambda kv: -kv[1].ece)[:k]


def evaluate_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 10,
    min_group_size: int = 30,
) -> PerGroupReport:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    groups = np.asarray(groups)

    report = PerGroupReport(
        overall=calibration_metrics(probabilities, labels, n_bins),
        curves={"global": reliability_curve(probabilities, labels, "global", n_bins)},
    )

    for group in np.unique(groups):
        mask = groups == group
        if int(mask.sum()) < min_group_size:
            continue  # too few to estimate calibration error meaningfully
        key = str(group)
        report.per_group[key] = calibration_metrics(probabilities[mask], labels[mask], n_bins)
        report.curves[key] = reliability_curve(probabilities[mask], labels[mask], key, n_bins)

    return report
