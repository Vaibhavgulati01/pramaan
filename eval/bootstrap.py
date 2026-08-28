"""Bootstrap confidence intervals (PRAMAAN_v2_architecture.md Sec.6).

Every headline figure carries a 95% CI from 2,000 resamples. A point
estimate on 607 test claims without an interval invites a reader to
treat a 0.02 PR-AUC difference as meaningful when it is noise, and at
these sample sizes it usually is.

Resampling is **at the claim level with a fixed seed**, so intervals are
reproducible run to run — which the determinism test depends on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_N_RESAMPLES = 2000
DEFAULT_SEED = 1337


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    lower: float
    upper: float
    n_resamples: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def as_dict(self) -> dict[str, float]:
        return {
            "point": self.point,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "ci_width": self.width,
            "n_resamples": float(self.n_resamples),
        }

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def overlaps(self, other: ConfidenceInterval) -> bool:
        """Whether two intervals overlap.

        Non-overlapping intervals imply a difference; overlapping ones do
        NOT imply no difference (that requires a CI on the paired
        difference, which `bootstrap_difference` provides). Reported so
        the weaker inference is available without inviting the stronger
        one to be read into it.
        """
        return not (self.upper < other.lower or other.upper < self.lower)


def bootstrap_metric(
    metric: Callable[[np.ndarray, np.ndarray], float],
    labels: np.ndarray,
    scores: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for a metric of (labels, scores)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(labels)
    if n == 0:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), 0)

    point = float(metric(labels, scores))

    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    drawn = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_labels = labels[idx]
        # A resample with one class present makes most ranking metrics
        # undefined. Skipping is honest; substituting 0 or 0.5 would
        # quietly shift the interval.
        if len(np.unique(resampled_labels)) < 2:
            continue
        values[drawn] = metric(resampled_labels, scores[idx])
        drawn += 1

    if drawn == 0:
        return ConfidenceInterval(point, float("nan"), float("nan"), 0)

    values = values[:drawn]
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return ConfidenceInterval(point, lower, upper, drawn)


def bootstrap_difference(
    metric: Callable[[np.ndarray, np.ndarray], float],
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """CI on the PAIRED difference metric(a) - metric(b).

    Paired on the same resampled claims, which is the correct comparison
    and much tighter than comparing two independent intervals. An
    interval excluding zero is evidence of a real difference; overlapping
    marginal intervals are not evidence of no difference.
    """
    labels = np.asarray(labels)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    n = len(labels)
    if n == 0:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), 0)

    point = float(metric(labels, scores_a) - metric(labels, scores_b))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    drawn = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = labels[idx]
        if len(np.unique(resampled)) < 2:
            continue
        diffs[drawn] = metric(resampled, scores_a[idx]) - metric(resampled, scores_b[idx])
        drawn += 1

    if drawn == 0:
        return ConfidenceInterval(point, float("nan"), float("nan"), 0)

    diffs = diffs[:drawn]
    return ConfidenceInterval(
        point,
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        drawn,
    )


def bootstrap_scalar(
    values: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """CI on the mean of per-claim values - used for rupee costs, where
    the statistic is a mean rather than a ranking metric."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    means = np.array(
        [values[rng.integers(0, n, size=n)].mean() for _ in range(n_resamples)]
    )
    return ConfidenceInterval(
        float(values.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
        n_resamples,
    )
