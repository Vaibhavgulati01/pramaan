"""Feature drift and the guarantee watchdog (Sec.4 L7).

## The watchdog is the interesting half

A certificate is a statement about a distribution. When the distribution
moves, the certificate expires — silently, with no error and no alert,
while the system keeps auto-denying under a bound that no longer holds.

`GuaranteeWatchdog` recomputes the realised false-denial rate on each
matured label batch and charts it against α. **When the realised rate
crosses α, the certificate has expired and the abstention band widens
automatically.** That is the natural consequence of Sec.4 L3, and almost
nobody builds it: a guarantee that cannot notice its own expiry is a
guarantee only until the first shift.

Widening rather than alerting is the deliberate choice. An alert asks a
human to act while the system keeps denying under an expired bound;
widening stops the unbounded behaviour immediately and *then* asks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

# PSI convention (widely used in credit risk, and stated rather than
# assumed): <0.10 stable, 0.10-0.25 moderate, >0.25 significant.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between a reference and a current sample.

    Bin edges come from the REFERENCE quantiles, not the pooled data:
    binning on the pooled sample would let the current batch move the
    bins and hide the very shift PSI exists to detect.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return 0.0  # a near-constant feature cannot meaningfully drift
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Floor at a small epsilon: an empty bin makes the log term infinite,
    # which would report catastrophic drift from a sampling accident.
    ref_pct = np.maximum(ref_counts / ref_counts.sum(), 1e-6)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    from scipy.stats import ks_2samp

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0
    return float(ks_2samp(reference, current).statistic)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks: float

    @property
    def severity(self) -> str:
        if self.psi >= PSI_SIGNIFICANT:
            return "significant"
        if self.psi >= PSI_MODERATE:
            return "moderate"
        return "stable"

    def as_dict(self) -> dict[str, object]:
        return {"feature": self.feature, "psi": self.psi, "ks": self.ks,
                "severity": self.severity}


@dataclass
class DriftReport:
    features: list[FeatureDrift] = field(default_factory=list)

    @property
    def n_significant(self) -> int:
        return sum(1 for f in self.features if f.severity == "significant")

    def worst(self, k: int = 5) -> list[FeatureDrift]:
        return sorted(self.features, key=lambda f: -f.psi)[:k]

    def as_dict(self) -> dict[str, object]:
        return {
            "features": [f.as_dict() for f in self.features],
            "n_significant": self.n_significant,
            "n_moderate": sum(1 for f in self.features if f.severity == "moderate"),
            "worst": [f.as_dict() for f in self.worst()],
        }


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
    """PSI + KS per feature, over two feature frames sharing a schema."""
    report = DriftReport()
    for column in reference.columns:
        if column not in current.columns:
            continue
        report.features.append(
            FeatureDrift(
                feature=str(column),
                psi=population_stability_index(
                    reference[column].to_numpy(), current[column].to_numpy()
                ),
                ks=ks_statistic(
                    reference[column].to_numpy(), current[column].to_numpy()
                ),
            )
        )
    return report


# --- the guarantee watchdog -------------------------------------------


@dataclass
class WatchdogObservation:
    as_of: datetime
    n_denied: int
    n_legit_denied: int
    realised_fdr: float
    alpha: float
    breached: bool
    action: str

    def as_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of.isoformat(),
            "n_denied": self.n_denied,
            "n_legit_denied": self.n_legit_denied,
            "realised_fdr": self.realised_fdr,
            "alpha": self.alpha,
            "breached": self.breached,
            "action": self.action,
        }


class GuaranteeWatchdog:
    """Tracks realised FDR against α and widens the band on breach.

    Widening rather than merely alerting is deliberate: an alert leaves
    the system auto-denying under a bound that no longer holds while it
    waits for a human. Widening stops the unbounded behaviour first.
    """

    def __init__(
        self,
        alpha: float,
        initial_t_deny: float,
        widen_step: float = 0.02,
        min_batch: int = 30,
    ) -> None:
        self.alpha = alpha
        self.t_deny = initial_t_deny
        self.initial_t_deny = initial_t_deny
        self.widen_step = widen_step
        self.min_batch = min_batch
        self.observations: list[WatchdogObservation] = []

    @property
    def has_breached(self) -> bool:
        return any(o.breached for o in self.observations)

    @property
    def total_widening(self) -> float:
        return self.t_deny - self.initial_t_deny

    def observe_batch(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        as_of: datetime,
    ) -> WatchdogObservation:
        """Processes one matured label batch."""
        probabilities = np.asarray(probabilities, dtype=float)
        labels = np.asarray(labels).astype(int)

        denied = probabilities >= self.t_deny
        n_denied = int(denied.sum())

        if n_denied < self.min_batch:
            # Too few denials to judge. Reported rather than skipped, so a
            # long run of uninformative batches is visible instead of
            # looking like a clean bill of health.
            observation = WatchdogObservation(
                as_of=as_of, n_denied=n_denied, n_legit_denied=0,
                realised_fdr=float("nan"), alpha=self.alpha, breached=False,
                action=f"no action: {n_denied} denials is below the {self.min_batch} "
                       "needed to judge the rate",
            )
            self.observations.append(observation)
            return observation

        n_legit_denied = int((labels[denied] == 0).sum())
        realised = n_legit_denied / n_denied
        breached = realised > self.alpha

        if breached:
            self.t_deny = min(1.0, self.t_deny + self.widen_step)
            action = (
                f"CERTIFICATE EXPIRED: realised FDR {realised:.4f} exceeds alpha "
                f"{self.alpha}. Abstention band widened; t_deny "
                f"{self.t_deny - self.widen_step:.3f} -> {self.t_deny:.3f}. "
                "Re-certify on fresh calibration data before narrowing again."
            )
        else:
            action = f"within bound (realised {realised:.4f} <= alpha {self.alpha})"

        observation = WatchdogObservation(
            as_of=as_of, n_denied=n_denied, n_legit_denied=n_legit_denied,
            realised_fdr=realised, alpha=self.alpha, breached=breached, action=action,
        )
        self.observations.append(observation)
        return observation

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "initial_t_deny": self.initial_t_deny,
            "current_t_deny": self.t_deny,
            "total_widening": self.total_widening,
            "has_breached": self.has_breached,
            "n_observations": len(self.observations),
            "observations": [o.as_dict() for o in self.observations],
        }
