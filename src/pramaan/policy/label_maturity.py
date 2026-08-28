"""Delayed labels and the maturity window
(PRAMAAN_v2_architecture.md Sec.4 L4).

Chargebacks arrive up to 120 days after a claim. A claim filed last week
has no settled outcome yet — it is *censored*, not legitimate, and the
difference matters enormously.

Treating immature claims as negatives is the single easiest way to
manufacture a flattering fraud metric: recent claims are
disproportionately unlabelled, unlabelled gets read as "no chargeback
happened", and the model appears to have driven fraud down when it has
only run out of time to observe it.

So evaluation happens **only on matured claims**, and the number censored
is reported alongside every metric rather than quietly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Chargeback windows in Indian card networks run to ~120 days. Configured
# rather than hard-coded at the call site so a merchant with different
# terms can set their own.
DEFAULT_MATURITY_DAYS = 120


@dataclass
class MaturityReport:
    """How much of a dataset has a trustworthy label yet."""

    as_of: datetime
    maturity_days: int
    n_total: int
    n_matured: int
    n_censored: int

    @property
    def matured_rate(self) -> float:
        return self.n_matured / self.n_total if self.n_total else 0.0

    @property
    def censored_rate(self) -> float:
        return self.n_censored / self.n_total if self.n_total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of.isoformat(),
            "maturity_days": self.maturity_days,
            "n_total": self.n_total,
            "n_matured": self.n_matured,
            "n_censored": self.n_censored,
            "matured_rate": self.matured_rate,
            "censored_rate": self.censored_rate,
        }

    def describe(self) -> str:
        return (
            f"{self.n_matured}/{self.n_total} claims matured "
            f"({self.matured_rate:.1%}) at {self.maturity_days}d as of "
            f"{self.as_of.date()}; {self.n_censored} still censored and "
            "excluded from evaluation"
        )


def maturity_mask(
    claim_timestamps: pd.Series | np.ndarray,
    as_of: datetime,
    maturity_days: int = DEFAULT_MATURITY_DAYS,
) -> np.ndarray:
    """True where a claim is old enough for its label to be trusted."""
    timestamps = pd.to_datetime(pd.Series(claim_timestamps))
    cutoff = as_of - timedelta(days=maturity_days)
    return (timestamps <= cutoff).to_numpy()


def assess_maturity(
    claim_timestamps: pd.Series | np.ndarray,
    as_of: datetime | None = None,
    maturity_days: int = DEFAULT_MATURITY_DAYS,
) -> MaturityReport:
    """Counts matured vs censored claims as of a given date.

    `as_of` defaults to the latest claim in the data rather than "now":
    on a fixed benchmark, wall-clock time is not the relevant clock, and
    using it would make the report change every day the tests are run.
    """
    timestamps = pd.to_datetime(pd.Series(claim_timestamps))
    if as_of is None:
        as_of = timestamps.max().to_pydatetime()

    matured = maturity_mask(timestamps, as_of, maturity_days)
    return MaturityReport(
        as_of=as_of,
        maturity_days=maturity_days,
        n_total=len(timestamps),
        n_matured=int(matured.sum()),
        n_censored=int((~matured).sum()),
    )


def filter_to_matured(
    frame: pd.DataFrame,
    timestamp_column: str = "claim_timestamp",
    as_of: datetime | None = None,
    maturity_days: int = DEFAULT_MATURITY_DAYS,
) -> tuple[pd.DataFrame, MaturityReport]:
    """Restricts a frame to matured claims, returning the report too.

    Returning both together is deliberate: it makes it awkward to filter
    without also having the censoring figure to hand, which is exactly
    the number that should accompany any metric computed downstream.
    """
    report = assess_maturity(frame[timestamp_column], as_of, maturity_days)
    mask = maturity_mask(frame[timestamp_column], report.as_of, maturity_days)
    return frame.loc[mask].copy(), report
