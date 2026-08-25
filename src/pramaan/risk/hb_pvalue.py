"""Hoeffding-Bentkus p-value for the null hypothesis H0: R > alpha, where R
is the true false-denial rate and r_hat is its empirical estimate over n
denied claims.

This is the statistical primitive underneath Learn-then-Test (Bates et al.
2021, "Distribution-Free, Risk-Controlling Prediction Sets"; Angelopoulos
et al., "Learn Then Test"). A small p-value lets us reject H0 and certify
R <= alpha with confidence >= 1 - delta (via the fixed-sequence testing in
certified_set.py, Phase 4). See docs/GUARANTEE.md.

Productionised from the pseudocode in PRAMAAN_v2_architecture.md Sec.4 L3 -
same math, with input validation and documented edge cases.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def kl_bernoulli(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)), clipped away from {0,1} so the
    log terms stay finite. Used by the Hoeffding half of hb_pvalue."""
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    q = float(np.clip(q, 1e-12, 1 - 1e-12))
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def hb_pvalue(r_hat: float, n: int, alpha: float) -> float:
    """P-value for H0: R > alpha, given n denied claims with empirical
    false-denial rate r_hat. min(1, Hoeffding bound, Bentkus bound) - the
    tighter of the two dominates depending on (r_hat, n, alpha).

    r_hat >= alpha trivially cannot reject H0 (the data is consistent
    with, or worse than, the null), so the p-value is 1.0 by convention.
    """
    if not 0 <= r_hat <= 1:
        raise ValueError(f"r_hat must be in [0, 1], got {r_hat}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if r_hat >= alpha:
        return 1.0

    hoeffding = float(np.exp(-n * kl_bernoulli(r_hat, alpha)))
    bentkus = float(np.e * binom.cdf(np.ceil(n * r_hat), n, alpha))
    return float(min(1.0, hoeffding, bentkus))
