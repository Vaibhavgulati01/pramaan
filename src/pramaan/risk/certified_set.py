"""Learn-then-Test over a threshold grid: the certified set
(PRAMAAN_v2_architecture.md Sec.4 L3).

Rather than picking a denial threshold by eyeballing a PR curve, this
returns the *set* of thresholds for which the false-denial rate is
statistically certified below alpha, at confidence 1-delta. Phase 5 then
picks the cost-minimising member of that set.

## The two risks, and why both are reported

Sec.4 L3's honesty note #1 is the subtle one, and it is easy to state
the guarantee more strongly than the maths supports:

- **Conditional** `FDR_deny = P(y=0 | denied)` is what a merchant
  actually cares about - "of the claims we auto-denied, what fraction
  were honest?". But it is a ratio of two random quantities: both the
  numerator and the denominator depend on which claims happened to land
  in the denial set. The clean Learn-then-Test guarantee is for a bounded
  loss with a *fixed* denominator, so applying it here means conditioning
  on the realised denial set and treating n_denied as given. That is an
  approximation, and it is stated rather than hidden.

- **Unconditional** `E[(1-y) * 1{deny}] <= alpha'` has a fixed
  denominator (all claims) and is therefore covered by the clean
  guarantee with no caveat at all.

Both are computed for every threshold. The README quotes the conditional
form because it is the one that means something operationally, and
`docs/GUARANTEE.md` carries the caveat alongside it.

## Fixed-sequence testing

Thresholds are tested from most conservative (highest t_hi, fewest
denials, safest) downward, and testing STOPS at the first failure. This
is what controls the family-wise error rate without a multiplicity
correction: under a fixed sequence, each test is only performed if all
prior ones passed, so no alpha-spending adjustment is needed.

It also means the certified set is an interval `[t*, 1]`, not an
arbitrary subset - which matches intuition, since a stricter threshold
than a safe one is also safe.

## Power, and why the sequence must not trip over the sparse end

Testing top-down has a failure mode that is easy to miss and fatal when
it bites. The most conservative thresholds deny the fewest claims, so
they have the *least* statistical power - and Hoeffding-Bentkus will
happily return p > delta for a threshold with 40 denials and a 1%
empirical error rate, purely because 40 observations cannot support the
claim. Under a naive fixed sequence that counts as a failure, the sweep
stops immediately, and thresholds further down that would certify
comfortably are never reached.

Observed directly while building this: a near-perfect model failed at
t=0.99 on 37 denials and certified nothing at all, despite denying 600
claims at 1.2% FDR further down the grid.

The fix is to treat *insufficient power* as a skip rather than a
failure. A threshold whose denial count could not certify at the rate we
planned for carries no evidence about lower thresholds - its p-value
reflects sample size, not the model. Observed concretely at
alpha=0.10/delta=0.10: t=0.99 denied 60 claims at a 3.3% empirical FDR -
comfortably *below* alpha - and still returned p=0.143, because 60
observations cannot establish it.

The floor comes from `min_n_for_rhat` at a **pre-committed planning
rate** (`planning_rate_fraction` x alpha, default 0.3), the same
assumption Phase 0.5 used to size the corpus. Two properties make this
legitimate rather than a data-dependent fudge:

- The planning rate is fixed in advance, not read off the results.
- `n_denied` depends only on the predicted probabilities, never on the
  labels. Filtering the grid by denial count therefore uses no
  information about the risk being tested, so the fixed sequence's error
  control is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pramaan.risk.hb_pvalue import hb_pvalue
from pramaan.risk.power_analysis import min_n_for_rhat, min_n_zero_errors

# Below this many denials, LTT refuses to certify at all. Sec.4 L3 sets
# it: a handful of denials cannot honestly support a claim about a rate,
# and Hoeffding-Bentkus would happily return a number anyway.
DEFAULT_MIN_DENIED = 30

# The empirical false-denial rate we plan for, as a fraction of alpha.
# Used only to set the power floor (see the module docstring), and fixed
# in advance rather than read from the data. 0.3 matches the optimistic
# scenario Phase 0.5 used when sizing the dev corpus.
DEFAULT_PLANNING_RATE_FRACTION = 0.3


@dataclass(frozen=True)
class ThresholdResult:
    """What was measured at one candidate threshold."""

    threshold: float
    n_denied: int
    n_total: int

    # Conditional: among denied claims, the fraction that were legitimate.
    conditional_rate: float
    conditional_pvalue: float
    conditional_certified: bool

    # Unconditional: over ALL claims, the rate of legitimate-and-denied.
    unconditional_rate: float
    unconditional_pvalue: float
    unconditional_certified: bool

    skipped_reason: str | None = None

    @property
    def deny_fraction(self) -> float:
        return self.n_denied / self.n_total if self.n_total else 0.0


@dataclass
class CertifiedSet:
    """The outcome of a Learn-then-Test sweep."""

    alpha: float
    delta: float
    min_denied: int
    certified_thresholds: list[float] = field(default_factory=list)
    results: list[ThresholdResult] = field(default_factory=list)
    stopped_at: float | None = None
    stop_reason: str | None = None
    # The power floor actually applied - max(caller's min_denied, the
    # denials needed to certify this (alpha, delta) even at zero errors).
    effective_min_denied: int = DEFAULT_MIN_DENIED

    @property
    def is_empty(self) -> bool:
        return not self.certified_thresholds

    @property
    def least_conservative(self) -> float | None:
        """The smallest certified threshold - the one that denies the most
        while still carrying the guarantee. Phase 5 picks from the
        certified set by rupee cost, which will usually land here or near
        it, but that is a policy decision made separately."""
        return min(self.certified_thresholds) if self.certified_thresholds else None

    def result_at(self, threshold: float) -> ThresholdResult | None:
        for result in self.results:
            if result.threshold == threshold:
                return result
        return None

    def describe(self) -> str:
        if self.is_empty:
            return (
                f"NO certified threshold at alpha={self.alpha}, delta={self.delta}. "
                + (self.stop_reason or "")
            )
        best = self.result_at(self.least_conservative)  # type: ignore[arg-type]
        assert best is not None
        return (
            f"certified {len(self.certified_thresholds)} threshold(s) at "
            f"alpha={self.alpha}, delta={self.delta}; least conservative "
            f"t={best.threshold:.4f} (n_denied={best.n_denied}, "
            f"empirical FDR={best.conditional_rate:.4f}, "
            f"HB p={best.conditional_pvalue:.4f})"
        )


def certify_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray | None = None,
    alpha: float = 0.03,
    delta: float = 0.10,
    min_denied: int = DEFAULT_MIN_DENIED,
    planning_rate_fraction: float = DEFAULT_PLANNING_RATE_FRACTION,
) -> CertifiedSet:
    """Fixed-sequence Learn-then-Test down a monotone grid of deny
    thresholds.

    `probabilities` are CALIBRATED fraud probabilities; `labels` are 1 for
    fraud, 0 for legitimate. A claim is denied when p >= t.

    Must be run on the calibration split, once. Running it on data the
    model was fitted on produces a certificate about the training set,
    which is not a claim anyone should act on.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels).astype(int)
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if probabilities.size == 0:
        raise ValueError("cannot certify on an empty sample")

    if grid is None:
        grid = np.round(np.arange(0.50, 1.00, 0.01), 4)
    # Most conservative (highest threshold, fewest denials) first: the
    # fixed sequence must start where certification is easiest.
    ordered = np.sort(np.asarray(grid, dtype=float))[::-1]

    # Power floor: a threshold that could not certify even with zero
    # observed errors has no power, and its p-value says nothing about
    # the model. Testing it would stop the sequence for a reason that is
    # about sample size rather than risk - see the module docstring.
    power_floor = min_n_for_rhat(
        alpha, delta, r_hat=planning_rate_fraction * alpha, min_denied=1
    ) or min_n_zero_errors(alpha, delta, min_denied=1)
    effective_min_denied = max(min_denied, power_floor)

    outcome = CertifiedSet(
        alpha=alpha,
        delta=delta,
        min_denied=min_denied,
        effective_min_denied=effective_min_denied,
    )
    n_total = int(probabilities.size)

    for threshold in ordered:
        denied = probabilities >= threshold
        n_denied = int(denied.sum())

        if n_denied < effective_min_denied:
            # Not a failure, so the sequence continues: too few denials to
            # test yet, and lowering the threshold can only add more.
            outcome.results.append(
                ThresholdResult(
                    threshold=float(threshold),
                    n_denied=n_denied,
                    n_total=n_total,
                    conditional_rate=float("nan"),
                    conditional_pvalue=1.0,
                    conditional_certified=False,
                    unconditional_rate=float("nan"),
                    unconditional_pvalue=1.0,
                    unconditional_certified=False,
                    skipped_reason=(
                        f"only {n_denied} denied (< {effective_min_denied}); "
                        "too few to certify even at zero errors"
                    ),
                )
            )
            continue

        n_legit_denied = int((labels[denied] == 0).sum())
        conditional_rate = n_legit_denied / n_denied
        conditional_p = hb_pvalue(conditional_rate, n_denied, alpha)

        # Unconditional: same numerator, denominator is every claim. The
        # clean LTT guarantee applies here without the ratio caveat.
        unconditional_rate = n_legit_denied / n_total
        unconditional_p = hb_pvalue(unconditional_rate, n_total, alpha)

        result = ThresholdResult(
            threshold=float(threshold),
            n_denied=n_denied,
            n_total=n_total,
            conditional_rate=conditional_rate,
            conditional_pvalue=conditional_p,
            conditional_certified=conditional_p <= delta,
            unconditional_rate=unconditional_rate,
            unconditional_pvalue=unconditional_p,
            unconditional_certified=unconditional_p <= delta,
        )
        outcome.results.append(result)

        if result.conditional_certified:
            outcome.certified_thresholds.append(float(threshold))
            continue

        # Fixed-sequence testing: stop at the first genuine failure. Every
        # lower threshold denies a superset of these claims, so continuing
        # would both be statistically invalid (no multiplicity control)
        # and pointless.
        outcome.stopped_at = float(threshold)
        outcome.stop_reason = (
            f"first failure at t={threshold:.4f}: empirical FDR "
            f"{conditional_rate:.4f} over {n_denied} denials gives HB "
            f"p={conditional_p:.4f} > delta={delta}"
        )
        break

    if outcome.is_empty and outcome.stop_reason is None:
        tested = [r for r in outcome.results if r.skipped_reason is None]
        outcome.stop_reason = (
            f"no threshold reached {effective_min_denied} denials, the minimum "
            f"that can certify alpha={alpha} at delta={delta} even with zero errors"
            if not tested
            else "every tested threshold failed"
        )

    return outcome


def walk_alpha_delta_ladder(
    probabilities: np.ndarray,
    labels: np.ndarray,
    ladder: list[tuple[float, float]],
    grid: np.ndarray | None = None,
    min_denied: int = DEFAULT_MIN_DENIED,
) -> tuple[CertifiedSet | None, list[CertifiedSet]]:
    """Walks the pre-committed (alpha, delta) ladder from
    docs/PREREGISTRATION.md, returning the first rung that certifies
    anything plus EVERY attempt.

    Returning the failures matters as much as returning the success:
    reporting only the rung that worked, without the stricter ones that
    did not, would misrepresent how hard the guarantee was to obtain.
    Deciding this ladder before seeing the data is what separates it from
    p-hacking.
    """
    attempts: list[CertifiedSet] = []
    for alpha, delta in ladder:
        outcome = certify_thresholds(
            probabilities, labels, grid=grid, alpha=alpha, delta=delta, min_denied=min_denied
        )
        attempts.append(outcome)
        if not outcome.is_empty:
            return outcome, attempts
    return None, attempts
