"""Size the benchmark from the guarantee's power requirement, not the
other way round.

The question this module answers: for a target (alpha, delta), what is
the smallest denied-claim set that could possibly certify, and — given
assumptions about prevalence, deny rate, and test-split fraction — how
big does the *full* corpus need to be to plausibly produce a denial set
that large? See docs/GUARANTEE.md for the resulting power curve and the
sizing decision it drove, and docs/PREREGISTRATION.md for the pre-
committed (alpha, delta) decision ladder this feeds.

Every "assumed_*" parameter here is a stated planning assumption, not a
guarantee — it gets revisited once a real cascade exists (Phase 6+) and
the actual deny rate is measurable instead of assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pramaan.risk.hb_pvalue import hb_pvalue


def min_n_zero_errors(alpha: float, delta: float, min_denied: int = 30) -> int:
    """Closed-form floor: the smallest n at which certification is even
    possible in principle, i.e. if every single denied claim in the
    sample happened to be a true fraud (r_hat = 0, the best case).

    At r_hat=0, hb_pvalue reduces to (1-alpha)**n (Hoeffding dominates
    Bentkus there since bentkus = e * hoeffding), so we need
    (1-alpha)**n <= delta, i.e. n >= log(delta) / log(1-alpha).
    """
    n = math.ceil(math.log(delta) / math.log(1 - alpha))
    return max(n, min_denied)


def min_n_for_rhat(
    alpha: float,
    delta: float,
    r_hat: float,
    min_denied: int = 30,
    n_max: int = 2_000_000,
) -> int | None:
    """The smallest denial-set size n such that certification holds at n
    **and at every larger n** — the point beyond which more evidence never
    takes the guarantee away again.

    Returns None if unreachable by n_max, or if r_hat already meets or
    exceeds alpha (no sample size can rescue that).

    ## Why this is not a binary search

    `hb_pvalue` is NOT monotone in n. It is the minimum of a Hoeffding and
    a Bentkus bound, and the Bentkus term contains `binom.cdf(ceil(n *
    r_hat), n, alpha)`; as n rises by one, `ceil(n * r_hat)` can jump,
    stepping the p-value *upward*. Measured across 30 (alpha, r_hat)
    combinations, 22 showed at least one such increase, the largest being
    0.096 — far too big to dismiss as floating-point noise.

    An earlier version binary-searched on the assumption of monotonicity
    and returned genuinely wrong answers: at alpha=0.03, r_hat=0.5*alpha
    it reported 489 when scanning finds certification holds from 455
    onward. That 489 was the figure used to size the corpus and published
    in docs/GUARANTEE.md.

    So this scans instead, and deliberately returns the **stable** floor
    rather than the smallest lucky n. An isolated n that certifies while
    n+1 does not is an artifact of binomial discreteness, and treating it
    as "the sample size you need" would be advice that stops being true if
    one more claim arrives.
    """
    if r_hat >= alpha:
        return None

    if hb_pvalue(r_hat, n_max, alpha) > delta:
        return None

    # Find any n that certifies, doubling upward.
    upper = max(min_denied, 1)
    while hb_pvalue(r_hat, upper, alpha) > delta:
        upper *= 2
        if upper > n_max:
            return None

    # Walk down from a point known to be inside the stable region, and
    # stop at the last n that fails. Everything above it certifies.
    # `upper` itself may sit in a lucky pocket, so start from a point
    # comfortably above it before descending.
    probe = min(upper * 2, n_max)
    while probe > 1 and hb_pvalue(r_hat, probe, alpha) <= delta:
        probe -= 1
    floor = probe + 1

    return max(floor, min_denied)


@dataclass(frozen=True)
class PowerCurvePoint:
    alpha: float
    delta: float
    r_hat_assumed: float
    r_hat_as_fraction_of_alpha: float
    min_denial_set_n: int | None


def power_curve(
    alphas: list[float],
    delta: float,
    r_hat_fractions_of_alpha: tuple[float, ...] = (0.0, 0.3, 0.5),
    min_denied: int = 30,
) -> list[PowerCurvePoint]:
    """The artifact docs/GUARANTEE.md publishes: minimum certifiable
    denial-set n vs. alpha, at fixed delta, for a few assumed empirical
    false-denial rates expressed as a fraction of alpha (0.0 = the
    zero-error floor; 0.3/0.5 = "the model comfortably clears the bar" /
    "the model just barely clears the bar" scenarios).
    """
    points: list[PowerCurvePoint] = []
    for alpha in alphas:
        for frac in r_hat_fractions_of_alpha:
            r_hat = frac * alpha
            n: int | None
            if frac == 0.0:
                n = min_n_zero_errors(alpha, delta, min_denied)
            else:
                n = min_n_for_rhat(alpha, delta, r_hat, min_denied)
            points.append(
                PowerCurvePoint(
                    alpha=alpha,
                    delta=delta,
                    r_hat_assumed=r_hat,
                    r_hat_as_fraction_of_alpha=frac,
                    min_denial_set_n=n,
                )
            )
    return points


@dataclass(frozen=True)
class CorpusSizeEstimate:
    alpha: float
    delta: float
    r_hat_assumed: float
    min_denial_set_n: int | None
    assumed_deny_rate: float
    assumed_test_split_fraction: float
    required_test_n: int | None
    required_full_corpus_n: int | None


def required_full_corpus_size(
    alpha: float,
    delta: float,
    r_hat_as_fraction_of_alpha: float = 0.3,
    assumed_deny_rate: float = 0.08,
    assumed_test_split_fraction: float = 0.20,
    min_denied: int = 30,
) -> CorpusSizeEstimate:
    """Back out a full-corpus size target from the power requirement.

    assumed_deny_rate: fraction of TEST-split claims expected to land in
    the DENY band. Fraud prevalence is ~0.15 (configs/data.yaml); DENY is
    the confident tail of that (REVIEW absorbs the uncertain middle), so
    0.08 (~half of prevalence) is used as a planning default until a
    real cascade makes this measurable.

    assumed_test_split_fraction: share of the full corpus held out as
    the frozen test set (0.20, standard).
    """
    r_hat = r_hat_as_fraction_of_alpha * alpha
    min_n = min_n_for_rhat(alpha, delta, r_hat, min_denied)

    if min_n is None:
        return CorpusSizeEstimate(
            alpha=alpha,
            delta=delta,
            r_hat_assumed=r_hat,
            min_denial_set_n=None,
            assumed_deny_rate=assumed_deny_rate,
            assumed_test_split_fraction=assumed_test_split_fraction,
            required_test_n=None,
            required_full_corpus_n=None,
        )

    required_test_n = math.ceil(min_n / assumed_deny_rate)
    required_full_corpus_n = math.ceil(required_test_n / assumed_test_split_fraction)

    return CorpusSizeEstimate(
        alpha=alpha,
        delta=delta,
        r_hat_assumed=r_hat,
        min_denial_set_n=min_n,
        assumed_deny_rate=assumed_deny_rate,
        assumed_test_split_fraction=assumed_test_split_fraction,
        required_test_n=required_test_n,
        required_full_corpus_n=required_full_corpus_n,
    )


def resolve_primary_target(
    ladder: list[tuple[float, float]],
    max_practical_full_n: int,
    r_hat_as_fraction_of_alpha: float = 0.3,
    assumed_deny_rate: float = 0.08,
    assumed_test_split_fraction: float = 0.20,
    min_denied: int = 30,
) -> tuple[CorpusSizeEstimate, list[CorpusSizeEstimate]]:
    """Walk the pre-committed (alpha, delta) ladder (docs/PREREGISTRATION.md)
    and return the first rung whose required full-corpus size is within
    max_practical_full_n, plus every rung attempted (so failed attempts
    are reportable, not silently dropped) - the same discipline the
    ladder itself applies at `full` scale, run here in advance against
    the *planning* assumptions so the corpus can be sized before any
    data exists.
    """
    attempts: list[CorpusSizeEstimate] = []
    for alpha, delta in ladder:
        estimate = required_full_corpus_size(
            alpha,
            delta,
            r_hat_as_fraction_of_alpha=r_hat_as_fraction_of_alpha,
            assumed_deny_rate=assumed_deny_rate,
            assumed_test_split_fraction=assumed_test_split_fraction,
            min_denied=min_denied,
        )
        attempts.append(estimate)
        if (
            estimate.required_full_corpus_n is not None
            and estimate.required_full_corpus_n <= max_practical_full_n
        ):
            return estimate, attempts
    # Nothing on the ladder was practical: return the last attempt (most
    # permissive rung) alongside the full attempt history.
    return attempts[-1], attempts
