import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pramaan.risk.hb_pvalue import hb_pvalue, kl_bernoulli


def test_rhat_at_or_above_alpha_is_pvalue_one() -> None:
    assert hb_pvalue(0.05, 100, 0.03) == 1.0
    assert hb_pvalue(0.03, 100, 0.03) == 1.0


def test_hand_computed_zero_error_boundary_alpha03_delta010() -> None:
    """At r_hat=0, hb_pvalue collapses to (1-alpha)**n (Hoeffding
    dominates Bentkus by a factor of e). n=76 is the hand-derived minimum
    n at which (0.97)**n first drops to <= 0.10; n=75 must not certify.
    This is the exact number docs/GUARANTEE.md's power curve is built on.
    """
    assert hb_pvalue(0.0, 75, 0.03) > 0.10
    assert hb_pvalue(0.0, 76, 0.03) <= 0.10
    # rel=1e-6, not tighter: kl_bernoulli clips p away from exactly 0 (to
    # 1e-12) to keep log() finite, which introduces a proportionally tiny
    # deviation from the pure closed form - expected, not a bug.
    assert hb_pvalue(0.0, 76, 0.03) == pytest.approx((0.97) ** 76, rel=1e-6)


def test_pvalue_decreases_toward_zero_as_rhat_shrinks() -> None:
    p_high = hb_pvalue(0.029, 200, 0.03)
    p_low = hb_pvalue(0.0, 200, 0.03)
    assert p_low < p_high


@given(
    alpha=st.floats(min_value=0.01, max_value=0.3),
    r_hat_frac=st.floats(min_value=0.0, max_value=0.95),
    n_small=st.integers(min_value=30, max_value=500),
    extra=st.integers(min_value=200, max_value=2000),
)
def test_pvalue_decreases_with_substantially_more_evidence(
    alpha: float, r_hat_frac: float, n_small: int, extra: int
) -> None:
    """Substantially more evidence at the same empirical rate tightens the
    bound.

    Note this is NOT pointwise monotonicity, which hb_pvalue does not
    have. The Bentkus term contains `binom.cdf(ceil(n * r_hat), n,
    alpha)`, and as n rises by one `ceil(n * r_hat)` can jump, stepping
    the p-value upward - measured increases up to 0.096 across 22 of 30
    (alpha, r_hat) combinations. An earlier version of this test asserted
    strict monotonicity for `extra >= 1` and Hypothesis correctly found
    the counterexample (alpha=0.297, n=33 -> 34).

    That discovery mattered: `min_n_for_rhat` had been binary-searching on
    the false assumption and is now a scan for the stable floor.
    """
    r_hat = r_hat_frac * alpha
    p_small = hb_pvalue(r_hat, n_small, alpha)
    p_large = hb_pvalue(r_hat, n_small + extra, alpha)
    assert p_large <= p_small + 1e-9


def test_pvalue_is_not_pointwise_monotone_in_n() -> None:
    """Pins the surprising property so it cannot be silently 'fixed' back
    into an assumption something else depends on."""
    alpha = 0.296875
    r_hat = 0.5 * alpha
    assert hb_pvalue(r_hat, 34, alpha) > hb_pvalue(r_hat, 33, alpha)


@pytest.mark.parametrize("bad_r_hat", [-0.1, 1.1])
def test_rejects_invalid_rhat(bad_r_hat: float) -> None:
    with pytest.raises(ValueError):
        hb_pvalue(bad_r_hat, 100, 0.03)


def test_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        hb_pvalue(0.01, 0, 0.03)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_rejects_invalid_alpha(bad_alpha: float) -> None:
    with pytest.raises(ValueError):
        hb_pvalue(0.01, 100, bad_alpha)


def test_kl_bernoulli_zero_at_equal_probabilities() -> None:
    assert kl_bernoulli(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)


def test_kl_bernoulli_matches_hand_formula() -> None:
    p, q = 0.02, 0.05
    expected = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    assert kl_bernoulli(p, q) == pytest.approx(expected)
