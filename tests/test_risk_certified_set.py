"""Learn-then-Test certified sets.

The properties that matter are the ones a plausible-looking bug would
silently violate: fixed-sequence testing must stop at the first failure
(continuing past it invalidates the family-wise error control), the
certified set must be an upward-closed interval, and both the conditional
and unconditional risks must be reported so the ratio caveat in
docs/GUARANTEE.md stays honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from pramaan.risk.certified_set import (
    CertifiedSet,
    certify_thresholds,
    walk_alpha_delta_ladder,
)

RNG = np.random.default_rng(11)


def _separable(n: int = 6000, fraud_rate: float = 0.15, legit_leak: float = 0.004):
    """A good model: fraud scores high, legit scores low, with a small
    fraction of legit claims leaking into the high band.

    `legit_leak` is applied to LEGIT claims only, so it maps directly onto
    the realised false-denial rate the certificate is about. At 0.004 with
    a 15% fraud rate the denial region sits near
    `0.004*0.85 / 0.15 ~= 2.3%` FDR, comfortably certifiable at
    alpha=0.10 and marginal at alpha=0.03 - which is the regime worth
    testing.

    An earlier version applied the noise to *all* claims at 2%, giving a
    true FDR near 10%. Testing that fixture at alpha=0.10 was asking the
    procedure to certify a rate equal to its own bound, and it correctly
    refused - a broken fixture, not a broken certifier.
    """
    labels = (RNG.uniform(0, 1, n) < fraud_rate).astype(int)
    probabilities = np.where(
        labels == 1, RNG.uniform(0.85, 1.0, n), RNG.uniform(0.0, 0.3, n)
    )
    leak = (labels == 0) & (RNG.uniform(0, 1, n) < legit_leak)
    probabilities[leak] = RNG.uniform(0.85, 1.0, int(leak.sum()))
    return np.clip(probabilities, 0, 1), labels


def _useless(n: int = 4000, fraud_rate: float = 0.15):
    """Scores unrelated to the label: nothing should certify."""
    labels = (RNG.uniform(0, 1, n) < fraud_rate).astype(int)
    return RNG.uniform(0, 1, n), labels


def test_good_model_certifies_something() -> None:
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.10, delta=0.10)
    assert not outcome.is_empty
    assert outcome.least_conservative is not None


def test_useless_model_certifies_nothing() -> None:
    """A model with no signal denies legitimate claims at the base rate,
    which must never clear a strict alpha."""
    p, y = _useless()
    outcome = certify_thresholds(p, y, alpha=0.03, delta=0.10)
    assert outcome.is_empty
    assert outcome.stop_reason


def test_certified_set_is_upward_closed() -> None:
    """If t certifies, every stricter threshold must too - a stricter
    threshold denies a subset of the same claims. A set with holes means
    the sequence logic is wrong."""
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.10, delta=0.10)
    assert not outcome.is_empty

    certified = sorted(outcome.certified_thresholds)
    tested = sorted(
        r.threshold for r in outcome.results if r.skipped_reason is None
    )
    above_floor = [t for t in tested if t >= certified[0]]
    assert set(above_floor) == set(certified), "certified set has a hole"


def test_fixed_sequence_stops_at_the_first_failure() -> None:
    """Continuing past a failure would test a family of hypotheses with no
    multiplicity control, which is exactly what the fixed sequence exists
    to avoid."""
    # Leaky enough that lower thresholds genuinely exceed alpha.
    p, y = _separable(legit_leak=0.05)
    outcome = certify_thresholds(p, y, alpha=0.03, delta=0.10)

    if outcome.stopped_at is not None:
        tested = [r.threshold for r in outcome.results if r.skipped_reason is None]
        # Nothing below the stopping point may have been tested at all.
        assert min(tested) == outcome.stopped_at
        assert all(t >= outcome.stopped_at for t in tested)


def test_thresholds_are_tested_most_conservative_first() -> None:
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.10, delta=0.10)
    order = [r.threshold for r in outcome.results]
    assert order == sorted(order, reverse=True)


def test_too_few_denials_is_skipped_not_failed() -> None:
    """A threshold denying 3 claims cannot support a rate claim, but it is
    not evidence against lower thresholds either - the sequence must
    continue rather than stop."""
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.10, delta=0.10, min_denied=50)
    skipped = [r for r in outcome.results if r.skipped_reason is not None]
    assert skipped
    # The applied floor is max(caller's floor, the power floor), so skips
    # are judged against the effective one.
    assert all(r.n_denied < outcome.effective_min_denied for r in skipped)
    # Having skipped some, it still went on to test lower thresholds.
    assert any(r.skipped_reason is None for r in outcome.results)


def test_power_floor_prevents_the_sparse_end_halting_the_sweep() -> None:
    """The failure this guards against, observed while building: the most
    conservative thresholds deny the fewest claims and therefore have the
    least power, so a naive fixed sequence fails at the very top and never
    reaches thresholds that would certify comfortably.

    A threshold that cannot certify even at ZERO observed errors carries
    no evidence about lower thresholds, so it must be skipped rather than
    treated as a failure.
    """
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.03, delta=0.10, min_denied=1)

    # At alpha=0.03/delta=0.10 the zero-error floor is 76 denials, well
    # above the caller's floor of 1.
    assert outcome.effective_min_denied >= 76

    for result in outcome.results:
        if result.skipped_reason is not None:
            assert result.n_denied < outcome.effective_min_denied
        else:
            assert result.n_denied >= outcome.effective_min_denied


def test_power_floor_is_at_least_the_callers_min_denied() -> None:
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.20, delta=0.20, min_denied=500)
    assert outcome.effective_min_denied >= 500


def test_min_denied_floor_is_respected() -> None:
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.10, delta=0.10, min_denied=200)
    for t in outcome.certified_thresholds:
        result = outcome.result_at(t)
        assert result is not None
        assert result.n_denied >= 200


def test_both_conditional_and_unconditional_risks_are_reported() -> None:
    """Sec.4 L3 honesty note #1: the conditional rate is what a merchant
    cares about but is a ratio of two random quantities; the unconditional
    one is covered by the clean guarantee. Both must be present."""
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.10, delta=0.10)
    tested = [r for r in outcome.results if r.skipped_reason is None]
    assert tested
    for result in tested:
        assert 0.0 <= result.conditional_rate <= 1.0
        assert 0.0 <= result.unconditional_rate <= 1.0
        assert 0.0 <= result.conditional_pvalue <= 1.0
        assert 0.0 <= result.unconditional_pvalue <= 1.0


def test_unconditional_rate_never_exceeds_conditional() -> None:
    """Same numerator, larger denominator - this must hold identically,
    and a violation means the two are computed inconsistently."""
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.10, delta=0.10)
    for result in outcome.results:
        if result.skipped_reason is None:
            assert result.unconditional_rate <= result.conditional_rate + 1e-12


def test_stricter_alpha_certifies_no_more_than_looser() -> None:
    p, y = _separable()
    strict = certify_thresholds(p, y, alpha=0.03, delta=0.10)
    loose = certify_thresholds(p, y, alpha=0.15, delta=0.10)
    assert len(strict.certified_thresholds) <= len(loose.certified_thresholds)


def test_stricter_delta_certifies_no_more_than_looser() -> None:
    p, y = _separable()
    strict = certify_thresholds(p, y, alpha=0.10, delta=0.01)
    loose = certify_thresholds(p, y, alpha=0.10, delta=0.30)
    assert len(strict.certified_thresholds) <= len(loose.certified_thresholds)


def test_certified_threshold_actually_has_low_empirical_rate() -> None:
    """Sanity: a certified threshold's realised FDR should sit at or below
    alpha. It is not guaranteed pointwise, but a certificate on a
    threshold whose empirical rate exceeds alpha would be nonsense."""
    p, y = _separable()
    outcome = certify_thresholds(p, y, alpha=0.10, delta=0.10)
    for t in outcome.certified_thresholds:
        result = outcome.result_at(t)
        assert result is not None
        assert result.conditional_rate <= outcome.alpha


def test_describe_reports_the_operating_point() -> None:
    p, y = _separable()
    text = certify_thresholds(p, y, alpha=0.10, delta=0.10).describe()
    assert "certified" in text
    assert "n_denied" in text


def test_describe_explains_an_empty_set() -> None:
    p, y = _useless()
    text = certify_thresholds(p, y, alpha=0.03, delta=0.10).describe()
    assert "NO certified threshold" in text


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        certify_thresholds(np.array([0.5, 0.6]), np.array([1]))


def test_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        certify_thresholds(np.array([]), np.array([]))


def test_custom_grid_is_honoured() -> None:
    p, y = _separable()
    grid = np.array([0.90, 0.95, 0.99])
    outcome = certify_thresholds(p, y, grid=grid, alpha=0.10, delta=0.10)
    assert {r.threshold for r in outcome.results} <= set(grid.tolist())


# --- the pre-committed ladder ----------------------------------------


def test_ladder_returns_the_first_rung_that_certifies() -> None:
    p, y = _separable()
    ladder = [(0.001, 0.01), (0.03, 0.10), (0.10, 0.10)]
    chosen, attempts = walk_alpha_delta_ladder(p, y, ladder)
    assert chosen is not None
    assert not chosen.is_empty
    # Every rung before the chosen one must have been tried and failed.
    index = next(i for i, a in enumerate(attempts) if a is chosen)
    assert all(attempts[i].is_empty for i in range(index))


def test_ladder_records_every_attempt_including_failures() -> None:
    """Reporting only the rung that worked would misrepresent how hard
    the guarantee was to obtain (docs/PREREGISTRATION.md)."""
    p, y = _separable()
    ladder = [(0.001, 0.01), (0.10, 0.10)]
    _, attempts = walk_alpha_delta_ladder(p, y, ladder)
    assert len(attempts) >= 2
    assert isinstance(attempts[0], CertifiedSet)


def test_ladder_returns_none_when_nothing_certifies() -> None:
    p, y = _useless()
    chosen, attempts = walk_alpha_delta_ladder(p, y, [(0.001, 0.01), (0.005, 0.01)])
    assert chosen is None
    assert len(attempts) == 2
    assert all(a.is_empty for a in attempts)
