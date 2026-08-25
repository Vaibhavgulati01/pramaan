from pramaan.risk.hb_pvalue import hb_pvalue
from pramaan.risk.power_analysis import (
    min_n_for_rhat,
    min_n_zero_errors,
    power_curve,
    required_full_corpus_size,
    resolve_primary_target,
)


def test_min_n_zero_errors_hand_computed() -> None:
    assert min_n_zero_errors(0.03, 0.10, min_denied=1) == 76


def test_min_n_zero_errors_respects_ltt_floor() -> None:
    # A generous (alpha, delta) would mathematically need fewer than 30,
    # but LTT itself refuses to certify below n=30 (spec Sec.4 L3).
    assert min_n_zero_errors(0.5, 0.5, min_denied=30) == 30


def test_min_n_for_rhat_matches_closed_form_at_zero() -> None:
    assert min_n_for_rhat(0.03, 0.10, r_hat=0.0, min_denied=1) == min_n_zero_errors(
        0.03, 0.10, min_denied=1
    )


def test_min_n_for_rhat_returned_n_actually_certifies() -> None:
    alpha, delta, r_hat = 0.03, 0.10, 0.009
    n = min_n_for_rhat(alpha, delta, r_hat, min_denied=1)
    assert n is not None
    assert hb_pvalue(r_hat, n, alpha) <= delta
    assert hb_pvalue(r_hat, n - 1, alpha) > delta  # n is genuinely minimal


def test_min_n_for_rhat_none_when_rhat_exceeds_alpha() -> None:
    assert min_n_for_rhat(0.03, 0.10, r_hat=0.05) is None


def test_larger_alpha_needs_fewer_samples() -> None:
    n_strict = min_n_zero_errors(0.03, 0.10, min_denied=1)
    n_loose = min_n_zero_errors(0.10, 0.10, min_denied=1)
    assert n_loose < n_strict


def test_power_curve_shape() -> None:
    points = power_curve(alphas=[0.03, 0.05], delta=0.10, r_hat_fractions_of_alpha=(0.0, 0.3))
    assert len(points) == 4
    for pt in points:
        assert pt.min_denial_set_n is None or pt.min_denial_set_n >= 30


def test_required_full_corpus_size_scales_with_deny_rate() -> None:
    generous = required_full_corpus_size(0.03, 0.10, assumed_deny_rate=0.16)
    stingy = required_full_corpus_size(0.03, 0.10, assumed_deny_rate=0.02)
    assert generous.required_full_corpus_n is not None
    assert stingy.required_full_corpus_n is not None
    # A lower deny rate means the same denial-set n needs a bigger corpus.
    assert stingy.required_full_corpus_n > generous.required_full_corpus_n


def test_resolve_primary_target_walks_ladder_and_records_attempts() -> None:
    ladder = [(0.03, 0.10), (0.05, 0.10), (0.10, 0.10), (0.10, 0.20)]
    chosen, attempts = resolve_primary_target(ladder, max_practical_full_n=1)
    # An absurdly small practical cap forces every rung to fail; the
    # ladder discipline still returns the last attempt rather than None.
    assert len(attempts) == len(ladder)
    assert chosen is attempts[-1]


def test_resolve_primary_target_picks_first_practical_rung() -> None:
    ladder = [(0.03, 0.10), (0.05, 0.10), (0.10, 0.10), (0.10, 0.20)]
    chosen, attempts = resolve_primary_target(ladder, max_practical_full_n=60_000)
    assert chosen.required_full_corpus_n is not None
    assert chosen.required_full_corpus_n <= 60_000
    # The chosen rung must be the first one (in ladder order) that is practical.
    idx = attempts.index(chosen)
    for earlier in attempts[:idx]:
        assert earlier.required_full_corpus_n is None or earlier.required_full_corpus_n > 60_000
