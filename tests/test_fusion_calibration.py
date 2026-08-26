"""Mondrian isotonic calibration and its metrics.

The property that matters most: a model well-calibrated overall can be
badly calibrated on a high-value segment, and the per-group report exists
to make that visible rather than averaged away.
"""

from __future__ import annotations

import numpy as np
import pytest

from pramaan.fusion.calibration import (
    MondrianIsotonicCalibrator,
    brier_score,
    calibration_metrics,
    evaluate_calibration,
    expected_calibration_error,
    reliability_curve,
)

RNG = np.random.default_rng(0)


def _well_calibrated(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    p = RNG.uniform(0, 1, n)
    y = (RNG.uniform(0, 1, n) < p).astype(float)
    return p, y


def _overconfident(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Predictions pushed toward the extremes relative to reality."""
    true_p = RNG.uniform(0.2, 0.8, n)
    y = (RNG.uniform(0, 1, n) < true_p).astype(float)
    p = np.clip((true_p - 0.5) * 3 + 0.5, 0.01, 0.99)
    return p, y


def test_brier_is_zero_for_perfect_predictions() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, y) == 0.0


def test_brier_is_one_for_maximally_wrong_predictions() -> None:
    y = np.array([0.0, 1.0])
    assert brier_score(1 - y, y) == 1.0


def test_ece_is_small_for_well_calibrated_predictions() -> None:
    p, y = _well_calibrated()
    ece, mce = expected_calibration_error(p, y)
    assert ece < 0.05
    assert mce < 0.15


def test_ece_is_large_for_overconfident_predictions() -> None:
    calibrated_ece, _ = expected_calibration_error(*_well_calibrated())
    over_ece, _ = expected_calibration_error(*_overconfident())
    assert over_ece > calibrated_ece
    assert over_ece > 0.05


def test_mce_is_at_least_ece() -> None:
    """MCE is a max over bins and ECE a weighted mean, so this must hold
    identically - a violation means the bins disagree between them."""
    for p, y in (_well_calibrated(), _overconfident()):
        ece, mce = expected_calibration_error(p, y)
        assert mce >= ece - 1e-12


def test_bins_are_equal_mass_not_equal_width() -> None:
    """Fraud scores pile up near zero. Equal-width bins would leave most
    bins nearly empty and let a handful of sparse high bins dominate ECE.
    """
    skewed = np.concatenate([RNG.uniform(0, 0.05, 950), RNG.uniform(0.8, 1.0, 50)])
    labels = (RNG.uniform(0, 1, 1000) < skewed).astype(float)
    curve = reliability_curve(skewed, labels, n_bins=10)
    counts = curve.bin_counts
    assert len(counts) == 10
    # Equal mass: every bin within one of 100. Equal width would give
    # something wildly lopsided instead.
    assert max(counts) - min(counts) <= 1


def test_metrics_handle_empty_input() -> None:
    ece, mce = expected_calibration_error(np.array([]), np.array([]))
    assert ece == 0.0 and mce == 0.0


# --- Mondrian calibrator ---------------------------------------------


def _grouped(n_per_cell: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two cells whose miscalibration runs in OPPOSITE directions, so a
    single global curve cannot fix both - which is the situation Mondrian
    calibration exists for."""
    p_a = RNG.uniform(0, 1, n_per_cell)
    y_a = (RNG.uniform(0, 1, n_per_cell) < np.clip(p_a * 0.4, 0, 1)).astype(float)

    p_b = RNG.uniform(0, 1, n_per_cell)
    y_b = (RNG.uniform(0, 1, n_per_cell) < np.clip(p_b * 0.5 + 0.5, 0, 1)).astype(float)

    p = np.concatenate([p_a, p_b])
    y = np.concatenate([y_a, y_b])
    g = np.array(["electronics|high"] * n_per_cell + ["apparel|low"] * n_per_cell)
    return p, y, g


def test_calibrator_improves_brier() -> None:
    p, y, g = _grouped()
    cal = MondrianIsotonicCalibrator().fit(p, y, g)
    assert brier_score(cal.predict(p, g), y) < brier_score(p, y)


def test_per_cell_calibration_beats_a_global_curve() -> None:
    """The whole argument for Mondrian: when cells are miscalibrated in
    opposite directions, one global curve averages them into being wrong
    for both."""
    p, y, g = _grouped()

    mondrian = MondrianIsotonicCalibrator(prior_strength=1.0).fit(p, y, g)
    global_only = MondrianIsotonicCalibrator(min_cell_size=10**9).fit(p, y, g)

    assert brier_score(mondrian.predict(p, g), y) < brier_score(global_only.predict(p, g), y)


def test_thin_cells_fall_back_to_the_global_curve() -> None:
    p, y, g = _grouped(n_per_cell=300)
    p = np.append(p, [0.5, 0.6])
    y = np.append(y, [1.0, 0.0])
    g = np.append(g, ["rare|band", "rare|band"])

    cal = MondrianIsotonicCalibrator(min_cell_size=20).fit(p, y, g)
    assert "rare|band" not in cal.cell_names
    assert cal.shrinkage_weight("rare|band") == 0.0
    # Still predicts, via the global curve, rather than failing.
    assert np.isfinite(cal.predict(np.array([0.5]), np.array(["rare|band"]))).all()


def test_shrinkage_weight_grows_with_cell_size() -> None:
    p, y, g = _grouped(n_per_cell=400)
    cal = MondrianIsotonicCalibrator(prior_strength=50.0).fit(p, y, g)
    weight = cal.shrinkage_weight("electronics|high")
    assert 0.0 < weight < 1.0
    assert weight == pytest.approx(400 / 450, abs=1e-6)


def test_single_class_cell_is_not_fitted() -> None:
    """Isotonic on an all-one-class cell emits a constant, which is
    strictly less informative than the global curve."""
    p = np.concatenate([RNG.uniform(0, 1, 300), RNG.uniform(0, 1, 100)])
    y = np.concatenate([(RNG.uniform(0, 1, 300) < 0.4).astype(float), np.zeros(100)])
    g = np.array(["mixed|band"] * 300 + ["allzero|band"] * 100)
    cal = MondrianIsotonicCalibrator().fit(p, y, g)
    assert "allzero|band" not in cal.cell_names
    assert "mixed|band" in cal.cell_names


def test_unseen_group_at_predict_time_uses_the_global_curve() -> None:
    p, y, g = _grouped()
    cal = MondrianIsotonicCalibrator().fit(p, y, g)
    out = cal.predict(np.array([0.3, 0.7]), np.array(["brand|new", "brand|new"]))
    assert np.isfinite(out).all()
    assert ((out >= 0) & (out <= 1)).all()


def test_output_is_always_a_probability() -> None:
    p, y, g = _grouped()
    cal = MondrianIsotonicCalibrator().fit(p, y, g)
    out = cal.predict(np.array([-0.5, 0.0, 0.5, 1.0, 1.5]), np.array(["electronics|high"] * 5))
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        MondrianIsotonicCalibrator().predict(np.array([0.5]), np.array(["a|b"]))


def test_fit_on_empty_raises() -> None:
    with pytest.raises(ValueError, match="zero samples"):
        MondrianIsotonicCalibrator().fit(np.array([]), np.array([]), np.array([]))


def test_group_key_is_stable() -> None:
    assert MondrianIsotonicCalibrator.group_key("electronics", "high") == "electronics|high"


# --- per-group reporting ---------------------------------------------


def test_per_group_report_surfaces_the_worst_cell() -> None:
    """Sec.4 L2's requirement: a system well-calibrated overall and badly
    calibrated on high-value electronics must not look fine."""
    n = 600
    good_p = RNG.uniform(0, 1, n)
    good_y = (RNG.uniform(0, 1, n) < good_p).astype(float)
    bad_p = RNG.uniform(0, 1, n)
    bad_y = (RNG.uniform(0, 1, n) < 0.9).astype(float)  # predictions unrelated to truth

    p = np.concatenate([good_p, bad_p])
    y = np.concatenate([good_y, bad_y])
    g = np.array(["apparel|low"] * n + ["electronics|high"] * n)

    report = evaluate_calibration(p, y, g)
    assert set(report.per_group) == {"apparel|low", "electronics|high"}
    worst_name, _ = report.worst_groups(1)[0]
    assert worst_name == "electronics|high"
    assert report.per_group["electronics|high"].ece > report.per_group["apparel|low"].ece


def test_small_groups_are_omitted_from_the_report() -> None:
    p, y, g = _grouped(n_per_cell=200)
    p = np.append(p, [0.5] * 5)
    y = np.append(y, [1.0] * 5)
    g = np.append(g, ["tiny|cell"] * 5)
    report = evaluate_calibration(p, y, g, min_group_size=30)
    assert "tiny|cell" not in report.per_group


def test_report_always_includes_a_global_curve() -> None:
    p, y, g = _grouped()
    assert "global" in evaluate_calibration(p, y, g).curves


def test_calibration_metrics_serialise() -> None:
    p, y = _well_calibrated()
    as_dict = calibration_metrics(p, y).as_dict()
    assert set(as_dict) == {"brier", "ece", "mce", "n", "n_bins"}
    assert all(isinstance(v, float) for v in as_dict.values())
