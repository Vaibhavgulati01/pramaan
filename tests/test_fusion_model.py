"""Fusion model: schema, monotone constraints, and split discipline.

Two properties are guarantees rather than niceties:

- **Monotone constraints actually bind.** They are the reason a reason
  code can be trusted and the reason a thin data slice cannot teach the
  model that more claimants sharing an image is *less* suspicious. A
  constraint that is declared but not applied is worse than none, because
  the docs claim it.
- **The calibration split is never used for fitting.** It is reserved for
  Learn-then-Test, and spending it here would void the guarantee in
  docs/GUARANTEE.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pramaan.fusion.model import FusionConfig, FusionModel, SplitDisciplineError
from pramaan.fusion.schema import (
    FEATURE_SCHEMA_VERSION,
    MONOTONE_CONSTRAINTS,
    Monotone,
    feature_names,
    monotone_constraint_vector,
    validate_schema,
)

RNG = np.random.default_rng(7)


def _synthetic(n: int = 600) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A dataset where reuse genuinely drives the label, so a fitted model
    has real signal to find."""
    names = feature_names()
    data = {name: RNG.normal(0, 1, n) for name in names}

    sharing = RNG.integers(0, 4, n).astype(float)
    data["reuse_n_distinct_claimants_sharing"] = sharing
    data["reuse_n_matches"] = sharing + RNG.integers(0, 2, n)
    data["ring_is_first_seen"] = (RNG.uniform(0, 1, n) < 0.3).astype(float)

    frame = pd.DataFrame(data, columns=list(names))
    risk = 0.1 + 0.25 * sharing - 0.15 * frame["ring_is_first_seen"]
    labels = (RNG.uniform(0, 1, n) < np.clip(risk, 0.02, 0.95)).astype(int)

    groups = np.array(
        [f"{c}|{b}" for c, b in zip(
            RNG.choice(["electronics", "apparel"], n),
            RNG.choice(["low", "high"], n),
            strict=True,
        )]
    )
    return frame, labels, groups


# --- schema -----------------------------------------------------------


def test_schema_validates() -> None:
    validate_schema()


def test_constraint_vector_matches_feature_count() -> None:
    """LightGBM's monotone_constraints is POSITIONAL, so a length or order
    mismatch silently applies constraints to the wrong features."""
    assert len(monotone_constraint_vector()) == len(feature_names())


def test_constraints_land_on_the_intended_features() -> None:
    names = feature_names()
    vector = monotone_constraint_vector()
    by_name = dict(zip(names, vector, strict=True))
    for constrained in MONOTONE_CONSTRAINTS:
        assert by_name[constrained.name] == int(constrained.direction)


def test_unconstrained_features_are_zero() -> None:
    constrained = {c.name for c in MONOTONE_CONSTRAINTS}
    by_name = dict(zip(feature_names(), monotone_constraint_vector(), strict=True))
    for name, value in by_name.items():
        if name not in constrained:
            assert value == 0


def test_every_constraint_carries_a_rationale() -> None:
    """A monotone constraint is a claim about the world. If it cannot be
    justified in a sentence it should not be imposed."""
    for constrained in MONOTONE_CONSTRAINTS:
        assert len(constrained.rationale) > 40


def test_similarity_and_distance_constraints_oppose_each_other() -> None:
    """reuse_max_clip_similarity and reuse_best_hamming measure the same
    thing in opposite directions; constraining them the same way would be
    incoherent."""
    by_name = {c.name: c.direction for c in MONOTONE_CONSTRAINTS}
    assert by_name["reuse_max_clip_similarity"] == Monotone.RISK_INCREASES
    assert by_name["reuse_best_hamming"] == Monotone.RISK_DECREASES


def test_validate_schema_rejects_an_unknown_feature(monkeypatch) -> None:
    from pramaan.fusion import schema

    bogus = (*MONOTONE_CONSTRAINTS, schema.ConstrainedFeature(
        "feature_that_does_not_exist", Monotone.RISK_INCREASES, "x" * 50
    ))
    monkeypatch.setattr(schema, "MONOTONE_CONSTRAINTS", bogus)
    with pytest.raises(ValueError, match="unknown features"):
        schema.validate_schema()


# --- fitting ----------------------------------------------------------


def test_fit_and_predict_returns_probabilities() -> None:
    frame, labels, groups = _synthetic()
    model = FusionModel(FusionConfig(n_estimators=40, n_folds=3)).fit(frame, labels, groups)
    p = model.predict(frame, groups)
    assert p.shape == (len(frame),)
    assert ((p >= 0) & (p <= 1)).all()


def test_model_learns_the_signal() -> None:
    frame, labels, groups = _synthetic()
    model = FusionModel(FusionConfig(n_estimators=60, n_folds=3)).fit(frame, labels, groups)
    p = model.predict(frame, groups)
    # Claims with more claimants sharing must score higher on average.
    low = p[frame["reuse_n_distinct_claimants_sharing"] == 0].mean()
    high = p[frame["reuse_n_distinct_claimants_sharing"] >= 2].mean()
    assert high > low


def test_monotone_constraint_actually_binds() -> None:
    """Sweep one constrained feature and assert the fitted response is
    non-decreasing. This is the test that catches a constraint being
    declared in the docs but not reaching LightGBM."""
    frame, labels, groups = _synthetic()
    model = FusionModel(FusionConfig(n_estimators=80, n_folds=3)).fit(frame, labels, groups)

    probe = frame.iloc[[0]].copy()
    scores = []
    for value in range(0, 8):
        probe = probe.copy()
        probe["reuse_n_distinct_claimants_sharing"] = float(value)
        scores.append(float(model.predict_uncalibrated(probe)[0]))

    assert all(b >= a - 1e-9 for a, b in zip(scores, scores[1:], strict=False)), scores


def test_decreasing_constraint_binds_in_the_other_direction() -> None:
    frame, labels, groups = _synthetic()
    model = FusionModel(FusionConfig(n_estimators=80, n_folds=3)).fit(frame, labels, groups)

    probe = frame.iloc[[0]].copy()
    scores = []
    for value in np.linspace(0, 64, 12):
        probe = probe.copy()
        probe["reuse_best_hamming"] = float(value)
        scores.append(float(model.predict_uncalibrated(probe)[0]))

    assert all(b <= a + 1e-9 for a, b in zip(scores, scores[1:], strict=False)), scores


def test_disabling_constraints_is_recorded_in_the_report() -> None:
    frame, labels, groups = _synthetic()
    model = FusionModel(
        FusionConfig(n_estimators=30, n_folds=3, apply_monotone_constraints=False)
    ).fit(frame, labels, groups)
    assert model.report is not None
    assert model.report.monotone_constraints_applied == 0


def test_calibration_improves_or_matches_oof_brier() -> None:
    frame, labels, groups = _synthetic(800)
    model = FusionModel(FusionConfig(n_estimators=60, n_folds=4)).fit(frame, labels, groups)
    assert model.report is not None
    assert model.report.oof_brier_calibrated <= model.report.oof_brier_uncalibrated + 1e-6


# --- split discipline -------------------------------------------------


def test_fit_rejects_the_calibration_split() -> None:
    """The calibration split belongs to Learn-then-Test. Spending it on
    model fitting would void docs/GUARANTEE.md, so the code refuses
    rather than relying on anyone remembering."""
    frame, labels, groups = _synthetic(200)
    splits = np.array(["train"] * 150 + ["calibration"] * 50)
    with pytest.raises(SplitDisciplineError, match="calibration"):
        FusionModel(FusionConfig(n_estimators=10, n_folds=2)).fit(
            frame, labels, groups, splits=splits
        )


def test_fit_rejects_the_test_split() -> None:
    frame, labels, groups = _synthetic(200)
    splits = np.array(["train"] * 150 + ["test"] * 50)
    with pytest.raises(SplitDisciplineError):
        FusionModel(FusionConfig(n_estimators=10, n_folds=2)).fit(
            frame, labels, groups, splits=splits
        )


def test_fit_accepts_pure_train() -> None:
    frame, labels, groups = _synthetic(200)
    model = FusionModel(FusionConfig(n_estimators=10, n_folds=2)).fit(
        frame, labels, groups, splits=np.full(len(frame), "train")
    )
    assert model.report is not None


# --- robustness -------------------------------------------------------


def test_missing_feature_column_raises() -> None:
    frame, labels, groups = _synthetic(100)
    with pytest.raises(ValueError, match="missing columns"):
        FusionModel(FusionConfig(n_estimators=10, n_folds=2)).fit(
            frame.drop(columns=["ring_size"]), labels, groups
        )


def test_column_order_does_not_change_predictions() -> None:
    """Guards the positional-constraint hazard: a caller handing columns
    in a different order must get the same answer, not silently
    mis-constrained scores."""
    frame, labels, groups = _synthetic(300)
    model = FusionModel(FusionConfig(n_estimators=40, n_folds=3)).fit(frame, labels, groups)

    shuffled = frame.loc[:, list(RNG.permutation(list(frame.columns)))]
    assert np.allclose(model.predict(frame, groups), model.predict(shuffled, groups))


def test_nan_features_are_handled() -> None:
    """Skipped cascade stages leave NaN; the model must treat them as
    missing rather than failing."""
    frame, labels, groups = _synthetic(300)
    model = FusionModel(FusionConfig(n_estimators=30, n_folds=3)).fit(frame, labels, groups)

    with_nan = frame.copy()
    with_nan.loc[:, [c for c in frame.columns if c.startswith("forensics_")]] = np.nan
    assert np.isfinite(model.predict(with_nan, groups)).all()


def test_single_class_training_data_raises() -> None:
    frame, _, groups = _synthetic(100)
    with pytest.raises(ValueError, match="one class"):
        FusionModel(FusionConfig(n_estimators=10, n_folds=2)).fit(
            frame, np.zeros(len(frame), dtype=int), groups
        )


def test_predict_before_fit_raises() -> None:
    frame, _, groups = _synthetic(50)
    with pytest.raises(RuntimeError, match="not fitted"):
        FusionModel().predict(frame, groups)


def test_deterministic_across_two_fits() -> None:
    """Phase 6 asserts byte-identical metrics across runs, which requires
    the model itself be reproducible."""
    frame, labels, groups = _synthetic(400)
    config = FusionConfig(n_estimators=50, n_folds=3)
    a = FusionModel(config).fit(frame, labels, groups).predict(frame, groups)
    b = FusionModel(config).fit(frame, labels, groups).predict(frame, groups)
    assert np.array_equal(a, b)


def test_save_and_load_roundtrip(tmp_path) -> None:
    frame, labels, groups = _synthetic(300)
    model = FusionModel(FusionConfig(n_estimators=40, n_folds=3)).fit(frame, labels, groups)
    before = model.predict(frame, groups)

    model.save(tmp_path / "model")
    reloaded = FusionModel.load(tmp_path / "model")
    assert np.allclose(reloaded.predict(frame, groups), before)


def test_load_rejects_a_schema_version_mismatch(tmp_path, monkeypatch) -> None:
    """A model trained on a different feature vector must never silently
    score the current one."""
    import json

    frame, labels, groups = _synthetic(200)
    model = FusionModel(FusionConfig(n_estimators=20, n_folds=2)).fit(frame, labels, groups)
    model.save(tmp_path / "model")

    meta_path = tmp_path / "model" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = "0.0.1-ancient"
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="schema"):
        FusionModel.load(tmp_path / "model")


def test_schema_version_is_recorded_in_the_report() -> None:
    frame, labels, groups = _synthetic(150)
    model = FusionModel(FusionConfig(n_estimators=20, n_folds=2)).fit(frame, labels, groups)
    assert model.report is not None
    assert model.report.schema_version == FEATURE_SCHEMA_VERSION
