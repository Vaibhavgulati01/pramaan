"""Evaluation infrastructure: bootstrap CIs, baselines, ablations,
negative controls, shift perturbations.

The recurring risk in evaluation code is that it produces a number
whatever you feed it. These tests check the harness fails when it should:
degenerate resamples are skipped rather than counted, controls detect
absent signal, and the source-controlled ablation is genuinely separate
from the confounded one.
"""

from __future__ import annotations

import io
import random

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import average_precision_score

from benchmarks.baselines.models import (
    ConstantBaseline,
    FeatureSubsetBaseline,
    RulesEngineBaseline,
    build_baselines,
)
from eval.ablations import PILLAR_ABLATIONS, run_ablations
from eval.bootstrap import bootstrap_difference, bootstrap_metric, bootstrap_scalar
from eval.negative_controls import (
    label_shuffle_control,
    random_feature_control,
    temporal_leak_control,
)
from eval.shift_matrix import (
    SHIFT_CONDITIONS,
    ShiftCell,
    ShiftMatrix,
    evaluate_frozen_certificate,
)
from pramaan.cascade.cascade import FEATURE_KEYS

RNG = np.random.default_rng(11)


def _dataset(n: int = 400):
    labels = (RNG.uniform(0, 1, n) < 0.15).astype(int)
    frame = pd.DataFrame(
        {name: RNG.normal(0, 1, n) for name in FEATURE_KEYS}, columns=list(FEATURE_KEYS)
    )
    # Give reuse and behaviour genuine signal so ablations have something
    # to remove.
    frame["reuse_n_distinct_claimants_sharing"] = labels * 2.0 + RNG.normal(0, 0.4, n)
    frame["behaviour_n_prior_claims_30d"] = labels * 1.5 + RNG.normal(0, 0.6, n)
    groups = np.array(["electronics|high"] * n)
    return frame, labels, groups


# --- bootstrap --------------------------------------------------------


def test_ci_contains_the_point_estimate() -> None:
    _, labels, _ = _dataset()
    scores = labels + RNG.normal(0, 0.5, len(labels))
    ci = bootstrap_metric(average_precision_score, labels, scores, n_resamples=200)
    assert ci.lower <= ci.point <= ci.upper


def test_ci_narrows_with_more_data() -> None:
    """A wider interval on less data is the whole reason CIs are here."""
    def ci_width(n: int) -> float:
        labels = (RNG.uniform(0, 1, n) < 0.3).astype(int)
        scores = labels + RNG.normal(0, 0.8, n)
        return bootstrap_metric(average_precision_score, labels, scores, 300).width

    assert ci_width(1200) < ci_width(120)


def test_degenerate_resamples_are_skipped_not_counted() -> None:
    """A resample with one class makes ranking metrics undefined.
    Skipping is honest; substituting 0 or 0.5 would shift the interval."""
    labels = np.array([0] * 60 + [1])  # single positive: many resamples degenerate
    scores = RNG.uniform(0, 1, 61)
    ci = bootstrap_metric(average_precision_score, labels, scores, n_resamples=300)
    assert ci.n_resamples < 300


def test_empty_input_yields_nan_not_a_number() -> None:
    ci = bootstrap_metric(average_precision_score, np.array([]), np.array([]), 100)
    assert np.isnan(ci.point)


def test_paired_difference_is_tighter_than_comparing_two_intervals() -> None:
    """Paired resampling on the same claims is the correct comparison and
    much tighter than eyeballing two marginal intervals."""
    _, labels, _ = _dataset(600)
    a = labels + RNG.normal(0, 0.5, len(labels))
    b = labels + RNG.normal(0, 0.55, len(labels))

    paired = bootstrap_difference(average_precision_score, labels, a, b, 300)
    ci_a = bootstrap_metric(average_precision_score, labels, a, 300)
    ci_b = bootstrap_metric(average_precision_score, labels, b, 300)
    assert paired.width < ci_a.width + ci_b.width


def test_overlap_helper_is_symmetric() -> None:
    _, labels, _ = _dataset()
    scores = labels + RNG.normal(0, 0.5, len(labels))
    a = bootstrap_metric(average_precision_score, labels, scores, 200)
    b = bootstrap_metric(average_precision_score, labels, scores, 200)
    assert a.overlaps(b) == b.overlaps(a)


def test_scalar_bootstrap_on_costs() -> None:
    values = RNG.uniform(0, 5000, 500)
    ci = bootstrap_scalar(values, 300)
    assert ci.lower <= ci.point <= ci.upper
    assert ci.point == pytest.approx(values.mean())


# --- baselines --------------------------------------------------------


def test_every_baseline_produces_probabilities() -> None:
    frame, labels, _ = _dataset()
    for baseline in build_baselines():
        baseline.fit(frame, labels)
        scores = baseline.predict_proba(frame)
        assert scores.shape == (len(frame),)
        assert ((scores >= 0) & (scores <= 1)).all(), baseline.name


def test_constant_baselines_are_constant() -> None:
    frame, _, _ = _dataset()
    assert (ConstantBaseline(0.0, "approve_all").predict_proba(frame) == 0.0).all()
    assert (ConstantBaseline(1.0, "deny_all").predict_proba(frame) == 1.0).all()


def test_rules_engine_is_not_fitted() -> None:
    """Hand-specified on purpose: fitting the thresholds would produce a
    small logistic model wearing a rules costume, and beating that would
    prove nothing about beating rules."""
    frame, labels, _ = _dataset()
    engine = RulesEngineBaseline()
    before = engine.predict_proba(frame)
    engine.fit(frame, labels)
    assert np.array_equal(before, engine.predict_proba(frame))


def test_resnet_style_baseline_is_never_called_cnnspot() -> None:
    """Naming discipline: it is not a faithful CNNSpot reimplementation,
    and small overclaims are what make a reader doubt the large ones."""
    names = {b.name for b in build_baselines()}
    assert "cnnspot" not in {n.lower() for n in names}
    resnet = next(b for b in build_baselines() if b.name == "resnet_style_pixel_probe")
    assert "NOT a faithful reimplementation" in resnet.description


def test_feature_subset_baseline_rejects_an_unknown_prefix() -> None:
    frame, labels, _ = _dataset()
    with pytest.raises(ValueError, match="no features with prefix"):
        FeatureSubsetBaseline("nonexistent_", "x", "y").fit(frame, labels)


def test_unfitted_subset_baseline_raises() -> None:
    frame, _, _ = _dataset()
    with pytest.raises(RuntimeError, match="not fitted"):
        FeatureSubsetBaseline("reuse_", "x", "y").predict_proba(frame)


# --- ablations --------------------------------------------------------


def test_ablations_cover_every_pillar() -> None:
    assert {"no_provenance", "no_forensics", "no_reuse", "no_behaviour", "no_rings"} <= set(
        PILLAR_ABLATIONS
    )


def test_removing_a_signal_pillar_hurts() -> None:
    frame, labels, groups = _dataset(500)
    from pramaan.fusion.model import FusionConfig

    suite = run_ablations(frame, labels, groups, config=FusionConfig(n_estimators=60, n_folds=3))
    reuse = next(r for r in suite.results if r.name == "no_reuse" and r.subset == "all")
    assert reuse.delta_pr_auc < 0


def test_source_controlled_subset_is_reported_separately() -> None:
    """The full-corpus ablation is confounded by source dataset; reporting
    only it would overstate the pixel pillars roughly threefold."""
    frame, labels, groups = _dataset(500)
    from pramaan.fusion.model import FusionConfig

    abo_mask = np.arange(len(labels)) % 2 == 0
    suite = run_ablations(
        frame, labels, groups, abo_mask=abo_mask,
        config=FusionConfig(n_estimators=60, n_folds=3),
    )
    assert suite.for_subset("all")
    assert suite.for_subset("abo_only")
    assert "confounded" in str(suite.as_dict()["interpretation_warning"])


def test_ablation_rejects_a_prefix_matching_nothing() -> None:
    frame, labels, groups = _dataset(200)
    from eval.ablations import _run_one
    from pramaan.fusion.model import FusionConfig

    with pytest.raises(ValueError, match="matched no columns"):
        _run_one("bogus", "all", frame, labels, groups, "nope_", FusionConfig(n_estimators=10))


# --- negative controls ------------------------------------------------


def test_label_shuffle_collapses_to_prevalence() -> None:
    """If a model finds signal in shuffled labels, something is carrying
    label information it should not have."""
    frame, labels, groups = _dataset(600)
    result = label_shuffle_control(frame, labels, groups, baseline_pr_auc=0.5)
    assert result.passed, result.note
    assert result.pr_auc < result.prevalence + 0.10


def test_random_features_collapse_to_prevalence() -> None:
    frame, labels, groups = _dataset(600)
    result = random_feature_control(frame, labels, groups, baseline_pr_auc=0.5)
    assert result.passed, result.note


def test_temporal_leak_control_measures_inflation() -> None:
    """A leaky index matches against the whole corpus including the
    future. Measuring the gap demonstrates the bug is understood."""
    from datetime import datetime, timedelta

    base = datetime(2026, 1, 1)
    shared = 0xABCD_1234_5678_9F01
    ids = [f"c{i}" for i in range(6)]
    phashes = dict.fromkeys(ids, shared)  # every claim shares an image

    measurement = temporal_leak_control(
        phashes=phashes,
        claim_order=ids,
        labels_by_claim=dict.fromkeys(ids, 0),
        claimants={c: f"claimant_{c}" for c in ids},
        merchants=dict.fromkeys(ids, "m1"),
        timestamps={c: base + timedelta(days=i) for i, c in enumerate(ids)},
    )
    # The first claim can never match a prior; the leaky version matches
    # everything, so inflation must be strictly positive.
    assert measurement.inflation > 0
    assert measurement.leaked_match_rate == 1.0
    assert measurement.honest_match_rate < 1.0


# --- shift matrix -----------------------------------------------------


def _jpeg(size=(120, 90)) -> bytes:
    arr = RNG.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_every_shift_condition_produces_a_decodable_image() -> None:
    raw = _jpeg()
    rng = random.Random(0)
    for name, transform in SHIFT_CONDITIONS.items():
        if transform is None:
            continue  # supplied by the generator-holdout split, not a transform
        out = transform(raw, rng)
        with Image.open(io.BytesIO(out)) as img:
            assert img.size[0] > 0, name


def test_shift_conditions_cover_the_specified_eight() -> None:
    assert set(SHIFT_CONDITIONS) == {
        "in_distribution",
        "unseen_generator_families",
        "jpeg_q60",
        "jpeg_q40",
        "metadata_stripped",
        "centre_crop_90",
        "screenshot_round_trip",
        "colour_jitter_rotate",
    }


def test_frozen_certificate_detects_a_violation() -> None:
    probabilities = np.array([0.95] * 100)
    labels = np.array([0] * 50 + [1] * 50)  # 50% of denials are legitimate
    fdr, holds, n_denied = evaluate_frozen_certificate(probabilities, labels, 0.9, alpha=0.03)
    assert n_denied == 100
    assert fdr == pytest.approx(0.5)
    assert not holds


def test_frozen_certificate_holds_when_denials_are_clean() -> None:
    probabilities = np.array([0.95] * 100)
    labels = np.ones(100, dtype=int)
    fdr, holds, _ = evaluate_frozen_certificate(probabilities, labels, 0.9, alpha=0.03)
    assert fdr == 0.0 and holds


def test_denying_nothing_is_flagged_by_n_denied() -> None:
    """Denying nothing cannot violate a false-denial bound, but it also
    demonstrates nothing - a reader must be able to see that."""
    _, holds, n_denied = evaluate_frozen_certificate(
        np.zeros(50), np.ones(50, dtype=int), 0.9, alpha=0.03
    )
    assert holds and n_denied == 0


def test_matrix_renders_markdown_with_both_experiments() -> None:
    matrix = ShiftMatrix(alpha=0.03, delta=0.10)
    matrix.cells.append(
        ShiftCell("jpeg_q40", 100, 0.4, 0.5, 0.09, False, recertified=True,
                  recertified_alpha=0.10)
    )
    text = matrix.to_markdown()
    assert "jpeg_q40" in text
    assert "❌" in text
    assert "recovered" in text
    assert matrix.cells[0].recoverable
