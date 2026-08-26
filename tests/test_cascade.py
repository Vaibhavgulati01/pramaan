"""Cost-ordered cascade with early exit.

Three properties carry the weight, and all three fail silently when
broken:

1. An early-exiting claim is still recorded in the reuse index. Otherwise
   a fraudster whose first submission exits early makes their own
   recycled follow-up undetectable.
2. Skipped pillars leave NaN, not 0.0 - a zero is a measurement, and a
   model cannot tell it apart from a real one.
3. The feature schema is identical regardless of which stages ran, or
   the fusion model sees a different vector shape per claim.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from pramaan.cascade.cascade import (
    FEATURE_KEYS,
    Cascade,
    CascadeConfig,
    Stage,
    default_interim_scorer,
)

BASE = datetime(2026, 1, 1)


def _image(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:128].astype(np.float64)
    base = 128 + 60 * np.sin(xx / 9.0) + 40 * np.cos(yy / 6.0)
    arr = np.clip(base[..., None] + rng.normal(0, 5, (96, 128, 3)), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _process(cascade: Cascade, claim_id: str, day: int, raw: bytes,
             claimant: str = "alice", merchant: str = "m1"):
    ts = BASE + timedelta(days=day)
    return cascade.process(
        claim_id=claim_id,
        claimant_id=claimant,
        merchant_id=merchant,
        category="electronics",
        timestamp=ts,
        order_date=ts - timedelta(days=2),
        order_value=1500.0,
        raw_bytes=raw,
    )


def test_runs_all_stages_when_early_exit_disabled() -> None:
    cascade = Cascade(CascadeConfig(enable_early_exit=False))
    result = _process(cascade, "c1", 0, _image())
    assert result.stages_run == [Stage.STAGE_1_CHEAP, Stage.STAGE_2_FORENSICS, Stage.STAGE_3_REUSE]
    assert result.ran_all_stages


def test_feature_schema_is_identical_regardless_of_stages_run() -> None:
    """The fusion model needs one fixed vector shape. A claim that exits
    at stage 1 must still carry every key."""
    full = Cascade(CascadeConfig(enable_early_exit=False))
    early = Cascade(CascadeConfig(tau_exit_lo=0.99, tau_exit_hi=0.0))  # always exits

    a = _process(full, "c1", 0, _image(1))
    b = _process(early, "c1", 0, _image(1))

    assert set(a.features) == set(FEATURE_KEYS)
    assert set(b.features) == set(FEATURE_KEYS)
    assert b.exited_at_stage == Stage.STAGE_1_CHEAP


def test_skipped_pillars_are_nan_not_zero() -> None:
    """A zero is a value. If a skipped forensics pillar reported 0.0 for
    thumbnail_mismatch, the model would read it as 'measured, and clean'
    rather than 'never looked'."""
    cascade = Cascade(CascadeConfig(tau_exit_lo=0.99, tau_exit_hi=0.0))
    result = _process(cascade, "c1", 0, _image())
    assert result.exited_at_stage == Stage.STAGE_1_CHEAP

    assert np.isnan(result.features["forensics_thumbnail_mismatch"])
    assert np.isnan(result.features["reuse_n_matches"])
    # Stage 1 did run, so its features are real numbers.
    assert not np.isnan(result.features["provenance_has_manifest"])
    assert not np.isnan(result.features["behaviour_n_prior_claims"])


def test_early_exiting_claim_is_still_indexed_for_reuse() -> None:
    """THE subtle one. A claim that exits early must still enter the reuse
    index, or a later claim recycling its image finds nothing. Without
    this, early exit silently creates the blind spot P3 exists to close.
    """
    cascade = Cascade(CascadeConfig(tau_exit_lo=0.99, tau_exit_hi=0.0))
    shared = _image(7)

    first = _process(cascade, "c1", 0, shared, claimant="alice")
    assert first.exited_at_stage == Stage.STAGE_1_CHEAP
    assert len(cascade.reuse_index) == 1, "early-exiting claim was not indexed"

    # Now a second claim with the same image, forced through all stages.
    cascade.config.tau_exit_lo = 0.0
    cascade.config.tau_exit_hi = 1.0
    second = _process(cascade, "c2", 5, shared, claimant="bob")
    assert second.ran_all_stages
    assert second.reuse is not None
    assert second.reuse.matched_prior_claim, "could not match a claim that exited early"


def test_index_stays_complete_across_a_mixed_stream() -> None:
    cascade = Cascade(CascadeConfig(tau_exit_lo=0.4, tau_exit_hi=0.6))
    for i in range(8):
        _process(cascade, f"c{i}", i, _image(i))
    assert len(cascade.reuse_index) == 8


def test_records_which_stage_the_claim_exited_at() -> None:
    """The certified guarantee must be computed on the cascade as
    deployed, which requires knowing what actually ran per claim."""
    cascade = Cascade(CascadeConfig(tau_exit_lo=0.99, tau_exit_hi=0.0))
    result = _process(cascade, "c1", 0, _image())
    assert result.exited_at_stage == Stage.STAGE_1_CHEAP
    assert Stage.STAGE_2_FORENSICS not in result.stages_run
    assert Stage.STAGE_3_REUSE not in result.stages_run


def test_compute_ms_is_recorded_and_positive() -> None:
    cascade = Cascade(CascadeConfig(enable_early_exit=False))
    result = _process(cascade, "c1", 0, _image())
    assert result.compute_ms > 0


def test_early_exit_is_cheaper_than_a_full_run() -> None:
    full = Cascade(CascadeConfig(enable_early_exit=False))
    early = Cascade(CascadeConfig(tau_exit_lo=0.99, tau_exit_hi=0.0))
    raw = _image(3)
    full_ms = _process(full, "c1", 0, raw).compute_ms
    early_ms = _process(early, "c1", 0, raw).compute_ms
    assert early_ms < full_ms


def test_behaviour_accumulates_across_claims() -> None:
    cascade = Cascade(CascadeConfig(enable_early_exit=False))
    _process(cascade, "c1", 0, _image(1), claimant="alice")
    second = _process(cascade, "c2", 3, _image(2), claimant="alice")
    assert second.behaviour is not None
    assert second.behaviour.n_prior_claims == 1


# --- ablation switches (Phase 6 leave-one-pillar-out) -----------------


@pytest.mark.parametrize(
    ("switch", "prefix"),
    [
        ("enable_provenance", "provenance_"),
        ("enable_forensics", "forensics_"),
        ("enable_behaviour", "behaviour_"),
    ],
)
def test_disabling_a_pillar_leaves_its_features_unobserved(switch: str, prefix: str) -> None:
    config = CascadeConfig(enable_early_exit=False)
    setattr(config, switch, False)
    result = _process(Cascade(config), "c1", 0, _image())
    disabled = [v for k, v in result.features.items() if k.startswith(prefix)]
    assert disabled and all(np.isnan(v) for v in disabled)


def test_disabling_reuse_also_skips_indexing() -> None:
    cascade = Cascade(CascadeConfig(enable_early_exit=False, enable_reuse=False))
    _process(cascade, "c1", 0, _image())
    assert len(cascade.reuse_index) == 0

    second = _process(cascade, "c2", 1, _image(2))
    assert np.isnan(second.features["reuse_n_matches"])
    assert np.isnan(second.features["ring_in_ring"])


# --- interim scorer ---------------------------------------------------


def test_interim_scorer_returns_a_probability() -> None:
    assert 0.0 <= default_interim_scorer({}) <= 1.0


def test_interim_scorer_treats_nan_as_absent_not_as_zero() -> None:
    """NaN must not propagate into the score, or every early-exit decision
    becomes NaN-poisoned and `_should_exit` silently stops firing."""
    features = dict.fromkeys(FEATURE_KEYS, float("nan"))
    score = default_interim_scorer(features)
    assert not np.isnan(score)
    assert 0.0 <= score <= 1.0


def test_reuse_raises_the_interim_score() -> None:
    clean = default_interim_scorer({})
    reused = default_interim_scorer(
        {"reuse_matched_prior_claim": 1.0, "reuse_n_distinct_claimants_sharing": 2.0}
    )
    assert reused > clean


def test_declared_ai_generation_raises_the_score_sharply() -> None:
    assert default_interim_scorer({"provenance_declares_ai_generation": 1.0}) > 0.85


def test_camera_capture_lowers_the_score() -> None:
    assert default_interim_scorer({"provenance_declares_camera_capture": 1.0}) < 0.5
