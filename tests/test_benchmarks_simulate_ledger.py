import pandas as pd

from benchmarks.simulate_ledger import (
    COHORT_FRACTIONS,
    GENERATOR_HOLDOUT,
    RING_FORMING_CLASSES,
    simulate_ledger,
)


def _ledger(n: int = 600, seed: int = 1) -> pd.DataFrame:
    return simulate_ledger(n_claims=n, merchants=["m1", "m2", "m3"], seed=seed)


def test_returns_expected_row_count_and_columns() -> None:
    df = _ledger(500)
    assert len(df) == 500
    expected_cols = {
        "claim_id",
        "cohort",
        "merchant_id",
        "fraud_class",
        "label",
        "generator_family",
        "image_group_id",
        "claimant_id",
        "phone",
        "email",
        "address",
        "pin",
        "order_value_inr",
        "category",
        "price_band",
        "order_date",
        "claim_timestamp",
    }
    assert expected_cols.issubset(df.columns)


def test_claim_ids_are_unique() -> None:
    df = _ledger(800)
    assert df["claim_id"].is_unique


def test_deterministic_for_fixed_seed() -> None:
    a = _ledger(300, seed=42)
    b = _ledger(300, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_output() -> None:
    a = _ledger(300, seed=1)
    b = _ledger(300, seed=2)
    assert not a["phone"].equals(b["phone"])


def test_label_matches_fraud_class() -> None:
    df = _ledger(500)
    legit = df[df["fraud_class"] == "legit_real_photo"]
    fraud = df[df["fraud_class"] != "legit_real_photo"]
    assert (legit["label"] == 0).all()
    assert (fraud["label"] == 1).all()


def test_generator_family_only_set_for_synthetic_image_class() -> None:
    df = _ledger(1500)
    synth = df[df["fraud_class"] == "fraud_synthetic_image"]
    other = df[df["fraud_class"] != "fraud_synthetic_image"]
    assert synth["generator_family"].notna().all()
    assert other["generator_family"].isna().all()


def test_generator_family_respects_holdout_by_cohort() -> None:
    df = _ledger(2000)
    synth = df[df["fraud_class"] == "fraud_synthetic_image"]
    for cohort, families in GENERATOR_HOLDOUT.items():
        subset = synth[synth["cohort"] == cohort]
        assert set(subset["generator_family"].unique()).issubset(set(families))


def test_cohort_fractions_are_roughly_respected() -> None:
    df = _ledger(3000, seed=7)
    counts = df["cohort"].value_counts(normalize=True)
    for cohort, target in COHORT_FRACTIONS.items():
        assert abs(counts.get(cohort, 0.0) - target) < 0.03


def test_composition_roughly_matches_target_shares() -> None:
    df = _ledger(3000, seed=7)
    shares = df["fraud_class"].value_counts(normalize=True)
    assert abs(shares.get("legit_real_photo", 0.0) - 0.85) < 0.03
    assert abs(shares.get("fraud_synthetic_image", 0.0) - 0.06) < 0.02


def test_ring_forming_claims_reuse_an_image_group_from_same_cohort() -> None:
    df = _ledger(2000, seed=3)
    ring_claims = df[df["fraud_class"].isin(RING_FORMING_CLASSES)]
    by_id = df.set_index("claim_id")
    reused = ring_claims[ring_claims["image_group_id"] != ring_claims["claim_id"]]
    assert len(reused) > 0  # with n=2000 there must be eligible originals to reuse
    for _, row in reused.iterrows():
        original = by_id.loc[row["image_group_id"]]
        assert original["cohort"] == row["cohort"]


def test_claim_timestamp_after_order_date() -> None:
    df = _ledger(400)
    ts = pd.to_datetime(df["claim_timestamp"])
    order = pd.to_datetime(df["order_date"])
    assert (ts >= order).all()


def test_repeat_claimants_stay_within_one_cohort() -> None:
    df = _ledger(2000, seed=5)
    per_claimant_cohorts = df.groupby("claimant_id")["cohort"].nunique()
    assert (per_claimant_cohorts == 1).all()


def test_order_value_within_declared_price_band_ranges() -> None:
    from benchmarks.simulate_ledger import PRICE_BAND_RANGES_INR

    df = _ledger(500)
    for band, (lo, hi) in PRICE_BAND_RANGES_INR.items():
        subset = df[df["price_band"] == band]
        assert (subset["order_value_inr"].between(lo, hi)).all()
