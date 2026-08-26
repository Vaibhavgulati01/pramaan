from datetime import datetime, timedelta

import pandas as pd
import pytest

from benchmarks.simulate_ledger import simulate_ledger
from benchmarks.splits import reconcile_splits, verify_splits


def _valid_claims() -> pd.DataFrame:
    """A minimal hand-built corpus that satisfies all 4 constraints, so
    each test below can violate exactly one and assert only that one is
    reported."""
    base = datetime(2025, 10, 1)
    return pd.DataFrame(
        [
            {
                "claim_id": "c1",
                "split": "train",
                "fraud_class": "legit_real_photo",
                "generator_family": None,
                "claim_timestamp": base,
                "image_group_id": "g1",
                "phone": "9000000001",
            },
            {
                "claim_id": "c2",
                "split": "train",
                "fraud_class": "fraud_synthetic_image",
                "generator_family": "SD15",
                "claim_timestamp": base + timedelta(days=1),
                "image_group_id": "g2",
                "phone": "9000000002",
            },
            {
                "claim_id": "c3",
                "split": "calibration",
                "fraud_class": "fraud_synthetic_image",
                "generator_family": "GLIDE",
                "claim_timestamp": base + timedelta(days=10),
                "image_group_id": "g3",
                "phone": "9000000003",
            },
            {
                "claim_id": "c4",
                "split": "test",
                "fraud_class": "fraud_synthetic_image",
                "generator_family": "Midjourney",
                "claim_timestamp": base + timedelta(days=20),
                "image_group_id": "g4",
                "phone": "9000000004",
            },
        ]
    )


def test_valid_corpus_passes_all_four_checks() -> None:
    report = verify_splits(_valid_claims())
    assert report.is_valid
    assert "OK" in report.describe()


def test_generator_holdout_violation_detected() -> None:
    claims = _valid_claims()
    # Midjourney is a TEST-only family; using it in train is a holdout leak.
    claims.loc[claims["claim_id"] == "c2", "generator_family"] = "Midjourney"
    report = verify_splits(claims)
    assert not report.is_valid
    assert len(report.generator_holdout_violations) == 1
    assert "train" in report.generator_holdout_violations[0]
    assert not report.temporal_violations
    assert not report.ring_leakage_violations


def test_temporal_violation_detected() -> None:
    claims = _valid_claims()
    # Push a train claim past the calibration window's start.
    claims.loc[claims["claim_id"] == "c2", "claim_timestamp"] = datetime(2025, 11, 1)
    report = verify_splits(claims)
    assert not report.is_valid
    assert report.temporal_violations
    assert not report.generator_holdout_violations


def test_entity_leakage_violation_detected() -> None:
    claims = _valid_claims()
    # Same canonical phone in train and test = the same human across the
    # boundary; delegated to eval/entity_leakage_audit.py.
    claims.loc[claims["claim_id"] == "c4", "phone"] = "+91 90000 00001"
    report = verify_splits(claims)
    assert not report.is_valid
    assert len(report.entity_leakage_violations) == 1
    assert report.entity_leakage_violations[0].splits == frozenset({"train", "test"})


def test_ring_leakage_violation_detected() -> None:
    claims = _valid_claims()
    # The same source image reused across the train/test boundary.
    claims.loc[claims["claim_id"] == "c4", "image_group_id"] = "g1"
    report = verify_splits(claims)
    assert not report.is_valid
    assert len(report.ring_leakage_violations) == 1
    assert "g1" in report.ring_leakage_violations[0]


def test_describe_lists_every_violation_kind() -> None:
    claims = _valid_claims()
    claims.loc[claims["claim_id"] == "c2", "generator_family"] = "Midjourney"
    claims.loc[claims["claim_id"] == "c4", "image_group_id"] = "g1"
    claims.loc[claims["claim_id"] == "c4", "phone"] = "+91 90000 00001"
    text = verify_splits(claims).describe()
    assert "[generator-holdout]" in text
    assert "[ring-disjoint]" in text
    assert "[entity-disjoint]" in text


def test_missing_required_column_raises() -> None:
    claims = _valid_claims().drop(columns=["image_group_id"])
    with pytest.raises(ValueError, match="image_group_id"):
        verify_splits(claims)


def test_simulated_ledger_satisfies_all_four_constraints() -> None:
    """The real point of this module: simulate_ledger.py claims to be
    leakage-free by construction, and this verifies it rather than
    trusting the design note."""
    ledger = simulate_ledger(n_claims=2000, merchants=["m1", "m2", "m3"], seed=11)
    ledger = ledger.rename(columns={"cohort": "split"})
    report = verify_splits(ledger)
    assert report.is_valid, report.describe()


# --- reconciliation ---------------------------------------------------


def test_reconcile_is_a_noop_on_an_already_valid_corpus() -> None:
    claims = _valid_claims()
    reconciled, report = reconcile_splits(claims)
    assert report.n_dropped == 0
    assert report.n_components_reconciled == 0
    assert len(reconciled) == len(claims)


def test_reconcile_drops_the_later_half_of_a_cross_split_entity_component() -> None:
    claims = _valid_claims()
    # c1 (train, earliest) and c4 (test) become the same canonical identity.
    claims.loc[claims["claim_id"] == "c4", "phone"] = "+91 90000 00001"
    reconciled, report = reconcile_splits(claims)

    assert report.n_components_reconciled == 1
    assert report.dropped_claim_ids == ("c4",)  # earliest claim's split (train) wins
    assert set(reconciled["claim_id"]) == {"c1", "c2", "c3"}
    assert verify_splits(reconciled).is_valid


def test_reconcile_drops_the_later_half_of_a_cross_split_ring_component() -> None:
    claims = _valid_claims()
    claims.loc[claims["claim_id"] == "c4", "image_group_id"] = "g1"
    reconciled, report = reconcile_splits(claims)

    assert report.dropped_claim_ids == ("c4",)
    assert verify_splits(reconciled).is_valid


def test_reconcile_unions_identity_and_ring_relations() -> None:
    """A component can be connected through *both* relation types: c2
    shares an image with c1 (train), and c4 shares an identity with c2.
    Handling the two relations independently would miss that c4 must go."""
    claims = _valid_claims()
    claims.loc[claims["claim_id"] == "c2", "image_group_id"] = "g1"  # c1 <-> c2 via ring
    claims.loc[claims["claim_id"] == "c4", "phone"] = "+91 90000 00002"  # c2 <-> c4 via identity
    reconciled, report = reconcile_splits(claims)

    assert report.n_components_reconciled == 1
    assert "c4" in report.dropped_claim_ids
    assert verify_splits(reconciled).is_valid


def test_reconcile_makes_verification_pass_on_a_known_bad_seed() -> None:
    """seed=99 at n=3000 produces a genuinely ambiguous fuzzy-address
    collision (two different households, same PIN, same house number,
    different street) that no threshold tuning reliably prevents - which
    is exactly why reconciliation exists rather than only a better
    matcher."""
    ledger = simulate_ledger(n_claims=3000, merchants=["m1", "m2", "m3"], seed=99)
    ledger = ledger.rename(columns={"cohort": "split"})
    assert not verify_splits(ledger).is_valid

    reconciled, report = reconcile_splits(ledger)
    assert report.n_dropped > 0
    assert report.drop_rate < 0.01  # loss must stay negligible, not just non-zero
    assert verify_splits(reconciled).is_valid


def test_reconcile_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="image_group_id"):
        reconcile_splits(_valid_claims().drop(columns=["image_group_id"]))
