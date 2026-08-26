import pandas as pd
import pytest

from eval.entity_leakage_audit import find_leakage_violations


def test_clean_split_has_no_violations() -> None:
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2", "c3", "c4"],
            "split": ["train", "train", "test", "test"],
            "phone": ["9876543210", "9876543210", "9111111111", "9111111111"],
        }
    )
    assert find_leakage_violations(claims) == []


def test_same_phone_across_train_and_test_is_flagged() -> None:
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2"],
            "split": ["train", "test"],
            "phone": ["9876543210", "+91 98765 43210"],
        }
    )
    violations = find_leakage_violations(claims)
    assert len(violations) == 1
    assert violations[0].splits == frozenset({"train", "test"})
    assert violations[0].claim_ids == ("c1", "c2")


def test_same_email_across_calibration_and_test_is_flagged() -> None:
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2"],
            "split": ["calibration", "test"],
            "email": ["a.b@gmail.com", "ab+dup@gmail.com"],
        }
    )
    violations = find_leakage_violations(claims)
    assert len(violations) == 1


def test_same_pin_and_similar_address_across_splits_is_flagged() -> None:
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2"],
            "split": ["train", "test"],
            "address": ["Flat 12 Opp City Mall", "flat no 12 opposite city mall"],
            "pin": ["400001", "400001"],
        }
    )
    violations = find_leakage_violations(claims)
    assert len(violations) == 1


def test_nan_signal_cells_do_not_crash_or_falsely_match() -> None:
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2", "c3"],
            "split": ["train", "test", "train"],
            "phone": ["9876543210", None, float("nan")],
            "email": [None, "a@b.com", float("nan")],
        }
    )
    # c3 has no usable signals at all and must not spuriously cluster
    # with c1 or c2 just because both their NaN cells stringify the same.
    assert find_leakage_violations(claims) == []


def test_cluster_confined_to_one_split_is_not_a_violation() -> None:
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2", "c3"],
            "split": ["train", "train", "test"],
            "phone": ["9876543210", "9876543210", "9111111111"],
        }
    )
    assert find_leakage_violations(claims) == []


def test_missing_required_column_raises() -> None:
    claims = pd.DataFrame({"claim_id": ["c1"]})
    with pytest.raises(ValueError, match="split"):
        find_leakage_violations(claims)
