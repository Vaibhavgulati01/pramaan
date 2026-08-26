"""P4 claimant behaviour.

Two properties carry the weight: aggregates must be strictly
backward-looking (a claim never sees its own future), and they must be
keyed by *canonical identity* rather than raw claimant id - otherwise
someone opening a fresh account per claim looks like N first-time
claimants instead of one repeat abuser, which is the entire point of the
pillar.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from pramaan.ingest.identity import ClaimIdentitySignals
from pramaan.pillars.p4_behaviour import BehaviourAggregator

BASE = datetime(2026, 1, 1)


def _add(agg: BehaviourAggregator, claim_id: str, claimant: str, day: int, value: float = 1000.0,
         merchant: str = "m1", category: str = "electronics", **kw):
    ts = BASE + timedelta(days=day)
    return agg.query_then_add(
        claim_id=claim_id,
        claimant_id=claimant,
        merchant_id=merchant,
        category=category,
        timestamp=ts,
        order_date=ts - timedelta(days=3),
        order_value=value,
        **kw,
    )


def test_first_claim_has_no_history() -> None:
    agg = BehaviourAggregator()
    f = _add(agg, "c1", "alice", 0)
    assert f.is_first_claim
    assert f.n_prior_claims == 0
    assert f.days_since_first_claim == 0.0


def test_claim_never_counts_itself() -> None:
    agg = BehaviourAggregator()
    f = _add(agg, "c1", "alice", 0)
    assert f.n_prior_claims == 0


def test_second_claim_sees_exactly_one_prior() -> None:
    agg = BehaviourAggregator()
    _add(agg, "c1", "alice", 0)
    f = _add(agg, "c2", "alice", 5)
    assert not f.is_first_claim
    assert f.n_prior_claims == 1
    assert f.days_since_last_claim == pytest.approx(5.0)
    assert f.days_since_first_claim == pytest.approx(5.0)


def test_prior_counts_grow_only_with_earlier_claims() -> None:
    """Feeding N claims in time order must yield exactly 0,1,2,...N-1
    priors. Any leak shows up as a count too high for its position."""
    agg = BehaviourAggregator()
    for i in range(6):
        f = _add(agg, f"c{i}", "alice", i * 2)
        assert f.n_prior_claims == i


def test_different_claimants_have_separate_histories() -> None:
    agg = BehaviourAggregator()
    _add(agg, "c1", "alice", 0)
    _add(agg, "c2", "alice", 1)
    f = _add(agg, "c3", "bob", 2)
    assert f.is_first_claim
    assert f.n_prior_claims == 0


def test_rolling_windows_respect_their_horizons() -> None:
    agg = BehaviourAggregator()
    _add(agg, "old", "alice", 0)     # 200 days before the query
    _add(agg, "mid", "alice", 140)   # 60 days before
    _add(agg, "recent", "alice", 190)  # 10 days before
    f = _add(agg, "now", "alice", 200)
    assert f.n_prior_claims == 3
    assert f.n_prior_claims_30d == 1
    assert f.n_prior_claims_90d == 2


def test_order_value_statistics_use_only_priors() -> None:
    agg = BehaviourAggregator()
    _add(agg, "c1", "alice", 0, value=1000)
    _add(agg, "c2", "alice", 1, value=3000)
    f = _add(agg, "c3", "alice", 2, value=8000)
    assert f.mean_prior_order_value == pytest.approx(2000.0)
    assert f.max_prior_order_value == pytest.approx(3000.0)
    # The escalation ratio is the interesting signal: this claim is 4x
    # the claimant's historical average.
    assert f.order_value_vs_prior_mean == pytest.approx(4.0)


def test_order_value_ratio_defaults_to_one_without_history() -> None:
    """Neutral, not extreme - a first-time claimant must not look like an
    outlier purely for lacking history."""
    agg = BehaviourAggregator()
    f = _add(agg, "c1", "alice", 0, value=99999)
    assert f.order_value_vs_prior_mean == 1.0


def test_counts_distinct_merchants_and_categories() -> None:
    agg = BehaviourAggregator()
    _add(agg, "c1", "alice", 0, merchant="m1", category="electronics")
    _add(agg, "c2", "alice", 1, merchant="m2", category="apparel")
    f = _add(agg, "c3", "alice", 2, merchant="m1", category="home")
    assert f.n_distinct_merchants == 2
    assert f.n_distinct_categories == 2


def test_counts_distinct_devices() -> None:
    agg = BehaviourAggregator()
    _add(agg, "c1", "alice", 0, device_ua="UA/1", device_screen="100x200")
    _add(agg, "c2", "alice", 1, device_ua="UA/2", device_screen="100x200")
    f = _add(agg, "c3", "alice", 2, device_ua="UA/1", device_screen="100x200")
    assert f.n_distinct_devices == 2


def test_missing_device_fields_are_not_counted() -> None:
    agg = BehaviourAggregator()
    _add(agg, "c1", "alice", 0)
    f = _add(agg, "c2", "alice", 1)
    assert f.n_distinct_devices == 0


def test_claim_to_order_latency() -> None:
    agg = BehaviourAggregator()
    f = _add(agg, "c1", "alice", 10)
    assert f.claim_to_order_days == pytest.approx(3.0)


# --- canonical identity ----------------------------------------------


def test_history_follows_canonical_identity_not_raw_claimant_id() -> None:
    """The pillar's reason for existing: a claimant using a fresh account
    per claim is one repeat claimant, not three first-timers."""
    agg = BehaviourAggregator()
    agg.register_identities(
        [
            ClaimIdentitySignals("c1", phone_raw="9876543210"),
            ClaimIdentitySignals("c2", phone_raw="+91 98765 43210"),
            ClaimIdentitySignals("c3", phone_raw="09876543210"),
        ]
    )
    first = _add(agg, "c1", "account_1", 0)
    second = _add(agg, "c2", "account_2", 1)
    third = _add(agg, "c3", "account_3", 2)

    assert first.is_first_claim
    assert second.n_prior_claims == 1
    assert third.n_prior_claims == 2


def test_unregistered_claims_fall_back_to_raw_claimant_id() -> None:
    """Identity resolution is optional; without it the pillar still works,
    just with the weaker key."""
    agg = BehaviourAggregator()
    _add(agg, "c1", "alice", 0)
    f = _add(agg, "c2", "alice", 1)
    assert f.n_prior_claims == 1


def test_distinct_identities_stay_separate() -> None:
    agg = BehaviourAggregator()
    agg.register_identities(
        [
            ClaimIdentitySignals("c1", phone_raw="9876543210"),
            ClaimIdentitySignals("c2", phone_raw="9111111111"),
        ]
    )
    _add(agg, "c1", "account_1", 0)
    f = _add(agg, "c2", "account_2", 1)
    assert f.is_first_claim


def test_feature_dict_is_all_finite_floats() -> None:
    agg = BehaviourAggregator()
    _add(agg, "c1", "alice", 0)
    as_dict = _add(agg, "c2", "alice", 1).as_dict()
    assert len(as_dict) == 13
    assert all(isinstance(v, float) for v in as_dict.values())
