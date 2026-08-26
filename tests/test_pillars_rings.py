"""Ring detection: clusters of claims sharing an image.

The two properties worth testing hardest are the ones that are silent
when broken: temporal correctness (a ring must not reflect its own
future) and first-seen immunity (an attacker must not be able to burn a
rival's genuine claim by submitting their photo first).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pramaan.pillars.p3_reuse import ReuseFeatures, ReuseMatch
from pramaan.pillars.rings import RingDetector

BASE = datetime(2026, 1, 1)


def _match(claim_id: str, claimant_id: str, merchant_id: str, ts: datetime) -> ReuseMatch:
    return ReuseMatch(
        matched_claim_id=claim_id,
        matched_claimant_id=claimant_id,
        matched_merchant_id=merchant_id,
        matched_timestamp=ts,
        hamming=0,
        clip_similarity=None,
        stage="phash_lsh",
    )


def _reuse(*matches: ReuseMatch) -> ReuseFeatures:
    return ReuseFeatures(n_matches=len(matches), matched_prior_claim=bool(matches),
                         matches=list(matches))


def test_isolated_claim_is_not_in_a_ring() -> None:
    det = RingDetector()
    features = det.observe("c1", "claimant_1", "m1", BASE, _reuse())
    assert not features.in_ring
    assert det.rings() == []


def test_first_claimant_on_a_cluster_is_immune() -> None:
    """First-seen immunity: submitting an image first must never be
    penalised, or an attacker can burn a rival's genuine claim by
    submitting their photo before them."""
    det = RingDetector()
    first = det.observe("c1", "victim", "m1", BASE, _reuse())
    assert first.is_first_seen_in_cluster
    assert not first.in_ring


def test_second_distinct_claimant_forms_a_ring_and_is_flagged() -> None:
    det = RingDetector()
    det.observe("c1", "victim", "m1", BASE, _reuse())
    second = det.observe(
        "c2", "attacker", "m1", BASE + timedelta(days=1),
        _reuse(_match("c1", "victim", "m1", BASE)),
    )
    assert second.in_ring
    assert second.ring_distinct_claimants == 2
    assert second.ring_size == 2
    assert not second.is_first_seen_in_cluster


def test_same_claimant_reusing_own_image_is_not_a_ring() -> None:
    """One person resubmitting their own photo is ordinary behaviour.
    Treating it as a ring would flood the ring signal with noise and
    blur the distinction the feature exists to draw."""
    det = RingDetector()
    det.observe("c1", "same_person", "m1", BASE, _reuse())
    second = det.observe(
        "c2", "same_person", "m1", BASE + timedelta(days=1),
        _reuse(_match("c1", "same_person", "m1", BASE)),
    )
    assert not second.in_ring
    assert det.rings() == []


def test_ring_grows_across_multiple_claimants() -> None:
    det = RingDetector()
    det.observe("c1", "a", "m1", BASE, _reuse())
    det.observe("c2", "b", "m1", BASE + timedelta(days=1),
                _reuse(_match("c1", "a", "m1", BASE)))
    third = det.observe(
        "c3", "c", "m2", BASE + timedelta(days=2),
        _reuse(_match("c1", "a", "m1", BASE),
               _match("c2", "b", "m1", BASE + timedelta(days=1))),
    )
    assert third.ring_size == 3
    assert third.ring_distinct_claimants == 3
    assert third.ring_distinct_merchants == 2
    assert third.ring_span_days == 2.0


def test_ring_state_never_reflects_the_future() -> None:
    """Claim 2's features must describe the cluster as of claim 2 - not
    as it looks once claim 3 arrives. Ring features feed the model, so a
    hindsight-inflated ring size would leak the label."""
    det = RingDetector()
    det.observe("c1", "a", "m1", BASE, _reuse())
    second = det.observe("c2", "b", "m1", BASE + timedelta(days=1),
                         _reuse(_match("c1", "a", "m1", BASE)))
    assert second.ring_size == 2

    det.observe("c3", "c", "m1", BASE + timedelta(days=2),
                _reuse(_match("c1", "a", "m1", BASE)))
    # The already-returned features are a snapshot and must not mutate.
    assert second.ring_size == 2


def test_transitive_merge_of_two_clusters_preserves_history() -> None:
    """A claim matching two separate clusters joins them. The merged
    cluster must retain both halves' claims and claimants, and the
    earlier of the two first-seen times."""
    det = RingDetector()
    det.observe("a1", "a", "m1", BASE, _reuse())
    det.observe("a2", "b", "m1", BASE + timedelta(days=1),
                _reuse(_match("a1", "a", "m1", BASE)))
    det.observe("b1", "c", "m2", BASE + timedelta(days=2), _reuse())
    det.observe("b2", "d", "m2", BASE + timedelta(days=3),
                _reuse(_match("b1", "c", "m2", BASE + timedelta(days=2))))

    assert len(det.rings()) == 2

    bridge = det.observe(
        "x", "e", "m3", BASE + timedelta(days=4),
        _reuse(_match("a1", "a", "m1", BASE),
               _match("b1", "c", "m2", BASE + timedelta(days=2))),
    )
    assert len(det.rings()) == 1
    assert bridge.ring_size == 5
    assert bridge.ring_distinct_claimants == 5
    assert bridge.ring_distinct_merchants == 3
    # first_seen must survive the merge as the earliest of the two.
    cluster = det.cluster_for("x")
    assert cluster is not None
    assert cluster.first_seen == BASE
    assert cluster.first_claimant_id == "a"


def test_cluster_lookup_returns_the_same_cluster_for_every_member() -> None:
    det = RingDetector()
    det.observe("c1", "a", "m1", BASE, _reuse())
    det.observe("c2", "b", "m1", BASE + timedelta(days=1),
                _reuse(_match("c1", "a", "m1", BASE)))
    assert det.cluster_for("c1") is det.cluster_for("c2")


def test_min_claimants_threshold_is_configurable() -> None:
    det = RingDetector(min_claimants_for_ring=3)
    det.observe("c1", "a", "m1", BASE, _reuse())
    second = det.observe("c2", "b", "m1", BASE + timedelta(days=1),
                         _reuse(_match("c1", "a", "m1", BASE)))
    assert not second.in_ring  # 2 claimants, bar is 3

    third = det.observe("c3", "c", "m1", BASE + timedelta(days=2),
                        _reuse(_match("c1", "a", "m1", BASE)))
    assert third.in_ring


def test_features_serialise_to_numeric_dict() -> None:
    det = RingDetector()
    det.observe("c1", "a", "m1", BASE, _reuse())
    features = det.observe("c2", "b", "m1", BASE + timedelta(days=1),
                           _reuse(_match("c1", "a", "m1", BASE)))
    as_dict = features.as_dict()
    assert as_dict["ring_in_ring"] == 1.0
    assert as_dict["ring_distinct_claimants"] == 2.0
    assert all(isinstance(v, float) for v in as_dict.values())
