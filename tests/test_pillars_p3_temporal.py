"""The temporal-leak test the spec calls out as mandatory (Sec.4 L1):

    "a claim may only match evidence submitted *before* it. Any
    implementation that queries the full index including future claims
    has a temporal leak that will inflate every number in the repo.
    Make it a unit test."

This is that unit test. It is deliberately paranoid: the failure mode is
silent - a leaked index produces *better* numbers, not an error - so
these tests assert the constraint from several directions rather than
trusting one happy-path check.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from pramaan.pillars.p3_reuse import (
    TemporalLeakError,
    TemporalReuseIndex,
    hamming_distance,
    lsh_bands,
)

BASE = datetime(2026, 1, 1)
PHASH_A = 0xABCD_1234_5678_9F01
PHASH_B = 0x0FED_CBA9_8765_4321


def _index() -> TemporalReuseIndex:
    return TemporalReuseIndex(hamming_threshold=10)


def test_first_claim_never_matches_anything() -> None:
    idx = _index()
    features = idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    assert features.n_matches == 0
    assert not features.matched_prior_claim


def test_later_claim_matches_identical_earlier_image() -> None:
    idx = _index()
    idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    later = idx.query_then_add(
        "c2", "claimant_2", "m1", BASE + timedelta(days=1), PHASH_A
    )
    assert later.matched_prior_claim
    assert later.n_matches == 1
    assert later.min_hamming == 0
    assert later.matches[0].matched_claim_id == "c1"


def test_a_claim_never_matches_itself() -> None:
    """The query must run against the index as it stood *before* this
    claim was added - otherwise every claim trivially matches itself."""
    idx = _index()
    features = idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    assert features.n_matches == 0
    assert "c1" not in {m.matched_claim_id for m in features.matches}


def test_earlier_claim_cannot_match_a_later_one() -> None:
    """THE test. Two claims share an image; only the later one may see
    the earlier. If the index answered against the full corpus, the first
    claim would report a match against a claim that had not happened yet.
    """
    idx = _index()
    first = idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    second = idx.query_then_add(
        "c2", "claimant_2", "m1", BASE + timedelta(days=5), PHASH_A
    )

    assert first.n_matches == 0, "earlier claim saw a future claim - temporal leak"
    assert second.n_matches == 1


def test_simultaneous_claims_do_not_match_each_other() -> None:
    """Equal timestamps are 'not earlier'. Admitting ties would make the
    result depend on insertion order, which is not a property of the
    data."""
    idx = _index()
    a = idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    b = idx.query_then_add("c2", "claimant_2", "m1", BASE, PHASH_A)
    assert a.n_matches == 0
    assert b.n_matches == 0


def test_out_of_order_claims_raise_rather_than_leak() -> None:
    """The index cannot enforce its guarantee if fed unsorted claims, so
    it must fail loudly rather than silently return future matches."""
    idx = _index()
    idx.query_then_add("c2", "claimant_2", "m1", BASE + timedelta(days=5), PHASH_A)
    with pytest.raises(TemporalLeakError, match="precedes the last indexed claim"):
        idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)


def test_duplicate_claim_id_is_rejected() -> None:
    idx = _index()
    idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    with pytest.raises(ValueError, match="already indexed"):
        idx.query_then_add("c1", "claimant_1", "m1", BASE + timedelta(days=1), PHASH_A)


def test_match_count_grows_only_with_prior_claims() -> None:
    """Feeding N copies of one image in time order must yield exactly
    0, 1, 2, ... N-1 matches. Any leak shows up immediately as a count
    that is too high for its position in the sequence."""
    idx = _index()
    for i in range(6):
        features = idx.query_then_add(
            f"c{i}", f"claimant_{i}", "m1", BASE + timedelta(days=i), PHASH_A
        )
        assert features.n_matches == i, f"claim {i} saw {features.n_matches} priors"


def test_unrelated_image_does_not_match() -> None:
    idx = _index()
    idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    features = idx.query_then_add(
        "c2", "claimant_2", "m1", BASE + timedelta(days=1), PHASH_B
    )
    assert hamming_distance(PHASH_A, PHASH_B) > 10
    assert features.n_matches == 0


def test_near_duplicate_within_threshold_matches() -> None:
    near = PHASH_A ^ 0b111  # 3 bits flipped, inside the threshold of 10
    idx = _index()
    idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    features = idx.query_then_add("c2", "claimant_2", "m1", BASE + timedelta(days=1), near)
    assert features.n_matches == 1
    assert features.min_hamming == 3


def test_distinct_claimants_excludes_the_querying_claimant() -> None:
    """One person resubmitting their own photo is not a ring. Counting
    them as a 'distinct claimant sharing' would conflate the two."""
    idx = _index()
    idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    same_person = idx.query_then_add(
        "c2", "claimant_1", "m1", BASE + timedelta(days=1), PHASH_A
    )
    assert same_person.n_matches == 1
    assert same_person.n_distinct_claimants_sharing == 0

    other_person = idx.query_then_add(
        "c3", "claimant_2", "m1", BASE + timedelta(days=2), PHASH_A
    )
    assert other_person.n_distinct_claimants_sharing == 1


def test_days_since_first_seen_uses_the_earliest_match() -> None:
    idx = _index()
    idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    idx.query_then_add("c2", "claimant_2", "m1", BASE + timedelta(days=3), PHASH_A)
    features = idx.query_then_add(
        "c3", "claimant_3", "m1", BASE + timedelta(days=10), PHASH_A
    )
    assert features.days_since_first_seen == pytest.approx(10.0)


def test_distinct_merchants_counted_across_the_ring() -> None:
    idx = _index()
    idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A)
    idx.query_then_add("c2", "claimant_2", "m2", BASE + timedelta(days=1), PHASH_A)
    features = idx.query_then_add(
        "c3", "claimant_3", "m3", BASE + timedelta(days=2), PHASH_A
    )
    assert features.n_distinct_merchants_sharing == 2


# --- LSH banding ------------------------------------------------------


def test_lsh_bands_partition_the_hash() -> None:
    bands = lsh_bands(PHASH_A, n_bands=16, band_bits=4)
    assert len(bands) == 16
    assert all(0 <= b < 16 for b in bands)

    rebuilt = 0
    for i, band in enumerate(bands):
        rebuilt |= band << (i * 4)
    assert rebuilt == PHASH_A


def test_identical_hashes_share_every_band() -> None:
    assert lsh_bands(PHASH_A) == lsh_bands(PHASH_A)


def test_near_duplicates_still_share_at_least_one_band() -> None:
    """LSH is only useful if small edits preserve some band - that is
    what lets candidate generation stay a dict lookup."""
    near = PHASH_A ^ 0b1  # single bit flip
    shared = set(enumerate(lsh_bands(PHASH_A))) & set(enumerate(lsh_bands(near)))
    assert shared
