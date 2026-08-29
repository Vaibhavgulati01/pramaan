"""Stage 2 of P3: CLIP + FAISS semantic near-duplicate matching.

Uses synthetic unit vectors rather than real CLIP embeddings - the point
here is the index's behaviour (temporal safety, thresholding,
deduplication against stage 1), not whether CLIP embeds images well.
Real embeddings are exercised end-to-end in the corpus run.

The temporal-constraint tests from tests/test_pillars_p3_temporal.py are
repeated here against the CLIP path specifically, because a leak in
stage 2 is just as fatal as one in stage 1 and the two paths reach the
index by different code.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from pramaan.pillars.p3_reuse import TemporalLeakError, TemporalReuseIndex

BASE = datetime(2026, 1, 1)
DIM = 8
PHASH_A = 0xABCD_1234_5678_9F01
PHASH_UNRELATED = 0x0FED_CBA9_8765_4321


def _vec(*values: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i, value in enumerate(values):
        v[i] = value
    return v


def _index(clip_threshold: float = 0.92) -> TemporalReuseIndex:
    return TemporalReuseIndex(clip_threshold=clip_threshold, clip_dim=DIM)


def test_clip_disabled_by_default() -> None:
    assert not TemporalReuseIndex().clip_enabled
    assert _index().clip_enabled


def test_semantically_identical_images_match_despite_unrelated_phash() -> None:
    """The whole reason stage 2 exists: crop/rotate/recolour reuse has a
    median pHash distance of 15, far outside any usable threshold, but
    remains semantically near-identical."""
    idx = _index()
    v = _vec(1.0)
    idx.query_then_add("c1", "claimant_1", "m1", BASE, PHASH_A, clip_embedding=v)
    features = idx.query_then_add(
        "c2",
        "claimant_2",
        "m1",
        BASE + timedelta(days=1),
        PHASH_UNRELATED,  # pHash sees nothing
        clip_embedding=v,
    )
    assert features.matched_prior_claim
    assert features.matches[0].stage == "clip"
    assert features.max_clip_similarity == pytest.approx(1.0, abs=1e-5)


def test_dissimilar_embeddings_do_not_match() -> None:
    idx = _index()
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0))
    features = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_UNRELATED,
        clip_embedding=_vec(0.0, 1.0),  # orthogonal -> similarity 0
    )
    assert features.n_matches == 0


def test_similarity_just_below_threshold_is_rejected() -> None:
    idx = TemporalReuseIndex(clip_threshold=0.95, clip_dim=DIM)
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0))
    # cos ~= 0.9487, deliberately just under 0.95
    features = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_UNRELATED,
        clip_embedding=_vec(3.0, 1.0),
    )
    assert features.n_matches == 0


def test_embeddings_are_normalised_so_magnitude_is_irrelevant() -> None:
    idx = _index()
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0))
    features = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_UNRELATED,
        clip_embedding=_vec(500.0),  # same direction, wildly different norm
    )
    assert features.max_clip_similarity == pytest.approx(1.0, abs=1e-5)


# --- temporal safety, again, via the CLIP path ------------------------


def test_clip_earlier_claim_cannot_match_a_later_one() -> None:
    idx = _index()
    v = _vec(1.0)
    first = idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=v)
    second = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=5), PHASH_UNRELATED, clip_embedding=v
    )
    assert first.n_matches == 0, "earlier claim saw a future claim via CLIP - temporal leak"
    assert second.n_matches == 1


def test_clip_claim_never_matches_itself() -> None:
    idx = _index()
    features = idx.query_then_add(
        "c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0)
    )
    assert features.n_matches == 0


def test_clip_simultaneous_claims_do_not_match() -> None:
    idx = _index()
    v = _vec(1.0)
    a = idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=v)
    b = idx.query_then_add("c2", "c_2", "m1", BASE, PHASH_UNRELATED, clip_embedding=v)
    assert a.n_matches == 0
    assert b.n_matches == 0


def test_clip_match_count_grows_only_with_prior_claims() -> None:
    idx = _index()
    v = _vec(1.0)
    for i in range(5):
        features = idx.query_then_add(
            f"c{i}", f"c_{i}", "m1", BASE + timedelta(days=i), PHASH_UNRELATED,
            clip_embedding=v,
        )
        assert features.n_matches == i


def test_clip_out_of_order_still_raises() -> None:
    idx = _index()
    v = _vec(1.0)
    idx.query_then_add("c2", "c_2", "m1", BASE + timedelta(days=5), PHASH_A, clip_embedding=v)
    with pytest.raises(TemporalLeakError):
        idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=v)


# --- interaction with stage 1 -----------------------------------------


def test_phash_and_clip_matches_are_not_double_counted() -> None:
    """A claim matched by both stages must appear once, or every
    downstream count (distinct claimants, ring size) is inflated."""
    idx = _index()
    v = _vec(1.0)
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=v)
    features = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_A, clip_embedding=v
    )
    assert features.n_matches == 1
    assert features.matches[0].stage == "phash_lsh"  # cheaper stage wins
    assert len({m.matched_claim_id for m in features.matches}) == 1


def test_index_works_without_embeddings_even_when_clip_enabled() -> None:
    """Stage 2 is opt-in per claim: omitting an embedding must degrade to
    pHash-only rather than raise."""
    idx = _index()
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A)
    features = idx.query_then_add("c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_A)
    assert features.n_matches == 1
    assert features.matches[0].stage == "phash_lsh"


def test_wrong_embedding_dimension_raises() -> None:
    idx = _index()
    with pytest.raises(ValueError, match="dimension"):
        idx.query_then_add(
            "c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=np.ones(DIM + 3, dtype=np.float32)
        )


def test_zero_embedding_raises_rather_than_producing_nan() -> None:
    idx = _index()
    with pytest.raises(ValueError, match="zero embedding"):
        idx.query_then_add(
            "c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=np.zeros(DIM, dtype=np.float32)
        )


# --- vector backends --------------------------------------------------


def test_numpy_backend_matches_faiss_exactly() -> None:
    """The default backend is NumPy rather than FAISS (see the backend
    note in p3_reuse.py: faiss-cpu and torch collide over OpenMP on
    Windows). That swap is only safe if the two agree, so this asserts
    they return the same matches on the same input.
    """
    faiss = pytest.importorskip("faiss")
    assert faiss is not None

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((40, DIM)).astype(np.float32)

    def run(backend: str) -> list[tuple[str, float]]:
        idx = TemporalReuseIndex(
            clip_threshold=0.5, clip_dim=DIM, vector_backend=backend
        )
        found: list[tuple[str, float]] = []
        for i, vec in enumerate(vectors):
            feats = idx.query_then_add(
                f"c{i}", f"c_{i}", "m1", BASE + timedelta(hours=i),
                PHASH_UNRELATED, clip_embedding=vec,
            )
            for m in feats.matches:
                found.append((f"c{i}->{m.matched_claim_id}", round(m.clip_similarity or 0.0, 5)))
        return found

    numpy_matches = run("numpy")
    faiss_matches = run("faiss")
    assert numpy_matches, "test is vacuous if neither backend found anything"
    assert numpy_matches == faiss_matches


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown vector backend"):
        TemporalReuseIndex(clip_dim=DIM, vector_backend="annoy")


def test_numpy_store_returns_descending_similarities() -> None:
    from pramaan.pillars.p3_reuse import NumpyFlatIP

    store = NumpyFlatIP(DIM)
    for v in (_vec(1.0), _vec(0.0, 1.0), _vec(0.7, 0.7)):
        store.add(v / np.linalg.norm(v))
    sims, idxs = store.search(_vec(1.0), k=3)
    assert store.ntotal == 3
    assert list(sims[0]) == sorted(sims[0], reverse=True)
    assert idxs[0][0] == 0  # the exact match ranks first


def test_numpy_store_handles_k_larger_than_contents() -> None:
    from pramaan.pillars.p3_reuse import NumpyFlatIP

    store = NumpyFlatIP(DIM)
    store.add(_vec(1.0))
    sims, idxs = store.search(_vec(1.0), k=50)
    assert sims.shape == (1, 1)
    assert idxs.shape == (1, 1)


def test_numpy_store_empty_search_returns_nothing() -> None:
    from pramaan.pillars.p3_reuse import NumpyFlatIP

    sims, idxs = NumpyFlatIP(DIM).search(_vec(1.0), k=5)
    assert sims.size == 0
    assert idxs.size == 0


# --- graded evidence --------------------------------------------------


def test_near_miss_is_reported_even_though_it_does_not_match() -> None:
    """The whole point of the graded features: crop/rotate/recolour reuse
    lands around CLIP 0.93, below any threshold that keeps false
    positives tolerable, but is obviously different from an unrelated
    pair. Reporting only a boolean would discard that."""
    idx = TemporalReuseIndex(clip_threshold=0.98, clip_dim=DIM)
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0))
    features = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_UNRELATED,
        clip_embedding=_vec(4.0, 1.0),  # cos ~= 0.970, under the 0.98 bar
    )
    assert not features.matched_prior_claim
    # Rejected as a match, but the near-miss is still reported.
    assert features.best_clip_similarity == pytest.approx(0.970, abs=1e-3)
    assert features.n_candidates_examined == 1


def test_graded_features_populated_below_threshold() -> None:
    idx = TemporalReuseIndex(clip_threshold=0.999, clip_dim=DIM)
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0))
    features = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_UNRELATED,
        clip_embedding=_vec(3.0, 1.0),  # cos ~= 0.949, well under 0.999
    )
    assert not features.matched_prior_claim
    assert features.best_clip_similarity == pytest.approx(0.9486, abs=1e-3)
    assert features.n_candidates_examined == 1


def test_best_hamming_reports_the_closest_candidate_not_only_matches() -> None:
    near_miss = PHASH_A ^ 0b1111  # 4 bits, outside a threshold of 2
    idx = TemporalReuseIndex(hamming_threshold=2)
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A)
    features = idx.query_then_add("c2", "c_2", "m1", BASE + timedelta(days=1), near_miss)
    assert not features.matched_prior_claim
    assert features.best_hamming == 4
    assert features.n_candidates_examined == 1


def test_graded_defaults_when_nothing_was_examined() -> None:
    idx = _index()
    features = idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0))
    assert features.n_candidates_examined == 0
    assert features.best_clip_similarity == 0.0
    assert features.best_hamming == 64


def test_graded_features_appear_in_the_feature_dict() -> None:
    idx = _index()
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0))
    as_dict = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_A, clip_embedding=_vec(1.0)
    ).as_dict()
    for key in ("reuse_best_hamming", "reuse_best_clip_similarity"):
        assert key in as_dict
        assert isinstance(as_dict[key], float)


def test_candidate_count_is_tracked_but_not_a_model_feature() -> None:
    """`n_candidates_examined` counts prior claims in the index, so it
    grows with corpus position and proxied split membership (691 -> 1597
    -> 2041 across train/calibration/test), drawing 10.2% of model gain.
    It stays available for audit records and diagnostics, but must not
    reach the feature vector."""
    idx = _index()
    idx.query_then_add("c1", "c_1", "m1", BASE, PHASH_A, clip_embedding=_vec(1.0))
    features = idx.query_then_add(
        "c2", "c_2", "m1", BASE + timedelta(days=1), PHASH_A, clip_embedding=_vec(1.0)
    )
    assert features.n_candidates_examined == 1  # still measured
    assert "reuse_n_candidates_examined" not in features.as_dict()  # not fed to the model


def test_search_does_not_rebuild_the_index_buffer() -> None:
    """The reuse index must not be quadratic in the query_then_add pattern.

    The original backend kept rows in a list and `vstack`ed them on
    demand, invalidating the cache on every `add`. Because
    `query_then_add` interleaves add and search, every query re-copied
    the whole index. Measured growth was O(N^2.28): ~1 s at 1,000 claims
    but an extrapolated ~47 minutes at the `full` tier's 35,000, paid
    again for every ablation that rebuilds the index.

    Asserting wall-clock here would be flaky on shared CI, so this pins
    the structural property instead: searching never reallocates, and
    adding reallocates only on a doubling boundary.
    """
    from pramaan.pillars.p3_reuse import NumpyFlatIP

    rng = np.random.default_rng(11)
    store = NumpyFlatIP(8)
    rows = rng.standard_normal((300, 8)).astype(np.float32)

    reallocations = 0
    previous = store._matrix
    for i, row in enumerate(rows):
        if i:
            store.search(row, k=3)
            assert store._matrix is previous, (
                "search reallocated the index buffer; that is the O(N^2) bug"
            )
        store.add(row)
        if store._matrix is not previous:
            reallocations += 1
            previous = store._matrix

    assert store.ntotal == len(rows)
    # 300 rows from a 1024-row initial buffer: no growth needed at all.
    assert reallocations == 0, f"unexpected reallocation(s): {reallocations}"


def test_index_buffer_grows_by_doubling() -> None:
    """Growth must be geometric, or `add` is O(N) and we are back to O(N^2)."""
    from pramaan.pillars.p3_reuse import NumpyFlatIP

    store = NumpyFlatIP(4)
    store._matrix = np.empty((2, 4), dtype=np.float32)  # shrink to exercise growth
    store._count = 0

    capacities = [store._matrix.shape[0]]
    rng = np.random.default_rng(3)
    for row in rng.standard_normal((64, 4)).astype(np.float32):
        store.add(row)
        if store._matrix.shape[0] != capacities[-1]:
            capacities.append(store._matrix.shape[0])

    # 64 rows into a capacity-2 buffer: the last add fits, so growth stops at 64.
    assert capacities == [2, 4, 8, 16, 32, 64], capacities
    assert store.ntotal == 64


def test_growth_preserves_every_row_exactly() -> None:
    """Reallocation must copy, not truncate or reorder."""
    from pramaan.pillars.p3_reuse import NumpyFlatIP

    rng = np.random.default_rng(5)
    rows = rng.standard_normal((200, 6)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)

    store = NumpyFlatIP(6)
    store._matrix = np.empty((2, 6), dtype=np.float32)
    store._count = 0
    for row in rows:
        store.add(row)

    # Every original row must still be retrievable as its own nearest
    # neighbour, at index i, after many reallocations.
    for i, row in enumerate(rows):
        sims, idx = store.search(row, k=1)
        assert idx[0][0] == i, f"row {i} lost or reordered by buffer growth"
        assert sims[0][0] == pytest.approx(1.0, abs=1e-5)
