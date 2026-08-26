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
