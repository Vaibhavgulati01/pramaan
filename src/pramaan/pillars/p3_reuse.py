"""P3 - the reuse graph. The moat (PRAMAAN_v2_architecture.md Sec.4 L1).

Detects a claim's image being a duplicate or near-duplicate of evidence
submitted on an *earlier* claim: the same damage photo recycled under a
new claimant, a merchant's own catalog image passed off as damage, or one
image circulating across a ring of colluding claimants.

Two-stage candidate generation, cheap first:

1. **pHash + LSH banding** (16 bands x 4 bits over a 64-bit perceptual
   hash). Cheap, exact-ish, catches crops/rotations/recolours. A pair is
   a candidate if any band matches - the standard LSH trick that turns
   "find hashes within k bits" into a dictionary lookup.
2. **CLIP + FAISS** for semantic near-duplicates that survive edits pHash
   cannot follow. Only consulted for claims stage 1 did not resolve.

**The temporal constraint is the whole ballgame.** A claim may only
match evidence submitted strictly before it. An index that answers
queries against the full corpus - including claims from the future -
leaks the label and inflates every number downstream of it. The spec
calls this out as a mandatory unit test, and
`tests/test_pillars_p3_temporal.py` is it.

The constraint is enforced structurally rather than by filtering results
after the fact: `TemporalReuseIndex.query_then_add` is the only public
way to use the index, and it queries what has been added so far *before*
adding the current claim. There is no method that queries a
future-inclusive view, so a caller cannot accidentally ask for one.
"""

from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import imagehash
import numpy as np
from PIL import Image

# NOTE ON THE VECTOR BACKEND (and why FAISS is not imported here).
#
# Stage 2 needs exact inner-product search over at most a few tens of
# thousands of unit vectors. FAISS's `IndexFlatIP` is a brute-force
# matmul - algorithmically identical to `queries @ corpus.T` - so NumPy
# gives the same answer (verified: identical top-k, max similarity
# difference 2.4e-07, i.e. float32 epsilon) at the same speed (~3ms for
# 5 queries against 3,000x512).
#
# Using NumPy instead buys two things that matter here:
#
# 1. It removes a real Windows failure mode. `faiss-cpu` and `torch` each
#    bundle their own OpenMP runtime, and importing both aborts with
#    "OMP: Error #15 ... multiple copies of the OpenMP runtime". The
#    documented escape hatch (KMP_DUPLICATE_LIB_OK=TRUE) is described by
#    its own authors as unsupported and liable to "silently produce
#    incorrect results" - not something to build a certified decision on.
# 2. It is exactly deterministic, with no thread-count sensitivity, which
#    Phase 6's byte-identical-metrics test depends on.
#
# FAISS earns its place at `full` scale on the Linux VM, where the OpenMP
# conflict does not arise and `IndexHNSWFlat`'s sublinear search actually
# matters (configs/model.yaml: reuse_index.full). That backend is opt-in
# via `vector_backend="faiss"` and imports FAISS lazily, so the default
# path never touches it.

DEFAULT_PHASH_BITS = 64
DEFAULT_LSH_BANDS = 16
DEFAULT_BAND_BITS = 4

# Hamming threshold for a pHash match, chosen from measured separation on
# the dev corpus rather than picked by feel:
#
#   thr   TPR(true reuse)   FP per unrelated pair
#     6            43.4%              <0.005%
#    10            53.5%               0.030%
#    16            75.0%               0.390%
#
# The per-pair FP rate is misleadingly small: every claim is compared
# against all prior claims, so a 0.03% per-pair rate compounds to a ~19%
# chance of at least one false match per claim at dev scale. Measured
# directly, threshold 10 flagged 19.1% of the legit class. Threshold 6
# produced no false positives in 20,000 unrelated pairs.
#
# The recall this costs is deliberately not recovered by loosening the
# threshold - crop/rotate/recolour reuse has a median distance of 15 and
# is simply not reachable by pHash at any precision worth having. That is
# what the CLIP stage is for (Sec.4 L1's two-stage design).
DEFAULT_HAMMING_THRESHOLD = 6

# Cosine similarity for a CLIP near-duplicate. CLIP embeddings of
# unrelated product photos routinely reach 0.8+, so this is set high.
DEFAULT_CLIP_THRESHOLD = 0.92


def compute_phash(raw_bytes: bytes, hash_size: int = 8) -> int:
    """64-bit perceptual hash as a plain int (hash_size=8 -> 8x8 bits)."""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        bits = imagehash.phash(img.convert("RGB"), hash_size=hash_size).hash.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def lsh_bands(
    phash: int, n_bands: int = DEFAULT_LSH_BANDS, band_bits: int = DEFAULT_BAND_BITS
) -> tuple[int, ...]:
    """Splits a pHash into `n_bands` chunks of `band_bits`. Two images
    sharing any band are LSH candidates; the probability of that rises
    sharply as Hamming distance falls, which is the point."""
    mask = (1 << band_bits) - 1
    return tuple((phash >> (i * band_bits)) & mask for i in range(n_bands))


@dataclass(frozen=True)
class ReuseMatch:
    """One earlier claim whose image matches the querying claim."""

    matched_claim_id: str
    matched_claimant_id: str
    matched_merchant_id: str
    matched_timestamp: datetime
    hamming: int
    clip_similarity: float | None
    stage: str  # "phash_lsh" or "clip"


@dataclass
class ReuseFeatures:
    """P3's contribution to the fused feature vector (Sec.4 L2).

    Deliberately counts *distinct claimants and merchants*, not raw
    matches: one person legitimately re-submitting their own photo across
    two claims is a very different signal from one image appearing under
    several unrelated identities, and collapsing the two would be the
    kind of nonsense inversion the monotone constraints exist to prevent.
    """

    n_matches: int = 0
    n_distinct_claimants_sharing: int = 0
    n_distinct_merchants_sharing: int = 0
    min_hamming: int = DEFAULT_PHASH_BITS
    max_clip_similarity: float = 0.0
    days_since_first_seen: float = 0.0
    matched_prior_claim: bool = False
    matches: list[ReuseMatch] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        """Numeric view for the feature schema (Phase 3)."""
        return {
            "reuse_n_matches": float(self.n_matches),
            "reuse_n_distinct_claimants_sharing": float(self.n_distinct_claimants_sharing),
            "reuse_n_distinct_merchants_sharing": float(self.n_distinct_merchants_sharing),
            "reuse_min_hamming": float(self.min_hamming),
            "reuse_max_clip_similarity": float(self.max_clip_similarity),
            "reuse_days_since_first_seen": float(self.days_since_first_seen),
            "reuse_matched_prior_claim": float(self.matched_prior_claim),
        }


@dataclass(frozen=True)
class _IndexedClaim:
    claim_id: str
    claimant_id: str
    merchant_id: str
    timestamp: datetime
    phash: int


class TemporalLeakError(RuntimeError):
    """Raised when claims are presented out of chronological order.

    The index cannot enforce "only match earlier evidence" if the caller
    feeds it a claim older than one already indexed - it would silently
    return matches that had not happened yet. Failing loudly is the only
    safe response; `benchmarks.loaders` yields claims in timestamp order
    precisely so this never fires in normal use.
    """


class VectorStore(Protocol):
    """Exact inner-product search over unit vectors. Both backends are
    brute force; see the backend note at the top of this module."""

    @property
    def ntotal(self) -> int: ...

    def add(self, row: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class NumpyFlatIP:
    """Default backend: exact, deterministic, no native dependency.

    Vectors are appended to a list and stacked lazily, so adding is O(1)
    and the stack is rebuilt only when a search follows new additions.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    @property
    def ntotal(self) -> int:
        return len(self._rows)

    def add(self, row: np.ndarray) -> None:
        self._rows.append(np.asarray(row, dtype=np.float32).reshape(-1))
        self._matrix = None  # invalidate

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._rows:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
        if self._matrix is None:
            self._matrix = np.vstack(self._rows)

        sims = (np.asarray(query, dtype=np.float32).reshape(1, -1) @ self._matrix.T)[0]
        k = min(k, sims.shape[0])
        # argpartition then sort the top-k: O(n) rather than O(n log n),
        # with a stable tiebreak on index so results are reproducible.
        top = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(sims.shape[0])
        order = np.lexsort((top, -sims[top]))
        top = top[order]
        return sims[top].reshape(1, -1), top.reshape(1, -1).astype(np.int64)


class FaissFlatIP:
    """Opt-in backend for `full` scale on Linux, where FAISS's OpenMP
    runtime does not collide with torch's. Imports FAISS lazily so the
    default path never loads it."""

    def __init__(self, dim: int) -> None:
        import faiss

        # Single-threaded: FAISS with OpenMP is not bitwise reproducible
        # across thread counts, which would make Phase 6's determinism
        # assertion flaky for reasons unrelated to the pipeline.
        faiss.omp_set_num_threads(1)
        self._index = faiss.IndexFlatIP(dim)

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def add(self, row: np.ndarray) -> None:
        self._index.add(np.asarray(row, dtype=np.float32).reshape(1, -1))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), k)


def make_vector_store(dim: int, backend: str = "numpy") -> VectorStore:
    if backend == "numpy":
        return NumpyFlatIP(dim)
    if backend == "faiss":
        return FaissFlatIP(dim)
    raise ValueError(f"unknown vector backend {backend!r}; expected 'numpy' or 'faiss'")


class TemporalReuseIndex:
    """Append-only, strictly time-ordered reuse index.

    Usage is always `query_then_add`: query against everything indexed so
    far, then add the current claim. Claims must arrive in non-decreasing
    timestamp order (`benchmarks.loaders.Corpus.iter_claims` guarantees
    this); presenting an out-of-order claim raises TemporalLeakError
    rather than quietly producing a leaked match.
    """

    def __init__(
        self,
        hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
        n_bands: int = DEFAULT_LSH_BANDS,
        band_bits: int = DEFAULT_BAND_BITS,
        clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
        clip_dim: int | None = None,
        vector_backend: str = "numpy",
    ) -> None:
        self.hamming_threshold = hamming_threshold
        self.n_bands = n_bands
        self.band_bits = band_bits
        self.clip_threshold = clip_threshold
        self._claims: dict[str, _IndexedClaim] = {}
        self._band_buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        self._last_timestamp: datetime | None = None

        # Stage 2. Kept entirely optional: pass clip_dim to enable it.
        # Embeddings are L2-normalised on the way in, so inner product IS
        # cosine similarity.
        self._clip_dim = clip_dim
        self._clip_index = make_vector_store(clip_dim, vector_backend) if clip_dim else None
        self._clip_row_to_claim: list[str] = []

    def __len__(self) -> int:
        return len(self._claims)

    @property
    def last_timestamp(self) -> datetime | None:
        return self._last_timestamp

    @property
    def clip_enabled(self) -> bool:
        return self._clip_index is not None

    def _candidates(self, phash: int) -> set[str]:
        candidates: set[str] = set()
        for band_index, band_value in enumerate(lsh_bands(phash, self.n_bands, self.band_bits)):
            candidates.update(self._band_buckets.get((band_index, band_value), ()))
        return candidates

    def _clip_matches(
        self,
        embedding: np.ndarray,
        timestamp: datetime,
        already_matched: set[str],
        top_k: int = 20,
    ) -> list[ReuseMatch]:
        """Semantic near-duplicates among strictly-earlier claims.

        Temporal safety comes from the same discipline as the pHash path:
        nothing is added to the FAISS index until after its own query has
        run, so every vector in it belongs to an earlier or simultaneous
        claim - and simultaneous ones are filtered out explicitly below.
        """
        if self._clip_index is None or self._clip_index.ntotal == 0:
            return []

        query = _as_normalised_row(embedding, self._clip_dim)
        k = min(top_k, self._clip_index.ntotal)
        similarities, indices = self._clip_index.search(query, k)

        matches: list[ReuseMatch] = []
        for similarity, row in zip(similarities[0], indices[0], strict=True):
            if row < 0:
                continue
            candidate = self._claims[self._clip_row_to_claim[int(row)]]
            if candidate.timestamp >= timestamp:
                continue
            if candidate.claim_id in already_matched:
                continue  # pHash already found it; don't double-count
            if float(similarity) < self.clip_threshold:
                continue
            matches.append(
                ReuseMatch(
                    matched_claim_id=candidate.claim_id,
                    matched_claimant_id=candidate.claimant_id,
                    matched_merchant_id=candidate.merchant_id,
                    matched_timestamp=candidate.timestamp,
                    hamming=DEFAULT_PHASH_BITS,  # not a pHash match
                    clip_similarity=float(similarity),
                    stage="clip",
                )
            )
        return matches

    def query_then_add(
        self,
        claim_id: str,
        claimant_id: str,
        merchant_id: str,
        timestamp: datetime,
        phash: int,
        clip_embedding: np.ndarray | None = None,
    ) -> ReuseFeatures:
        """Matches `phash` (and optionally a CLIP embedding) against
        strictly-earlier claims, then indexes this claim.

        Returns features describing what it matched. The current claim is
        never in its own result: it is added only after the query runs.
        """
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise TemporalLeakError(
                f"claim {claim_id} has timestamp {timestamp} which precedes the last "
                f"indexed claim ({self._last_timestamp}). Feed claims in "
                "non-decreasing timestamp order - see benchmarks.loaders.Corpus.iter_claims."
            )
        if claim_id in self._claims:
            raise ValueError(f"claim {claim_id} is already indexed")

        matches: list[ReuseMatch] = []
        for candidate_id in self._candidates(phash):
            candidate = self._claims[candidate_id]
            # Equal timestamps are treated as "not earlier": simultaneous
            # claims cannot have seen each other, and admitting ties would
            # make the result depend on insertion order.
            if candidate.timestamp >= timestamp:
                continue
            distance = hamming_distance(phash, candidate.phash)
            if distance <= self.hamming_threshold:
                matches.append(
                    ReuseMatch(
                        matched_claim_id=candidate.claim_id,
                        matched_claimant_id=candidate.claimant_id,
                        matched_merchant_id=candidate.merchant_id,
                        matched_timestamp=candidate.timestamp,
                        hamming=distance,
                        clip_similarity=None,
                        stage="phash_lsh",
                    )
                )

        if clip_embedding is not None and self._clip_index is not None:
            matches.extend(
                self._clip_matches(
                    clip_embedding,
                    timestamp,
                    already_matched={m.matched_claim_id for m in matches},
                )
            )

        features = _features_from_matches(matches, timestamp, own_claimant_id=claimant_id)

        indexed = _IndexedClaim(claim_id, claimant_id, merchant_id, timestamp, phash)
        self._claims[claim_id] = indexed
        for band_index, band_value in enumerate(lsh_bands(phash, self.n_bands, self.band_bits)):
            self._band_buckets[(band_index, band_value)].append(claim_id)
        if clip_embedding is not None and self._clip_index is not None:
            self._clip_index.add(_as_normalised_row(clip_embedding, self._clip_dim))
            self._clip_row_to_claim.append(claim_id)
        self._last_timestamp = timestamp

        return features


def _features_from_matches(
    matches: list[ReuseMatch], timestamp: datetime, own_claimant_id: str
) -> ReuseFeatures:
    if not matches:
        return ReuseFeatures()

    # "Distinct claimants sharing this image" excludes the querying
    # claimant: a person resubmitting their own photo is not evidence of
    # a ring, and counting them would blur the two situations.
    other_claimants = {m.matched_claimant_id for m in matches} - {own_claimant_id}
    other_merchants = {m.matched_merchant_id for m in matches}
    first_seen = min(m.matched_timestamp for m in matches)

    return ReuseFeatures(
        n_matches=len(matches),
        n_distinct_claimants_sharing=len(other_claimants),
        n_distinct_merchants_sharing=len(other_merchants),
        min_hamming=min(m.hamming for m in matches),
        max_clip_similarity=max((m.clip_similarity or 0.0) for m in matches),
        days_since_first_seen=(timestamp - first_seen).total_seconds() / 86400.0,
        matched_prior_claim=True,
        matches=sorted(matches, key=lambda m: (m.hamming, m.matched_claim_id)),
    )


def _as_normalised_row(embedding: np.ndarray, expected_dim: int | None) -> np.ndarray:
    """A (1, d) float32 row with unit L2 norm, so FAISS inner product is
    cosine similarity. Rejects a dimension mismatch loudly - a silently
    wrong-sized embedding would produce meaningless similarities rather
    than an error."""
    row = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    if expected_dim is not None and row.shape[1] != expected_dim:
        raise ValueError(
            f"embedding has dimension {row.shape[1]}, index expects {expected_dim}"
        )
    norm = float(np.linalg.norm(row))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero embedding")
    return row / norm


def phash_to_vector(phash: int, bits: int = DEFAULT_PHASH_BITS) -> np.ndarray:
    """pHash as a +/-1 float vector, for cosine/inner-product search."""
    return np.array(
        [1.0 if (phash >> i) & 1 else -1.0 for i in reversed(range(bits))],
        dtype=np.float32,
    )
