"""A small, tested cuckoo filter (Sec.4 L6).

Written here rather than pulled from PyPI because the available packages
are unmaintained and this is ~150 lines of well-specified data structure
that the federated index depends on for a correctness property: **no
false negatives**.

That asymmetry is the whole reason a filter is usable here. A false
positive means one extra candidate check. A false negative would mean a
reused image silently invisible to the consortium — the exact failure the
index exists to prevent.

Cuckoo over Bloom because it supports **deletion**, which the 180-day
decay rule (Sec.4 L6) requires: evidence has to age out, and a Bloom
filter cannot remove an entry without rebuilding.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

DEFAULT_BUCKET_SIZE = 4
DEFAULT_FINGERPRINT_BITS = 16
DEFAULT_MAX_KICKS = 500


class CuckooFilterFull(RuntimeError):
    """Raised when insertion fails after `max_kicks` relocations.

    Deliberately loud. Silently dropping an insert would produce exactly
    the false negative the structure promises never to have.
    """


@dataclass
class CuckooFilterStats:
    capacity: int
    n_items: int
    bucket_size: int
    fingerprint_bits: int

    @property
    def load_factor(self) -> float:
        return self.n_items / (self.capacity * self.bucket_size) if self.capacity else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "capacity": float(self.capacity),
            "n_items": float(self.n_items),
            "load_factor": self.load_factor,
            "bucket_size": float(self.bucket_size),
            "fingerprint_bits": float(self.fingerprint_bits),
        }


class CuckooFilter:
    """Approximate set membership with deletion and no false negatives."""

    def __init__(
        self,
        capacity: int = 4096,
        bucket_size: int = DEFAULT_BUCKET_SIZE,
        fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS,
        max_kicks: int = DEFAULT_MAX_KICKS,
        seed: int = 1337,
    ) -> None:
        if capacity & (capacity - 1) != 0:
            raise ValueError("capacity must be a power of two (index masking assumes it)")
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self.max_kicks = max_kicks
        self._buckets: list[list[int]] = [[] for _ in range(capacity)]
        self._n_items = 0
        # Seeded RNG so eviction choice is reproducible - the determinism
        # test (Sec.6) covers everything downstream of this.
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._n_items

    def __contains__(self, item: bytes) -> bool:
        return self.contains(item)

    # --- hashing ------------------------------------------------------

    def _fingerprint(self, item: bytes) -> int:
        digest = hashlib.blake2b(item, digest_size=8).digest()
        value = int.from_bytes(digest, "big") & ((1 << self.fingerprint_bits) - 1)
        # Fingerprint 0 is reserved as "empty", so fold it to 1. Without
        # this a legitimate item could be mistaken for a free slot.
        return value or 1

    def _index(self, item: bytes) -> int:
        digest = hashlib.blake2b(item, digest_size=8, person=b"idx").digest()
        return int.from_bytes(digest, "big") % self.capacity

    def _alt_index(self, index: int, fingerprint: int) -> int:
        """The partial-key trick: the alternate bucket is derivable from
        the fingerprint alone, so relocation never needs the original
        item - which is what lets the filter store hashes and nothing
        else."""
        digest = hashlib.blake2b(
            fingerprint.to_bytes(4, "big"), digest_size=8, person=b"alt"
        ).digest()
        return (index ^ int.from_bytes(digest, "big")) % self.capacity

    # --- operations ---------------------------------------------------

    def insert(self, item: bytes) -> None:
        fingerprint = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fingerprint)

        for index in (i1, i2):
            if len(self._buckets[index]) < self.bucket_size:
                self._buckets[index].append(fingerprint)
                self._n_items += 1
                return

        index = self._rng.choice((i1, i2))
        for _ in range(self.max_kicks):
            slot = self._rng.randrange(self.bucket_size)
            fingerprint, self._buckets[index][slot] = self._buckets[index][slot], fingerprint
            index = self._alt_index(index, fingerprint)
            if len(self._buckets[index]) < self.bucket_size:
                self._buckets[index].append(fingerprint)
                self._n_items += 1
                return

        raise CuckooFilterFull(
            f"insertion failed after {self.max_kicks} relocations at load "
            f"{self.stats().load_factor:.2f}; grow the filter rather than "
            "dropping the item - a dropped insert becomes a false negative, "
            "which this structure must never have"
        )

    def contains(self, item: bytes) -> bool:
        fingerprint = self._fingerprint(item)
        i1 = self._index(item)
        return fingerprint in self._buckets[i1] or fingerprint in self._buckets[
            self._alt_index(i1, fingerprint)
        ]

    def delete(self, item: bytes) -> bool:
        """Removes one copy. Needed for the 180-day decay rule."""
        fingerprint = self._fingerprint(item)
        i1 = self._index(item)
        for index in (i1, self._alt_index(i1, fingerprint)):
            if fingerprint in self._buckets[index]:
                self._buckets[index].remove(fingerprint)
                self._n_items -= 1
                return True
        return False

    def stats(self) -> CuckooFilterStats:
        return CuckooFilterStats(
            capacity=self.capacity,
            n_items=self._n_items,
            bucket_size=self.bucket_size,
            fingerprint_bits=self.fingerprint_bits,
        )

    def expected_false_positive_rate(self) -> float:
        """Standard cuckoo bound: 2b / 2^f, scaled by load."""
        return min(
            1.0,
            (2 * self.bucket_size / (2**self.fingerprint_bits))
            * max(self.stats().load_factor, 1e-9),
        )
