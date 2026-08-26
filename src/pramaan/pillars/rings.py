"""Ring detection: score image clusters, not just claims
(PRAMAAN_v2_architecture.md Sec.4 L1).

A single claim reusing one prior image is weak evidence. The same image
circulating among several unrelated claimants is a *ring*, and it is a
much stronger signal than any per-claim feature - which is why the spec
asks for both claim-level and ring-level predictions, evaluated
separately.

The structure is a temporal bipartite graph, `claimant <-> image_cluster`,
built incrementally from P3's match stream. Two properties matter:

**Time-ordered.** Rings are grown only from matches P3 already produced
under its strictly-earlier constraint, so a ring's state at claim N
reflects only claims 1..N-1. Ring features are therefore usable as model
inputs without leaking, and `time_to_detection` is a real forward-looking
measurement rather than hindsight.

**First-seen immunity.** The earliest claimant on a cluster is never
penalised by that cluster. Otherwise submitting a rival's genuine photo
first would let an attacker burn their claim - the poisoning attack the
federated index's anti-poisoning rules (Sec.4 L6) also defend against.
The rule belongs here too, because the ring graph is attackable by the
same move even without federation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pramaan.common.union_find import UnionFind
from pramaan.pillars.p3_reuse import ReuseFeatures

# A cluster only contributes risk once it holds at least this many
# distinct claimants. One person reusing their own photo across two
# claims is ordinary behaviour, not a ring.
DEFAULT_MIN_CLAIMANTS_FOR_RING = 2


@dataclass
class ImageCluster:
    """One image circulating across claims."""

    cluster_id: str
    claim_ids: list[str] = field(default_factory=list)
    claimant_ids: set[str] = field(default_factory=set)
    merchant_ids: set[str] = field(default_factory=set)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    first_claim_id: str | None = None
    first_claimant_id: str | None = None

    @property
    def size(self) -> int:
        return len(self.claim_ids)

    @property
    def n_distinct_claimants(self) -> int:
        return len(self.claimant_ids)

    @property
    def n_distinct_merchants(self) -> int:
        return len(self.merchant_ids)

    def is_ring(self, min_claimants: int = DEFAULT_MIN_CLAIMANTS_FOR_RING) -> bool:
        return self.n_distinct_claimants >= min_claimants

    def span_days(self) -> float:
        if self.first_seen is None or self.last_seen is None:
            return 0.0
        return (self.last_seen - self.first_seen).total_seconds() / 86400.0


@dataclass
class RingFeatures:
    """Ring-level contribution to a claim's feature vector."""

    in_ring: bool = False
    ring_size: int = 0
    ring_distinct_claimants: int = 0
    ring_distinct_merchants: int = 0
    ring_span_days: float = 0.0
    is_first_seen_in_cluster: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "ring_in_ring": float(self.in_ring),
            "ring_size": float(self.ring_size),
            "ring_distinct_claimants": float(self.ring_distinct_claimants),
            "ring_distinct_merchants": float(self.ring_distinct_merchants),
            "ring_span_days": float(self.ring_span_days),
            "ring_is_first_seen": float(self.is_first_seen_in_cluster),
        }


class RingDetector:
    """Builds image clusters incrementally from P3's match stream.

    Fed one claim at a time in timestamp order, alongside the
    `ReuseFeatures` P3 produced for it. Clusters are merged with
    union-find: a claim matching two previously-separate clusters joins
    them, because a shared image is a transitive relation even when P3
    only ever compared pairs.
    """

    def __init__(self, min_claimants_for_ring: int = DEFAULT_MIN_CLAIMANTS_FOR_RING) -> None:
        self.min_claimants_for_ring = min_claimants_for_ring
        self._uf = UnionFind([])
        self._cluster_of_claim: dict[str, str] = {}
        self._clusters: dict[str, ImageCluster] = {}

    def _ensure_node(self, claim_id: str) -> None:
        if claim_id not in self._uf.parent:
            self._uf.parent[claim_id] = claim_id

    def observe(
        self,
        claim_id: str,
        claimant_id: str,
        merchant_id: str,
        timestamp: datetime,
        reuse: ReuseFeatures,
    ) -> RingFeatures:
        """Adds one claim, merging it into any cluster it matched, and
        returns the ring features *as visible at this claim's moment* -
        never including the claim's own future."""
        self._ensure_node(claim_id)
        for match in reuse.matches:
            self._ensure_node(match.matched_claim_id)
            self._uf.union(claim_id, match.matched_claim_id)

        root = self._uf.find(claim_id)
        # Members are the union-find component; the cluster's canonical id
        # is derived from the earliest claim in it so it is stable as the
        # component grows and merges.
        members = [c for c in self._uf.parent if self._uf.find(c) == root]
        cluster_id = f"cluster_{min(members)}"

        # Always fold every pre-existing cluster covered by this component
        # into one, rather than only when the canonical id is new. A merge
        # frequently lands on an id that already exists (the surviving
        # cluster keeps the component's minimum claim id), and treating
        # that as "nothing to merge" silently orphans the other half.
        old_ids = {self._cluster_of_claim[m] for m in members if m in self._cluster_of_claim}
        cluster = ImageCluster(cluster_id=cluster_id)
        for old_id in old_ids:
            old = self._clusters.pop(old_id, None)
            if old is None:
                continue
            cluster.claim_ids.extend(old.claim_ids)
            cluster.claimant_ids |= old.claimant_ids
            cluster.merchant_ids |= old.merchant_ids
            if old.first_seen is not None and (
                cluster.first_seen is None or old.first_seen < cluster.first_seen
            ):
                cluster.first_seen = old.first_seen
                cluster.first_claim_id = old.first_claim_id
                cluster.first_claimant_id = old.first_claimant_id
            if old.last_seen is not None and (
                cluster.last_seen is None or old.last_seen > cluster.last_seen
            ):
                cluster.last_seen = old.last_seen
        self._clusters[cluster_id] = cluster

        is_first = cluster.first_seen is None or timestamp < cluster.first_seen

        cluster.claim_ids.append(claim_id)
        cluster.claimant_ids.add(claimant_id)
        cluster.merchant_ids.add(merchant_id)
        if is_first:
            cluster.first_seen = timestamp
            cluster.first_claim_id = claim_id
            cluster.first_claimant_id = claimant_id
        if cluster.last_seen is None or timestamp > cluster.last_seen:
            cluster.last_seen = timestamp

        for member in members:
            self._cluster_of_claim[member] = cluster_id
        self._cluster_of_claim[claim_id] = cluster_id

        # First-seen immunity: the earliest claimant on a cluster carries
        # no ring risk from it, so an attacker cannot burn a rival's
        # genuine claim by submitting their photo first.
        if is_first or claimant_id == cluster.first_claimant_id:
            return RingFeatures(is_first_seen_in_cluster=True)

        if not cluster.is_ring(self.min_claimants_for_ring):
            return RingFeatures(is_first_seen_in_cluster=False)

        return RingFeatures(
            in_ring=True,
            ring_size=cluster.size,
            ring_distinct_claimants=cluster.n_distinct_claimants,
            ring_distinct_merchants=cluster.n_distinct_merchants,
            ring_span_days=cluster.span_days(),
            is_first_seen_in_cluster=False,
        )

    def clusters(self) -> list[ImageCluster]:
        return list(self._clusters.values())

    def rings(self) -> list[ImageCluster]:
        """Clusters that meet the ring bar (>= min distinct claimants)."""
        return [c for c in self._clusters.values() if c.is_ring(self.min_claimants_for_ring)]

    def cluster_for(self, claim_id: str) -> ImageCluster | None:
        cluster_id = self._cluster_of_claim.get(claim_id)
        return self._clusters.get(cluster_id) if cluster_id else None
