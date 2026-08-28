"""The federated hash index and its anti-poisoning rules (Sec.4 L6).

The reuse graph is far more powerful across merchants than within one —
a ring operating against three merchants is invisible to each of them
alone. Making that lawful is architecture, not paperwork.

## What crosses the boundary

Merchants publish **salted perceptual-hash bands** (HMAC under a
consortium-rotated key) into a cuckoo filter. Never images, never names,
phone numbers or addresses. A membership query returns a boolean plus a
**differentially-private count**.

The salt matters: raw pHash bands are invertible enough to be a privacy
problem across a consortium, and HMAC under a rotating key means a
participant who leaves cannot keep querying yesterday's index.

## A shared index is itself an attack surface

Four rules, each answering a specific attack:

1. **First-seen immunity** — the earliest claimant on a cluster is never
   penalised by it. Without this I can burn a rival's genuine claim by
   submitting their photo first. Already enforced in
   `pillars/rings.py`; enforced again here because the federated path
   does not go through that code.
2. **k-independence** — a cluster contributes risk only once it holds ≥2
   claimants from ≥2 *merchant-independent* identity groups. One
   merchant cannot manufacture a ring alone.
3. **Rate limits + append-only signed log** — poisoning attempts are
   detectable after the fact even if not prevented in the moment.
4. **Decay** — cluster evidence half-lives at 180 days, so a poisoned
   entry cannot persist indefinitely.

Measured results for rules 1 and 2, including a poisoning attack run
against this index, live in `docs/THREAT_MODEL.md`.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from pramaan.federated.cuckoo_filter import CuckooFilter
from pramaan.pillars.p3_reuse import lsh_bands

DECAY_HALF_LIFE_DAYS = 180
DEFAULT_DP_EPSILON = 1.0
MIN_INDEPENDENT_GROUPS = 2


def salted_band(band_index: int, band_value: int, key: bytes) -> bytes:
    """HMAC of one LSH band under the consortium key.

    Both the index and the value are covered, so band 3 holding value 7
    cannot collide with band 7 holding value 3.
    """
    message = f"{band_index}:{band_value}".encode()
    return hmac.new(key, message, hashlib.sha256).digest()[:16]


@dataclass
class ClusterEvidence:
    """What the consortium knows about one image cluster.

    Deliberately holds identity *group* ids rather than claimant ids: the
    index must be able to apply k-independence without learning who the
    claimants are.
    """

    first_seen: datetime
    first_seen_group: str
    first_seen_merchant: str
    identity_groups: set[str] = field(default_factory=set)
    merchants: set[str] = field(default_factory=set)
    last_seen: datetime | None = None
    n_submissions: int = 0

    def independent_group_count(self) -> int:
        return len(self.identity_groups)

    def is_actionable(self, min_groups: int = MIN_INDEPENDENT_GROUPS) -> bool:
        """k-independence: enough distinct identity groups, from enough
        distinct merchants, that no single party manufactured it."""
        return (
            self.independent_group_count() >= min_groups
            and len(self.merchants) >= min_groups
        )

    def decayed_weight(self, as_of: datetime, half_life_days: int = DECAY_HALF_LIFE_DAYS) -> float:
        """Exponential decay so a poisoned entry cannot persist forever."""
        if self.last_seen is None:
            return 0.0
        age_days = (as_of - self.last_seen).total_seconds() / 86400.0
        return float(0.5 ** (age_days / half_life_days))


@dataclass
class FederatedQueryResult:
    in_index: bool
    dp_count: float
    actionable: bool
    is_first_seen: bool
    independent_groups: int
    decayed_weight: float
    suppression_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "in_index": self.in_index,
            "dp_count": self.dp_count,
            "actionable": self.actionable,
            "is_first_seen": self.is_first_seen,
            "independent_groups": self.independent_groups,
            "decayed_weight": self.decayed_weight,
            "suppression_reason": self.suppression_reason,
        }


@dataclass
class RateLimitState:
    submissions: list[datetime] = field(default_factory=list)


class FederatedIndex:
    """Salted-band membership with DP counts and anti-poisoning rules."""

    def __init__(
        self,
        key: bytes,
        capacity: int = 8192,
        dp_epsilon: float = DEFAULT_DP_EPSILON,
        max_submissions_per_window: int = 50,
        rate_limit_window_hours: int = 24,
        seed: int = 1337,
    ) -> None:
        self.key = key
        self.dp_epsilon = dp_epsilon
        self.max_submissions_per_window = max_submissions_per_window
        self.rate_limit_window = timedelta(hours=rate_limit_window_hours)

        self._filter = CuckooFilter(capacity=capacity, seed=seed)
        self._clusters: dict[str, ClusterEvidence] = {}
        self._rate_limits: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._audit_log: list[dict[str, object]] = []
        self._rng = np.random.default_rng(seed)
        self._epsilon_spent = 0.0

    # --- publishing ---------------------------------------------------

    def _cluster_key(self, phash: int) -> str:
        """Stable cluster id from the salted bands. Two images sharing
        enough bands map to the same key without either being stored."""
        bands = lsh_bands(phash)
        digest = hashlib.sha256()
        for index, value in enumerate(bands):
            digest.update(salted_band(index, value, self.key))
        return digest.hexdigest()[:24]

    def publish(
        self,
        phash: int,
        identity_group: str,
        merchant_id: str,
        timestamp: datetime,
    ) -> bool:
        """Adds one observation. Returns False if rate-limited.

        `identity_group` is a canonical identity cluster id, not a
        claimant id - the consortium learns that two submissions came
        from the same person without learning who.
        """
        state = self._rate_limits[merchant_id]
        cutoff = timestamp - self.rate_limit_window
        state.submissions = [t for t in state.submissions if t > cutoff]
        if len(state.submissions) >= self.max_submissions_per_window:
            self._log(
                "rate_limited", merchant_id=merchant_id, timestamp=timestamp,
                n_in_window=len(state.submissions),
            )
            return False
        state.submissions.append(timestamp)

        cluster_key = self._cluster_key(phash)
        for index, value in enumerate(lsh_bands(phash)):
            band = salted_band(index, value, self.key)
            if not self._filter.contains(band):
                self._filter.insert(band)

        cluster = self._clusters.get(cluster_key)
        if cluster is None:
            cluster = ClusterEvidence(
                first_seen=timestamp,
                first_seen_group=identity_group,
                first_seen_merchant=merchant_id,
            )
            self._clusters[cluster_key] = cluster

        cluster.identity_groups.add(identity_group)
        cluster.merchants.add(merchant_id)
        cluster.n_submissions += 1
        if cluster.last_seen is None or timestamp > cluster.last_seen:
            cluster.last_seen = timestamp

        self._log(
            "publish", merchant_id=merchant_id, timestamp=timestamp,
            cluster=cluster_key, n_groups=cluster.independent_group_count(),
        )
        return True

    # --- querying -----------------------------------------------------

    def query(
        self,
        phash: int,
        identity_group: str,
        as_of: datetime,
    ) -> FederatedQueryResult:
        """Membership plus a DP count, with the anti-poisoning rules applied."""
        cluster_key = self._cluster_key(phash)
        bands = lsh_bands(phash)
        in_index = any(
            self._filter.contains(salted_band(i, v, self.key)) for i, v in enumerate(bands)
        )

        cluster = self._clusters.get(cluster_key)
        if cluster is None:
            return FederatedQueryResult(
                in_index=in_index,
                dp_count=self._dp_count(0),
                actionable=False,
                is_first_seen=False,
                independent_groups=0,
                decayed_weight=0.0,
                suppression_reason=None if in_index else "not in index",
            )

        # Rule 1: first-seen immunity. Checked BEFORE actionability, so
        # the earliest claimant is immune even to a cluster that later
        # becomes a genuine ring.
        is_first_seen = identity_group == cluster.first_seen_group
        if is_first_seen:
            return FederatedQueryResult(
                in_index=in_index,
                dp_count=self._dp_count(cluster.independent_group_count()),
                actionable=False,
                is_first_seen=True,
                independent_groups=cluster.independent_group_count(),
                decayed_weight=cluster.decayed_weight(as_of),
                suppression_reason=(
                    "first-seen immunity: this identity was the earliest on the "
                    "cluster and is never penalised by it"
                ),
            )

        # Rule 2: k-independence.
        actionable = cluster.is_actionable()
        reason = (
            None
            if actionable
            else (
                f"k-independence not met: {cluster.independent_group_count()} identity "
                f"group(s) across {len(cluster.merchants)} merchant(s); "
                f"{MIN_INDEPENDENT_GROUPS} of each required"
            )
        )

        return FederatedQueryResult(
            in_index=in_index,
            dp_count=self._dp_count(cluster.independent_group_count()),
            actionable=actionable,
            is_first_seen=False,
            independent_groups=cluster.independent_group_count(),
            # Rule 4: decay.
            decayed_weight=cluster.decayed_weight(as_of),
            suppression_reason=reason,
        )

    def _dp_count(self, true_count: int) -> float:
        """Laplace mechanism, sensitivity 1 (one claimant changes the
        count by at most one). Epsilon spend is tracked and reported."""
        self._epsilon_spent += self.dp_epsilon
        noise = self._rng.laplace(0.0, 1.0 / self.dp_epsilon)
        return float(max(0.0, true_count + noise))

    # --- rule 3: append-only signed log --------------------------------

    def _log(self, event: str, **fields: object) -> None:
        """Append-only, hash-chained. Each entry commits to the previous
        one, so removing or editing an entry breaks the chain - which is
        what makes a poisoning attempt detectable after the fact even
        though it was not prevented."""
        previous = self._audit_log[-1]["entry_hash"] if self._audit_log else "genesis"
        payload = {"event": event, "prev": previous, **{k: str(v) for k, v in fields.items()}}
        entry_hash = hashlib.sha256(
            hmac.new(
                self.key, repr(sorted(payload.items())).encode(), hashlib.sha256
            ).digest()
        ).hexdigest()[:32]
        self._audit_log.append({**payload, "entry_hash": entry_hash})

    def verify_log(self) -> bool:
        """Recomputes the chain. False means an entry was edited or removed."""
        previous = "genesis"
        for entry in self._audit_log:
            payload = {k: v for k, v in entry.items() if k != "entry_hash"}
            if payload.get("prev") != previous:
                return False
            expected = hashlib.sha256(
                hmac.new(
                    self.key, repr(sorted(payload.items())).encode(), hashlib.sha256
                ).digest()
            ).hexdigest()[:32]
            if expected != entry["entry_hash"]:
                return False
            previous = str(entry["entry_hash"])
        return True

    # --- reporting ----------------------------------------------------

    def stats(self) -> dict[str, object]:
        return {
            "n_clusters": len(self._clusters),
            "n_actionable_clusters": sum(
                1 for c in self._clusters.values() if c.is_actionable()
            ),
            "filter": self._filter.stats().as_dict(),
            "expected_false_positive_rate": self._filter.expected_false_positive_rate(),
            "epsilon_spent": self._epsilon_spent,
            "audit_log_entries": len(self._audit_log),
            "audit_log_intact": self.verify_log(),
        }

    def prune_expired(self, as_of: datetime, min_weight: float = 0.01) -> int:
        """Rule 4: drops clusters decayed below usefulness. Returns the
        number removed."""
        expired = [
            key
            for key, cluster in self._clusters.items()
            if cluster.decayed_weight(as_of) < min_weight
        ]
        for key in expired:
            del self._clusters[key]
        if expired:
            self._log("prune", n_removed=len(expired), as_of=as_of)
        return len(expired)


def half_life_weight(age_days: float, half_life_days: int = DECAY_HALF_LIFE_DAYS) -> float:
    return float(math.pow(0.5, age_days / half_life_days))
