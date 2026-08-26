"""P4 - claimant behaviour (PRAMAAN_v2_architecture.md Sec.4 L1, stage 1).

Aggregates over a claimant's history: how many claims they have filed,
how fast, at what values, from how many devices. Cheap enough to run in
the cascade's first stage alongside P1, because it is arithmetic over
already-stored records rather than image work.

**This pillar runs entirely on simulated data.** There is no public
dataset of refund claims with claimant history, so the ledger backing
these aggregates is synthetic (`benchmarks/simulate_ledger.py`, declared
in three places per Sec.5). Its measured contribution is therefore
evidence about our simulator, not about real claimants, and is reported
separately from the image pillars for exactly that reason. See
docs/DATA_CARD.md and docs/REAL_DATA_ONRAMP.md.

Like P3, aggregates are strictly backward-looking: a claim sees only its
claimant's *earlier* claims. `query_then_add` is the only public entry
point, and it computes before it records - the same structural guarantee,
for the same reason.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from pramaan.ingest.device import canonicalize_device
from pramaan.ingest.identity import ClaimIdentitySignals, resolve_canonical_identities

# A claimant with no history is the common case, not a suspicious one:
# most claimants are first-time. Features default to "no history" values
# that a monotone-constrained model reads as neutral, never as adverse.
NO_HISTORY_DAYS = 0.0


@dataclass
class BehaviourFeatures:
    """P4's contribution to the fused feature vector."""

    n_prior_claims: int = 0
    n_prior_claims_30d: int = 0
    n_prior_claims_90d: int = 0
    days_since_first_claim: float = NO_HISTORY_DAYS
    days_since_last_claim: float = NO_HISTORY_DAYS
    mean_prior_order_value: float = 0.0
    max_prior_order_value: float = 0.0
    order_value_vs_prior_mean: float = 1.0
    n_distinct_devices: int = 0
    n_distinct_merchants: int = 0
    n_distinct_categories: int = 0
    claim_to_order_days: float = 0.0
    is_first_claim: bool = True

    def as_dict(self) -> dict[str, float]:
        return {
            "behaviour_n_prior_claims": float(self.n_prior_claims),
            "behaviour_n_prior_claims_30d": float(self.n_prior_claims_30d),
            "behaviour_n_prior_claims_90d": float(self.n_prior_claims_90d),
            "behaviour_days_since_first_claim": self.days_since_first_claim,
            "behaviour_days_since_last_claim": self.days_since_last_claim,
            "behaviour_mean_prior_order_value": self.mean_prior_order_value,
            "behaviour_max_prior_order_value": self.max_prior_order_value,
            "behaviour_order_value_vs_prior_mean": self.order_value_vs_prior_mean,
            "behaviour_n_distinct_devices": float(self.n_distinct_devices),
            "behaviour_n_distinct_merchants": float(self.n_distinct_merchants),
            "behaviour_n_distinct_categories": float(self.n_distinct_categories),
            "behaviour_claim_to_order_days": self.claim_to_order_days,
            "behaviour_is_first_claim": float(self.is_first_claim),
        }


@dataclass
class _ClaimantHistory:
    timestamps: list[datetime] = field(default_factory=list)
    order_values: list[float] = field(default_factory=list)
    devices: set[str] = field(default_factory=set)
    merchants: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)


class BehaviourAggregator:
    """Backward-looking claimant aggregates.

    Keyed by **canonical identity**, not the raw `claimant_id`. That
    distinction is the whole point: a claimant opening a fresh account per
    claim looks like N first-time claimants under raw ids, and like one
    repeat claimant once phone/email/address are canonicalised
    (`pramaan.ingest.identity`). Resolving identity is what makes this
    pillar mean anything.
    """

    def __init__(self) -> None:
        self._history: dict[str, _ClaimantHistory] = defaultdict(_ClaimantHistory)
        self._identity_of: dict[str, str] = {}
        self._last_timestamp: datetime | None = None

    def __len__(self) -> int:
        return len(self._history)

    def register_identities(self, claims: list[ClaimIdentitySignals]) -> None:
        """Resolves canonical identities for a batch of claims up front.

        Entity resolution is inherently batch-shaped (it needs to compare
        claims against each other), so it happens once here rather than
        incrementally. This does NOT leak: identity is a property of the
        claimant's contact details, not of the claim outcome, and the
        aggregates built from it remain strictly backward-looking.
        Resolving identities across the whole corpus is exactly what
        `eval/entity_leakage_audit.py` already does to enforce split
        disjointness.
        """
        self._identity_of.update(resolve_canonical_identities(claims))

    def identity_for(self, claim_id: str, fallback_claimant_id: str) -> str:
        return self._identity_of.get(claim_id, fallback_claimant_id)

    def query_then_add(
        self,
        claim_id: str,
        claimant_id: str,
        merchant_id: str,
        category: str,
        timestamp: datetime,
        order_date: datetime,
        order_value: float,
        device_ua: str | None = None,
        device_screen: str | None = None,
        device_timezone: str | None = None,
        device_fonts: list[str] | None = None,
    ) -> BehaviourFeatures:
        """Aggregates this claimant's *earlier* claims, then records this
        one. The current claim never contributes to its own features."""
        identity = self.identity_for(claim_id, claimant_id)
        history = self._history[identity]

        features = BehaviourFeatures(
            claim_to_order_days=(timestamp - order_date).total_seconds() / 86400.0
        )

        if history.timestamps:
            features.is_first_claim = False
            features.n_prior_claims = len(history.timestamps)
            features.n_prior_claims_30d = sum(
                1 for t in history.timestamps if (timestamp - t).days <= 30
            )
            features.n_prior_claims_90d = sum(
                1 for t in history.timestamps if (timestamp - t).days <= 90
            )
            features.days_since_first_claim = (
                timestamp - min(history.timestamps)
            ).total_seconds() / 86400.0
            features.days_since_last_claim = (
                timestamp - max(history.timestamps)
            ).total_seconds() / 86400.0

            mean_prior = sum(history.order_values) / len(history.order_values)
            features.mean_prior_order_value = mean_prior
            features.max_prior_order_value = max(history.order_values)
            features.order_value_vs_prior_mean = order_value / mean_prior if mean_prior else 1.0

            features.n_distinct_devices = len(history.devices)
            features.n_distinct_merchants = len(history.merchants)
            features.n_distinct_categories = len(history.categories)

        history.timestamps.append(timestamp)
        history.order_values.append(order_value)
        history.merchants.add(merchant_id)
        history.categories.add(category)
        device = canonicalize_device(device_ua, device_screen, device_timezone, device_fonts)
        if device:
            history.devices.add(device)
        self._last_timestamp = timestamp

        return features
