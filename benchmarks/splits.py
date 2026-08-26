"""Verifies the 4-way split (PRAMAAN_v2_architecture.md Sec.6) actually
holds on generated PRAMAAN-Bench-v1 output - not assumed correct just
because `simulate_ledger.py` was designed to make it hold by
construction. Four checks:

1. **Generator-family holdout** - a fraud_synthetic_image claim's
   generator must be in the split's allowed family set.
2. **Temporal** - all train claims strictly before all calibration
   claims, strictly before all test claims.
3. **Entity-disjoint** - delegates to `eval/entity_leakage_audit.py`
   rather than duplicating its logic.
4. **Ring-disjoint** - `image_group_id` clusters (claims sharing/derived
   from the same source image) must not span more than one split.

`benchmarks/build_bench.py` calls `reconcile_splits()` then
`verify_splits()`, and hard-fails the build if anything is still violated
- the Phase 1 gate ("leakage audits green before any Phase 2 model
code", PROGRESS.md) applies here too, not just to identity leakage.

**Why reconciliation is needed at all**, given the simulator is designed
to be leakage-free by construction: it isn't, quite, and that is not a
fixable simulator bug. Entity resolution is *fuzzy* - two independently
generated claimants can land on genuinely ambiguous addresses (observed:
`H.No. 06 Tata Marg, Sri Ganganagar` vs `H.No. 06 Sarraf, Sri
Ganganagar`, same PIN, same house number, different street) that a
reasonable matcher may merge. No threshold tuning drives that
probability to zero, and real merchant data has the same property. So
rather than trusting the matcher to be perfect, the build resolves
entity+ring components across the whole corpus and *enforces* one cohort
per component, reporting how many claims it dropped to do so. See
`docs/DATA_CARD.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from benchmarks.simulate_ledger import GENERATOR_HOLDOUT
from eval.entity_leakage_audit import (
    LeakageViolation,
    build_identity_signals,
    find_leakage_violations,
)
from pramaan.common.union_find import UnionFind
from pramaan.ingest.identity import resolve_canonical_identities

REQUIRED_COLUMNS = {
    "claim_id",
    "split",
    "fraud_class",
    "generator_family",
    "claim_timestamp",
    "image_group_id",
}

_SPLIT_ORDER = ("train", "calibration", "test")


@dataclass(frozen=True)
class SplitVerificationReport:
    generator_holdout_violations: list[str]
    temporal_violations: list[str]
    entity_leakage_violations: list[LeakageViolation]
    ring_leakage_violations: list[str]

    @property
    def is_valid(self) -> bool:
        return not (
            self.generator_holdout_violations
            or self.temporal_violations
            or self.entity_leakage_violations
            or self.ring_leakage_violations
        )

    def describe(self) -> str:
        if self.is_valid:
            return "OK: all 4 split constraints hold."
        lines = ["SPLIT VERIFICATION FAILED:"]
        for msg in self.generator_holdout_violations:
            lines.append(f"  [generator-holdout] {msg}")
        for msg in self.temporal_violations:
            lines.append(f"  [temporal] {msg}")
        for leak in self.entity_leakage_violations:
            lines.append(
                f"  [entity-disjoint] identity={leak.canonical_identity_id} "
                f"splits={sorted(leak.splits)} claims={leak.claim_ids}"
            )
        for msg in self.ring_leakage_violations:
            lines.append(f"  [ring-disjoint] {msg}")
        return "\n".join(lines)


def _check_generator_holdout(claims: pd.DataFrame) -> list[str]:
    violations = []
    synth = claims[claims["fraud_class"] == "fraud_synthetic_image"]
    for split, allowed in GENERATOR_HOLDOUT.items():
        subset = synth[synth["split"] == split]
        used = set(subset["generator_family"].dropna().unique())
        disallowed = used - set(allowed)
        if disallowed:
            violations.append(
                f"split={split} used disallowed generator families: {sorted(disallowed)}"
            )
    return violations


def _check_temporal(claims: pd.DataFrame) -> list[str]:
    violations = []
    timestamps = pd.to_datetime(claims["claim_timestamp"])
    present_order = [s for s in _SPLIT_ORDER if (claims["split"] == s).any()]
    for earlier, later in zip(present_order, present_order[1:], strict=False):
        max_earlier = timestamps[claims["split"] == earlier].max()
        min_later = timestamps[claims["split"] == later].min()
        if not (max_earlier < min_later):
            violations.append(
                f"max(claim_timestamp) for split={earlier} ({max_earlier}) is not "
                f"strictly before min(claim_timestamp) for split={later} ({min_later})"
            )
    return violations


def _check_ring_disjoint(claims: pd.DataFrame) -> list[str]:
    violations = []
    for group_id, splits in claims.groupby("image_group_id")["split"].unique().items():
        if len(splits) > 1:
            violations.append(f"image_group_id={group_id} spans splits {sorted(splits)}")
    return violations


def verify_splits(
    claims: pd.DataFrame, address_match_threshold: float = 85.0
) -> SplitVerificationReport:
    missing = REQUIRED_COLUMNS - set(claims.columns)
    if missing:
        raise ValueError(f"claims table missing required columns: {sorted(missing)}")

    return SplitVerificationReport(
        generator_holdout_violations=_check_generator_holdout(claims),
        temporal_violations=_check_temporal(claims),
        entity_leakage_violations=find_leakage_violations(claims, address_match_threshold),
        ring_leakage_violations=_check_ring_disjoint(claims),
    )


@dataclass(frozen=True)
class ReconciliationReport:
    n_claims_before: int
    n_claims_after: int
    n_dropped: int
    n_components_reconciled: int
    dropped_claim_ids: tuple[str, ...]

    @property
    def drop_rate(self) -> float:
        return self.n_dropped / self.n_claims_before if self.n_claims_before else 0.0

    def describe(self) -> str:
        if self.n_dropped == 0:
            return f"Reconciliation: no cross-split components ({self.n_claims_before} claims)."
        return (
            f"Reconciliation: dropped {self.n_dropped} claim(s) "
            f"({self.drop_rate:.3%}) from {self.n_components_reconciled} "
            f"cross-split component(s); {self.n_claims_before} -> {self.n_claims_after}."
        )


def reconcile_splits(
    claims: pd.DataFrame, address_match_threshold: float = 85.0
) -> tuple[pd.DataFrame, ReconciliationReport]:
    """Enforces one split per entity+ring component, dropping the claims
    that would otherwise straddle a boundary.

    Two claims are in the same component if they share a canonical
    identity (fuzzy - see `pramaan.ingest.identity`) OR the same
    `image_group_id` (a reuse ring). Both relations are unioned before
    assignment, because handling them independently can conflict: claim A
    can share an identity with B while sharing an image with C, and B and
    C may sit in different splits.

    Each component keeps the split of its **earliest** claim ("first seen
    wins", mirroring the federated index's first-seen-immunity rule in
    Sec.4 L6) and every member in another split is dropped.

    Dropping rather than reassigning is deliberate: each split occupies a
    disjoint time window (`simulate_ledger.COHORT_WINDOW_DAYS`), so moving
    a claim into another split would put its timestamp outside that
    window and break the temporal constraint instead. Components are tiny
    (2-3 claims) and rare, so the observed loss is well under 0.1% - and
    it is reported rather than silently absorbed.
    """
    missing = REQUIRED_COLUMNS - set(claims.columns)
    if missing:
        raise ValueError(f"claims table missing required columns: {sorted(missing)}")

    claim_ids = [str(c) for c in claims["claim_id"]]
    uf = UnionFind(claim_ids)

    canonical = resolve_canonical_identities(
        build_identity_signals(claims), address_match_threshold
    )
    by_identity: dict[str, list[str]] = {}
    for claim_id, canonical_id in canonical.items():
        by_identity.setdefault(canonical_id, []).append(claim_id)
    for members in by_identity.values():
        for other in members[1:]:
            uf.union(members[0], other)

    for _, members_idx in claims.groupby("image_group_id").groups.items():
        members = [str(c) for c in claims.loc[members_idx, "claim_id"]]
        for other in members[1:]:
            uf.union(members[0], other)

    split_by_claim = dict(zip(claim_ids, claims["split"], strict=True))
    ts_by_claim = dict(zip(claim_ids, pd.to_datetime(claims["claim_timestamp"]), strict=True))

    dropped: list[str] = []
    n_components_reconciled = 0
    for members in uf.clusters().values():
        splits = {split_by_claim[m] for m in members}
        if len(splits) <= 1:
            continue
        n_components_reconciled += 1
        # Tie-break on claim_id so the kept split is deterministic even if
        # two members share a timestamp.
        anchor = min(members, key=lambda m: (ts_by_claim[m], m))
        keep_split = split_by_claim[anchor]
        dropped.extend(m for m in members if split_by_claim[m] != keep_split)

    dropped_set = set(dropped)
    reconciled = claims[~claims["claim_id"].astype(str).isin(dropped_set)].reset_index(drop=True)

    return reconciled, ReconciliationReport(
        n_claims_before=len(claims),
        n_claims_after=len(reconciled),
        n_dropped=len(dropped),
        n_components_reconciled=n_components_reconciled,
        dropped_claim_ids=tuple(sorted(dropped)),
    )
