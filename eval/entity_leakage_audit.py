"""Asserts zero canonical-identity overlap across splits and fails loudly
if violated (PRAMAAN_v2_architecture.md Sec.4 L0 / Sec.6).

If the same human appears as two "different" claimants across a
train/calibration/test boundary, the split is contaminated and any
certificate (docs/GUARANTEE.md) computed on it is fiction. This is why
the build order gates all of Phase 2 onward on this script being green
(see PROGRESS.md) - resolve entity clusters via
`pramaan.ingest.identity.resolve_canonical_identities`, then check that
no cluster with more than one member spans more than one split.

CLI usage:
    python eval/entity_leakage_audit.py --claims path/to/claims.csv \
        [--pin-threshold 85]

`claims.csv` must have at least `claim_id` and `split` columns, plus any
of `phone`, `email`, `address`, `pin` that are available - columns that
are entirely absent are simply not used as identity signals, but a
column that exists with sparse/NaN values is handled per-row (NaN is
treated as "no signal for this claim", not as a literal string).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from pramaan.ingest.identity import ClaimIdentitySignals, resolve_canonical_identities

REQUIRED_COLUMNS = {"claim_id", "split"}
OPTIONAL_SIGNAL_COLUMNS = ("phone", "email", "address", "pin")


@dataclass(frozen=True)
class LeakageViolation:
    canonical_identity_id: str
    splits: frozenset[str]
    claim_ids: tuple[str, ...]


def _clean(value: object) -> str | None:
    """NaN-safe stringification: pandas represents a missing cell as
    float('nan'), which is truthy in Python and would otherwise sail past
    the "is this signal present" checks in ingest.* and be canonicalised
    as the literal string 'nan'."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def build_identity_signals(claims: pd.DataFrame) -> list[ClaimIdentitySignals]:
    """Extracts the identity signals this audit clusters on. Shared with
    `benchmarks/splits.py`'s reconciliation pass so both derive entity
    clusters from exactly the same inputs - a reconciliation that used
    different signals than the audit could 'fix' a corpus the audit then
    still rejects."""
    return [
        ClaimIdentitySignals(
            claim_id=str(row["claim_id"]),
            phone_raw=_clean(row.get("phone")),
            email_raw=_clean(row.get("email")),
            address_raw=_clean(row.get("address")),
            pin_raw=_clean(row.get("pin")),
        )
        for _, row in claims.iterrows()
    ]


def find_leakage_violations(
    claims: pd.DataFrame, address_match_threshold: float = 85.0
) -> list[LeakageViolation]:
    missing = REQUIRED_COLUMNS - set(claims.columns)
    if missing:
        raise ValueError(f"claims table missing required columns: {sorted(missing)}")

    signals = build_identity_signals(claims)
    canonical = resolve_canonical_identities(signals, address_match_threshold)

    split_by_claim = {
        str(cid): split for cid, split in zip(claims["claim_id"], claims["split"], strict=True)
    }

    clusters: dict[str, list[str]] = defaultdict(list)
    for claim_id, canonical_id in canonical.items():
        clusters[canonical_id].append(claim_id)

    violations: list[LeakageViolation] = []
    for canonical_id, members in clusters.items():
        if len(members) < 2:
            continue
        splits = {split_by_claim[m] for m in members}
        if len(splits) > 1:
            violations.append(
                LeakageViolation(
                    canonical_identity_id=canonical_id,
                    splits=frozenset(splits),
                    claim_ids=tuple(sorted(members)),
                )
            )
    return violations


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, help="Path to a claims CSV.")
    parser.add_argument("--pin-threshold", type=float, default=85.0)
    args = parser.parse_args()

    claims = pd.read_csv(args.claims, dtype={"claim_id": str})
    violations = find_leakage_violations(claims, address_match_threshold=args.pin_threshold)

    if violations:
        print(
            f"ENTITY LEAKAGE DETECTED: {len(violations)} identity cluster(s) "
            "span multiple splits."
        )
        for v in violations:
            print(
                f"  identity={v.canonical_identity_id} splits={sorted(v.splits)} "
                f"claims={v.claim_ids}"
            )
        return 1

    n_claims = len(claims)
    print(f"OK: no entity leakage across splits ({n_claims} claims checked).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
