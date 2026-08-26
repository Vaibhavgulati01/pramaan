"""Cross-signal entity resolution: cluster claims into canonical
claimant identities from their canonicalised phone/email/address
signals (PRAMAAN_v2_architecture.md Sec.4 L0).

This is THE module `eval/entity_leakage_audit.py` depends on: if the
same human appears as two "different" claimants across a train/test
boundary, the split is contaminated and any certificate computed on it
is fiction. Device fingerprints are deliberately excluded from hard
identity linking here - see the instability caveat in `ingest/device.py`
- they're a behavioural feature (Phase 2), not an identity signal.

Union-find over three edge types: exact canonical-phone match, exact
canonical-email match, and PIN-bucketed fuzzy-address match. Output is
deterministic regardless of input order (canonical id = "entity_" +
min claim_id in the resolved cluster), which matters once this feeds a
determinism test (Phase 6).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pramaan.common.union_find import UnionFind
from pramaan.ingest.address import addresses_match, canonicalize_address, canonicalize_pin
from pramaan.ingest.email import canonicalize_email
from pramaan.ingest.phone import canonicalize_phone


@dataclass(frozen=True)
class ClaimIdentitySignals:
    claim_id: str
    phone_raw: str | None = None
    email_raw: str | None = None
    address_raw: str | None = None
    pin_raw: str | None = None


def resolve_canonical_identities(
    claims: list[ClaimIdentitySignals],
    address_match_threshold: float = 85.0,
) -> dict[str, str]:
    """Returns {claim_id: canonical_identity_id}. Every input claim_id is
    a key in the output, even ones that matched no other claim (they
    form a singleton cluster of their own)."""
    claim_ids = [c.claim_id for c in claims]
    uf = UnionFind(claim_ids)

    by_phone: dict[str, list[str]] = defaultdict(list)
    by_email: dict[str, list[str]] = defaultdict(list)
    by_pin: dict[str, list[str]] = defaultdict(list)
    norm_addr_by_id: dict[str, str] = {}
    pin_by_id: dict[str, str | None] = {}

    for c in claims:
        phone = canonicalize_phone(c.phone_raw)
        email = canonicalize_email(c.email_raw)
        addr_norm = canonicalize_address(c.address_raw)
        pin = canonicalize_pin(c.pin_raw)

        norm_addr_by_id[c.claim_id] = addr_norm
        pin_by_id[c.claim_id] = pin

        if phone:
            by_phone[phone].append(c.claim_id)
        if email:
            by_email[email].append(c.claim_id)
        if pin and addr_norm:
            by_pin[pin].append(c.claim_id)

    for group in by_phone.values():
        for other in group[1:]:
            uf.union(group[0], other)
    for group in by_email.values():
        for other in group[1:]:
            uf.union(group[0], other)

    for ids in by_pin.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if addresses_match(
                    norm_addr_by_id[a],
                    pin_by_id[a],
                    norm_addr_by_id[b],
                    pin_by_id[b],
                    address_match_threshold,
                ):
                    uf.union(a, b)

    result: dict[str, str] = {}
    for members in uf.clusters().values():
        canonical_id = f"entity_{min(members)}"
        for cid in members:
            result[cid] = canonical_id
    return result
