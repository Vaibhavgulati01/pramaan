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
from collections.abc import Iterator
from dataclasses import dataclass

from pramaan.common.union_find import UnionFind
from pramaan.ingest.address import (
    addresses_match,
    canonicalize_address,
    canonicalize_pin,
    extract_numeric_tokens,
)
from pramaan.ingest.email import canonicalize_email
from pramaan.ingest.phone import canonicalize_phone


@dataclass(frozen=True)
class ClaimIdentitySignals:
    claim_id: str
    phone_raw: str | None = None
    email_raw: str | None = None
    address_raw: str | None = None
    pin_raw: str | None = None


def _address_candidate_pairs(
    ids: list[str], norm_addr_by_id: dict[str, str]
) -> Iterator[tuple[str, str]]:
    """Pairs within one PIN bucket that could possibly match.

    Comparing every pair in a bucket is quadratic, and the PIN bucket is
    not a small unit: the simulated ledger draws from **8 distinct PIN
    codes** regardless of corpus size, so bucket size grows linearly with
    claim count and pair count grows as N^2. Measured on the simulator:

        claims    pairs        seconds
         1,500      141,683       0.62
         3,000      562,063       4.09
         6,000    2,250,092      16.81
        12,000    8,997,231      69.90

    A fitted exponent of 2.27 puts the `full` tier's 35,000 claims at
    ~13 minutes of fuzzy string matching, all of it on the critical path
    of the corpus build.

    This is a *blocking* step, and it is exact rather than heuristic:
    `addresses_match` rejects any pair where both sides carry numeric
    tokens that do not intersect (rule 3 in its docstring), so two
    numbered addresses sharing no number cannot match, and skipping them
    changes no result. Claims with no numeric token at all are exempt
    from that rule, so they are still compared against everything in the
    bucket.

    Equivalence to the brute-force version is asserted directly, over
    generated inputs, in `tests/test_ingest_identity.py`.
    """
    numeric = {cid: extract_numeric_tokens(norm_addr_by_id[cid]) for cid in ids}
    numbered = [cid for cid in ids if numeric[cid]]
    unnumbered = [cid for cid in ids if not numeric[cid]]

    # Rule 3 never fires for these, so they keep their full comparison set.
    for i, a in enumerate(unnumbered):
        for b in unnumbered[i + 1 :]:
            yield a, b
        for b in numbered:
            yield a, b

    # Both sides numbered: only a shared number can survive rule 3.
    by_token: dict[str, list[str]] = defaultdict(list)
    for cid in numbered:
        for token in numeric[cid]:
            by_token[token].append(cid)

    seen: set[tuple[str, str]] = set()
    for bucket in by_token.values():
        for i, a in enumerate(bucket):
            for b in bucket[i + 1 :]:
                # A pair sharing two numbers would otherwise be compared twice.
                pair = (a, b) if a <= b else (b, a)
                if pair not in seen:
                    seen.add(pair)
                    yield pair


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
        for a, b in _address_candidate_pairs(ids, norm_addr_by_id):
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
