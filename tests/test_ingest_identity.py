import random

import pytest

from pramaan.ingest.identity import ClaimIdentitySignals, resolve_canonical_identities


def test_same_phone_clusters_together() -> None:
    claims = [
        ClaimIdentitySignals("c1", phone_raw="9876543210"),
        ClaimIdentitySignals("c2", phone_raw="+91 98765 43210"),
        ClaimIdentitySignals("c3", phone_raw="9111111111"),
    ]
    result = resolve_canonical_identities(claims)
    assert result["c1"] == result["c2"]
    assert result["c1"] != result["c3"]


def test_same_gmail_family_email_clusters_together() -> None:
    claims = [
        ClaimIdentitySignals("c1", email_raw="a.b@gmail.com"),
        ClaimIdentitySignals("c2", email_raw="ab+refund@gmail.com"),
        ClaimIdentitySignals("c3", email_raw="unrelated@yahoo.com"),
    ]
    result = resolve_canonical_identities(claims)
    assert result["c1"] == result["c2"]
    assert result["c1"] != result["c3"]


def test_same_pin_similar_address_clusters_together() -> None:
    claims = [
        ClaimIdentitySignals("c1", address_raw="Flat 12, Opp City Mall", pin_raw="400001"),
        ClaimIdentitySignals(
            "c2", address_raw="flat no 12 opposite city mall", pin_raw="400001"
        ),
        ClaimIdentitySignals("c3", address_raw="Shop 45 Industrial Estate", pin_raw="400001"),
    ]
    result = resolve_canonical_identities(claims)
    assert result["c1"] == result["c2"]
    assert result["c1"] != result["c3"]


def test_different_pin_never_clusters_even_if_address_text_identical() -> None:
    claims = [
        ClaimIdentitySignals("c1", address_raw="Flat 12 City Mall", pin_raw="400001"),
        ClaimIdentitySignals("c2", address_raw="Flat 12 City Mall", pin_raw="560001"),
    ]
    result = resolve_canonical_identities(claims)
    assert result["c1"] != result["c2"]


def test_transitive_clustering_across_different_signals() -> None:
    # c1-c2 share a phone; c2-c3 share an email. c1 and c3 share nothing
    # directly, but must end up in the same cluster (same human, two
    # different contact details reused across the pair of overlaps).
    claims = [
        ClaimIdentitySignals("c1", phone_raw="9876543210", email_raw="c1@yahoo.com"),
        ClaimIdentitySignals(
            "c2", phone_raw="9876543210", email_raw="shared@gmail.com"
        ),
        ClaimIdentitySignals("c3", phone_raw="9111111111", email_raw="shared@gmail.com"),
    ]
    result = resolve_canonical_identities(claims)
    assert result["c1"] == result["c2"] == result["c3"]


def test_no_signals_forms_singleton_and_does_not_crash() -> None:
    claims = [
        ClaimIdentitySignals("c1"),
        ClaimIdentitySignals("c2"),
    ]
    result = resolve_canonical_identities(claims)
    assert result["c1"] != result["c2"]


def test_deterministic_across_runs_and_input_order() -> None:
    claims = [
        ClaimIdentitySignals("c2", phone_raw="9876543210"),
        ClaimIdentitySignals("c1", phone_raw="9876543210"),
        ClaimIdentitySignals("c3", email_raw="x@yahoo.com"),
    ]
    result_a = resolve_canonical_identities(claims)
    result_b = resolve_canonical_identities(list(reversed(claims)))
    assert result_a["c1"] == result_a["c2"]
    # Canonical id is derived from min(member ids), independent of
    # processing order, so both orderings agree on every claim's id.
    assert result_a == result_b


def test_every_input_claim_id_is_a_key_in_output() -> None:
    claims = [ClaimIdentitySignals(f"c{i}") for i in range(5)]
    result = resolve_canonical_identities(claims)
    assert set(result.keys()) == {c.claim_id for c in claims}


# ---------------------------------------------------------------------
# Blocking equivalence.
#
# `_address_candidate_pairs` skips pairs inside a PIN bucket that cannot
# match, to avoid comparing every pair. That is only safe if it skips
# exactly the pairs `addresses_match` would have rejected anyway, so the
# property under test is equality of *output*, not of pair count.
#
# Why it matters: the simulator draws from 8 distinct PIN codes at any
# corpus size, so bucket size grows linearly with claims and comparisons
# grow as N^2. Brute force measured ~64 s at 12,000 claims, extrapolating
# to ~13 minutes at the full tier's 35,000. With blocking it is ~5 s, and
# the clusters are identical.
# ---------------------------------------------------------------------


def _brute_force_address_pairs(ids: list[str]) -> list[tuple[str, str]]:
    return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]


def _resolve_with_all_pairs(
    claims: list[ClaimIdentitySignals], threshold: float = 85.0
) -> dict[str, str]:
    """Reference implementation: compare every pair in every PIN bucket."""
    from collections import defaultdict

    from pramaan.common.union_find import UnionFind
    from pramaan.ingest.address import (
        addresses_match,
        canonicalize_address,
        canonicalize_pin,
    )
    from pramaan.ingest.email import canonicalize_email
    from pramaan.ingest.phone import canonicalize_phone

    uf = UnionFind([c.claim_id for c in claims])
    by_phone: dict[str, list[str]] = defaultdict(list)
    by_email: dict[str, list[str]] = defaultdict(list)
    by_pin: dict[str, list[str]] = defaultdict(list)
    addr: dict[str, str] = {}
    pins: dict[str, str | None] = {}

    for c in claims:
        phone = canonicalize_phone(c.phone_raw)
        email = canonicalize_email(c.email_raw)
        addr[c.claim_id] = canonicalize_address(c.address_raw)
        pins[c.claim_id] = canonicalize_pin(c.pin_raw)
        if phone:
            by_phone[phone].append(c.claim_id)
        if email:
            by_email[email].append(c.claim_id)
        if pins[c.claim_id] and addr[c.claim_id]:
            by_pin[pins[c.claim_id]].append(c.claim_id)

    for group in list(by_phone.values()) + list(by_email.values()):
        for other in group[1:]:
            uf.union(group[0], other)
    for ids in by_pin.values():
        for a, b in _brute_force_address_pairs(ids):
            if addresses_match(addr[a], pins[a], addr[b], pins[b], threshold):
                uf.union(a, b)

    out: dict[str, str] = {}
    for members in uf.clusters().values():
        canonical = f"entity_{min(members)}"
        for member in members:
            out[member] = canonical
    return out


def _signals(n: int, seed: int) -> list[ClaimIdentitySignals]:
    rng = random.Random(seed)
    streets = ["Chanda Marg", "Garg Marg", "Tata Marg", "MG Road", "Sarraf Lane"]
    out = []
    for i in range(n):
        # A deliberately collision-heavy population: few PINs, few streets,
        # a narrow house-number range, and some addresses with no number at
        # all -- the case blocking must not skip.
        numbered = rng.random() > 0.15
        house = f"H.No. {rng.randint(1, 40)} " if numbered else ""
        out.append(
            ClaimIdentitySignals(
                claim_id=f"c{i:04d}",
                phone_raw=f"+9198765{rng.randint(0, 60):05d}",
                email_raw=f"user{rng.randint(0, 80)}@example.com",
                address_raw=f"{house}{rng.choice(streets)}",
                pin_raw=rng.choice(["110001", "400001", "560001"]),
            )
        )
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_blocking_gives_identical_clusters_to_comparing_every_pair(seed: int) -> None:
    claims = _signals(160, seed)
    assert resolve_canonical_identities(claims) == _resolve_with_all_pairs(claims)


def test_blocking_holds_for_addresses_with_no_house_number() -> None:
    """Unnumbered addresses are exempt from the numeric-token rule.

    If blocking treated them like numbered ones they would be dropped
    from every candidate pair, and genuine merges would be lost.
    """
    claims = [
        ClaimIdentitySignals(claim_id="a", address_raw="Nehru Nagar Colony", pin_raw="110001"),
        ClaimIdentitySignals(
            claim_id="b", address_raw="Nehru Nagar Colony West", pin_raw="110001"
        ),
        ClaimIdentitySignals(
            claim_id="c", address_raw="H.No. 9 Nehru Nagar Colony", pin_raw="110001"
        ),
    ]
    resolved = resolve_canonical_identities(claims)
    assert resolved == _resolve_with_all_pairs(claims)
    # a and b carry no number, so text similarity alone decides, and
    # blocking must not have removed the pair from consideration.
    assert resolved["a"] == resolved["b"]
    # c is numbered but a is not, so rule 3 is skipped for that pair too.
    assert resolved["a"] == resolved["c"]


def test_blocking_still_separates_different_house_numbers() -> None:
    """The false-merge case the numeric rule exists for, post-blocking."""
    claims = [
        ClaimIdentitySignals(claim_id="a", address_raw="H.No. 97 Chanda Marg", pin_raw="121001"),
        ClaimIdentitySignals(claim_id="b", address_raw="H.No. 51 Garg Marg", pin_raw="121001"),
    ]
    resolved = resolve_canonical_identities(claims)
    assert resolved["a"] != resolved["b"]


def test_pairs_sharing_two_numbers_are_yielded_once() -> None:
    from pramaan.ingest.identity import _address_candidate_pairs

    addrs = {"a": "flat 3 block 7 mg road", "b": "flat 3 block 7 mg road"}
    pairs = list(_address_candidate_pairs(["a", "b"], addrs))
    assert pairs == [("a", "b")], pairs
