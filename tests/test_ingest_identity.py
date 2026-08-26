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
