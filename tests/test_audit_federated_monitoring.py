"""Audit trail, federated index, and monitoring.

The federation half includes the **poisoning attack** the kill-gate in
`docs/PREREGISTRATION.md` requires: first-seen immunity must block it,
and the same attack must succeed with the rule disabled. A defence that
is never shown failing without its guard has not been demonstrated.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pramaan.audit.evidence_pack import (
    EvidenceRecord,
    SignedEvidencePack,
    sign_record,
    verify_record,
)
from pramaan.audit.reason_codes import REASON_TEMPLATES, ReasonCode, render_reason_codes
from pramaan.federated.cuckoo_filter import CuckooFilter, CuckooFilterFull
from pramaan.federated.index import FederatedIndex, salted_band
from pramaan.monitoring.drift import (
    GuaranteeWatchdog,
    compute_drift,
    population_stability_index,
)

KEY = b"consortium-key-for-tests-only"
BASE = datetime(2026, 1, 1)


# --- reason codes -----------------------------------------------------


def test_renders_a_template_not_free_text() -> None:
    codes = render_reason_codes(
        {"reuse_n_distinct_claimants_sharing": 0.4},
        {"n_matches": 2, "n_distinct_claimants_sharing": 2, "days_since_first_seen": 6},
    )
    assert codes[0].code == "REUSE_MULTI_CLAIMANT"
    assert "2 distinct claimant(s)" in codes[0].text


def test_is_deterministic() -> None:
    """An adverse-action decision must be re-derivable byte for byte
    months later - which a stochastic generator cannot promise."""
    shap = {"reuse_n_distinct_claimants_sharing": 0.4}
    values = {"n_matches": 2, "n_distinct_claimants_sharing": 2, "days_since_first_seen": 6}
    assert render_reason_codes(shap, values)[0].text == render_reason_codes(shap, values)[0].text


def test_only_risk_increasing_factors_become_reasons() -> None:
    """This explains why a claim was flagged. Mixing exculpatory factors
    into an adverse-action list muddles what the claimant is told."""
    codes = render_reason_codes(
        {"provenance_declares_camera_capture": -0.5, "reuse_matched_prior_claim": 0.3},
        {"reuse_min_hamming": 2},
    )
    assert all(c.shap > 0 for c in codes)
    assert "PROVENANCE_CAMERA_CAPTURE" not in {c.code for c in codes}


def test_negligible_attributions_are_dropped() -> None:
    assert render_reason_codes({"reuse_matched_prior_claim": 0.0001}, {}) == []


def test_features_without_a_template_never_become_reasons() -> None:
    """A feature can influence the score without being quotable - a
    reason a claimant cannot act on is not a reason."""
    codes = render_reason_codes({"forensics_fft_peak_ratio": 0.9}, {})
    assert codes == []


def test_top_k_is_respected_and_ordered_by_attribution() -> None:
    codes = render_reason_codes(
        {
            "reuse_n_distinct_claimants_sharing": 0.5,
            "ring_distinct_claimants": 0.4,
            "behaviour_n_prior_claims_30d": 0.3,
            "forensics_thumbnail_mismatch": 0.2,
        },
        {
            "n_matches": 1, "n_distinct_claimants_sharing": 2, "days_since_first_seen": 3,
            "ring_distinct_claimants": 2, "ring_distinct_merchants": 2,
            "behaviour_n_prior_claims_30d": 4, "forensics_thumbnail_mismatch": 0.3,
        },
        top_k=2,
    )
    assert len(codes) == 2
    assert codes[0].shap >= codes[1].shap


def test_missing_template_value_degrades_rather_than_raising() -> None:
    """A decision must remain explainable even if a template drifts."""
    codes = render_reason_codes({"reuse_n_distinct_claimants_sharing": 0.5}, {})
    assert codes and codes[0].code == "REUSE_MULTI_CLAIMANT"


def test_shap_is_rounded_in_the_serialised_record() -> None:
    """Full-precision attributions edge toward the gradient oracle
    SAFETY.md rules out."""
    as_dict = ReasonCode("X", "text", 0.123456789, "f").as_dict()
    assert as_dict["shap"] == 0.123


def test_every_template_is_wellformed() -> None:
    for feature, (code, template) in REASON_TEMPLATES.items():
        assert code.isupper(), feature
        assert len(template) > 20, feature


# --- evidence pack ----------------------------------------------------


def _record() -> EvidenceRecord:
    return EvidenceRecord(
        claim_id="claim_000123",
        decision="DENY",
        p_hat=0.9412345,
        reason_codes=[ReasonCode("REUSE_MULTI_CLAIMANT", "matched 2 prior claims", 0.41, "f")],
        certified=True,
        certified_alpha=0.03,
        certified_delta=0.10,
        cascade_stage_exited=3,
        compute_ms=118.4,
        model_sha="abc123",
        policy_sha="def456",
        feature_schema_version="1.1.0",
        image_sha256="f" * 64,
    )


def test_signature_verifies() -> None:
    record = _record()
    assert verify_record(record, sign_record(record, KEY), KEY)


def test_tampering_invalidates_the_signature() -> None:
    record = _record()
    signature = sign_record(record, KEY)
    record.decision = "APPROVE"
    assert not verify_record(record, signature, KEY)


def test_a_different_key_does_not_verify() -> None:
    record = _record()
    assert not verify_record(record, sign_record(record, KEY), b"other-key")


def test_canonical_json_is_order_independent() -> None:
    """The signature must not depend on incidental dict ordering."""
    assert _record().canonical_json() == _record().canonical_json()


def test_absent_provenance_is_labelled_not_adverse() -> None:
    payload = _record().to_payload()
    assert "not adverse" in payload["provenance"]["c2pa"]


def test_record_carries_the_certificate_parameters() -> None:
    payload = _record().to_payload()
    assert payload["certified"] and payload["certified_alpha"] == 0.03


def test_p_hat_is_rounded() -> None:
    assert _record().to_payload()["p_hat"] == 0.9412


def test_signed_pack_roundtrip() -> None:
    pack = SignedEvidencePack.create(_record(), KEY)
    assert pack.verify(KEY)
    assert "signature" in pack.as_dict()


# --- cuckoo filter ----------------------------------------------------


def test_no_false_negatives() -> None:
    """The correctness property the federated index depends on: a false
    negative would make a reused image invisible to the consortium."""
    cuckoo = CuckooFilter(capacity=1024)
    items = [f"item-{i}".encode() for i in range(500)]
    for item in items:
        cuckoo.insert(item)
    assert all(item in cuckoo for item in items)


def test_false_positive_rate_is_low() -> None:
    cuckoo = CuckooFilter(capacity=1024)
    for i in range(400):
        cuckoo.insert(f"present-{i}".encode())
    absent = [f"absent-{i}".encode() for i in range(2000)]
    rate = sum(1 for item in absent if item in cuckoo) / len(absent)
    assert rate < 0.01


def test_deletion_supports_the_decay_rule() -> None:
    """Cuckoo over Bloom precisely because evidence must age out."""
    cuckoo = CuckooFilter(capacity=256)
    item = b"decaying"
    cuckoo.insert(item)
    assert item in cuckoo
    assert cuckoo.delete(item)
    assert item not in cuckoo


def test_deleting_an_absent_item_returns_false() -> None:
    assert not CuckooFilter(capacity=256).delete(b"never-inserted")


def test_overfull_filter_raises_rather_than_dropping() -> None:
    """A silently dropped insert becomes exactly the false negative this
    structure promises never to have."""
    cuckoo = CuckooFilter(capacity=16, bucket_size=2, max_kicks=20)
    with pytest.raises(CuckooFilterFull):
        for i in range(500):
            cuckoo.insert(f"item-{i}".encode())


def test_capacity_must_be_a_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        CuckooFilter(capacity=1000)


# --- federated index --------------------------------------------------


def test_salted_bands_hide_the_underlying_value() -> None:
    band = salted_band(3, 7, KEY)
    assert band != salted_band(7, 3, KEY)  # index and value both covered
    assert band != salted_band(3, 7, b"different-key")


def test_publish_then_query_finds_the_cluster() -> None:
    index = FederatedIndex(KEY)
    phash = 0xABCD_1234_5678_9F01
    index.publish(phash, "group_a", "merchant_1", BASE)
    result = index.query(phash, "group_b", BASE + timedelta(days=1))
    assert result.in_index


def test_first_seen_immunity_blocks_the_poisoning_attack() -> None:
    """THE attack the kill-gate requires demonstrating.

    An attacker submits a rival's genuine photo FIRST, hoping the rival's
    later legitimate claim is flagged as reuse. First-seen immunity means
    the earliest claimant on a cluster is never penalised by it - but
    here the attacker is first, so it is the ATTACKER who gains immunity
    and the victim who would be flagged.

    That is why the rule alone is not enough, and why k-independence
    (rule 2) must also hold: a single attacker cannot manufacture the two
    independent identity groups an actionable cluster requires.
    """
    index = FederatedIndex(KEY)
    phash = 0x1234_5678_9ABC_DEF0

    # Attacker publishes the victim's photo first.
    index.publish(phash, "attacker_group", "merchant_attacker", BASE)

    # Victim submits their own genuine claim later.
    victim = index.query(phash, "victim_group", BASE + timedelta(days=1))

    # The cluster has only ONE identity group so far, so k-independence
    # is not met and no risk is contributed. The attack fails.
    assert not victim.actionable
    assert "k-independence not met" in (victim.suppression_reason or "")


def test_attack_succeeds_when_k_independence_is_disabled() -> None:
    """The same attack, with the guard removed - which is what makes the
    guard's effect demonstrated rather than asserted."""
    index = FederatedIndex(KEY)
    phash = 0x1234_5678_9ABC_DEF0
    index.publish(phash, "attacker_group", "merchant_attacker", BASE)

    cluster = next(iter(index._clusters.values()))  # noqa: SLF001 - white-box by design
    # Without k-independence, a single-group cluster would be actionable.
    assert not cluster.is_actionable(min_groups=2)
    assert cluster.is_actionable(min_groups=1)


def test_first_claimant_is_immune_to_their_own_cluster() -> None:
    index = FederatedIndex(KEY)
    phash = 0xAAAA_BBBB_CCCC_DDDD
    index.publish(phash, "first_group", "merchant_1", BASE)
    index.publish(phash, "second_group", "merchant_2", BASE + timedelta(days=1))

    first = index.query(phash, "first_group", BASE + timedelta(days=2))
    assert first.is_first_seen
    assert not first.actionable
    assert "first-seen immunity" in (first.suppression_reason or "")

    # A later, independent claimant on the same cluster IS actionable.
    later = index.query(phash, "third_group", BASE + timedelta(days=3))
    assert later.actionable


def test_k_independence_requires_two_merchants_too() -> None:
    """One merchant cannot manufacture a ring alone."""
    index = FederatedIndex(KEY)
    phash = 0xFEED_FACE_CAFE_BEEF
    index.publish(phash, "group_a", "merchant_1", BASE)
    index.publish(phash, "group_b", "merchant_1", BASE + timedelta(days=1))

    result = index.query(phash, "group_c", BASE + timedelta(days=2))
    assert not result.actionable


def test_rate_limit_rejects_a_flood() -> None:
    index = FederatedIndex(KEY, max_submissions_per_window=5)
    accepted = sum(
        index.publish(0x1000 + i, f"group_{i}", "spammer", BASE + timedelta(minutes=i))
        for i in range(20)
    )
    assert accepted == 5


def test_dp_count_is_noisy_but_non_negative() -> None:
    index = FederatedIndex(KEY, dp_epsilon=1.0)
    phash = 0x9999_8888_7777_6666
    index.publish(phash, "group_a", "merchant_1", BASE)
    counts = [index.query(phash, "group_x", BASE).dp_count for _ in range(30)]
    assert all(c >= 0 for c in counts)
    assert len(set(counts)) > 1  # genuinely noised, not a constant


def test_epsilon_budget_is_tracked() -> None:
    """Only QUERIES spend privacy budget - publishing adds to the index
    without releasing a noised count, so it costs nothing."""
    index = FederatedIndex(KEY, dp_epsilon=0.5)
    phash = 0x1111_2222_3333_4444
    index.publish(phash, "group_a", "merchant_1", BASE)
    assert index.stats()["epsilon_spent"] == 0.0

    for _ in range(4):
        index.query(phash, "group_x", BASE)
    assert index.stats()["epsilon_spent"] == pytest.approx(4 * 0.5)


def test_decay_reduces_weight_over_time() -> None:
    index = FederatedIndex(KEY)
    phash = 0xDEAD_BEEF_1234_5678
    index.publish(phash, "group_a", "merchant_1", BASE)

    fresh = index.query(phash, "group_x", BASE).decayed_weight
    aged = index.query(phash, "group_x", BASE + timedelta(days=180)).decayed_weight
    assert aged == pytest.approx(fresh * 0.5, rel=0.05)


def test_expired_clusters_are_pruned() -> None:
    index = FederatedIndex(KEY)
    index.publish(0xABCD, "group_a", "merchant_1", BASE)
    assert index.stats()["n_clusters"] == 1
    assert index.prune_expired(BASE + timedelta(days=365 * 4)) == 1
    assert index.stats()["n_clusters"] == 0


def test_audit_log_is_hash_chained() -> None:
    index = FederatedIndex(KEY)
    for i in range(5):
        index.publish(0x100 + i, f"group_{i}", "merchant_1", BASE + timedelta(hours=i))
    assert index.verify_log()


def test_editing_the_audit_log_is_detected() -> None:
    """Rule 3: poisoning attempts are detectable after the fact even if
    they were not prevented in the moment."""
    index = FederatedIndex(KEY)
    for i in range(5):
        index.publish(0x200 + i, f"group_{i}", "merchant_1", BASE + timedelta(hours=i))

    index._audit_log[2]["event"] = "tampered"  # noqa: SLF001 - white-box by design
    assert not index.verify_log()


def test_index_never_stores_raw_identifiers() -> None:
    """Hash-only: never images, names, phone numbers or addresses."""
    index = FederatedIndex(KEY)
    index.publish(0xABCD_1234, "group_a", "merchant_1", BASE)
    blob = repr(index._clusters) + repr(index._audit_log)  # noqa: SLF001
    assert "0xABCD_1234" not in blob


# --- monitoring -------------------------------------------------------


def test_psi_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5000)
    assert population_stability_index(sample, rng.normal(0, 1, 5000)) < 0.05


def test_psi_detects_a_shift() -> None:
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)) > 0.25


def test_psi_bins_on_the_reference_not_the_pooled_sample() -> None:
    """Binning on pooled data would let the current batch move the bins
    and hide the very shift PSI exists to detect."""
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 4000)
    shifted = rng.normal(3, 1, 4000)
    assert population_stability_index(reference, shifted) > population_stability_index(
        reference, reference
    )


def test_psi_handles_a_constant_feature() -> None:
    constant = np.ones(500)
    assert population_stability_index(constant, constant) == 0.0


def test_drift_report_ranks_the_worst_feature() -> None:
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "drifting": rng.normal(0, 1, 2000)})
    current = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "drifting": rng.normal(3, 1, 2000)})
    report = compute_drift(reference, current)
    assert report.worst(1)[0].feature == "drifting"
    assert report.n_significant >= 1


def test_watchdog_stays_quiet_within_the_bound() -> None:
    watchdog = GuaranteeWatchdog(alpha=0.10, initial_t_deny=0.9)
    probabilities = np.full(100, 0.95)
    labels = np.ones(100, dtype=int)  # every denial correct
    observation = watchdog.observe_batch(probabilities, labels, BASE)
    assert not observation.breached
    assert watchdog.t_deny == 0.9


def test_watchdog_widens_the_band_on_breach() -> None:
    """The consequence of Sec.4 L3 almost nobody builds: when the
    realised rate crosses alpha the certificate has expired, and the
    system stops the unbounded behaviour rather than merely alerting."""
    watchdog = GuaranteeWatchdog(alpha=0.05, initial_t_deny=0.90, widen_step=0.02)
    probabilities = np.full(100, 0.95)
    labels = np.array([0] * 30 + [1] * 70)  # 30% of denials are legitimate

    observation = watchdog.observe_batch(probabilities, labels, BASE)
    assert observation.breached
    assert "CERTIFICATE EXPIRED" in observation.action
    assert watchdog.t_deny == pytest.approx(0.92)
    assert watchdog.total_widening == pytest.approx(0.02)


def test_watchdog_reports_uninformative_batches_rather_than_skipping() -> None:
    """A long run of tiny batches must be visible, not look like a clean
    bill of health."""
    watchdog = GuaranteeWatchdog(alpha=0.05, initial_t_deny=0.9, min_batch=30)
    observation = watchdog.observe_batch(np.full(5, 0.95), np.zeros(5, dtype=int), BASE)
    assert not observation.breached
    assert "below the 30" in observation.action
    assert np.isnan(observation.realised_fdr)


def test_watchdog_accumulates_widening_across_batches() -> None:
    watchdog = GuaranteeWatchdog(alpha=0.05, initial_t_deny=0.90, widen_step=0.02)
    probabilities = np.full(100, 0.99)
    labels = np.array([0] * 40 + [1] * 60)
    for day in range(3):
        watchdog.observe_batch(probabilities, labels, BASE + timedelta(days=day))
    assert watchdog.total_widening == pytest.approx(0.06)
    assert watchdog.has_breached
