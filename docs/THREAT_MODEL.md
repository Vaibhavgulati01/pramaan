# Threat model

A repository that models attacks against its own defence reads
differently from one that does not. This covers both directions: attacks
that *use* PRAMAAN, and attacks *on* it.

---

## Attacks using PRAMAAN (evasion)

### The API is not a scoring oracle

An attacker's most valuable tool is a system that tells them how close
they are. `src/pramaan/api/` returns a tiered decision and templated
reason codes only — never raw probabilities, per-feature SHAP values,
gradients, distances, similarities, match counts, or the identity of a
matched prior claim.

Enforced in `src/pramaan/audit/reason_codes.py`: SHAP supplies the
attribution, a fixed template supplies the words, and the serialised
record rounds attributions to three decimals. Asserted by
`test_shap_is_rounded_in_the_serialised_record`.

### Evading the pillars

| Pillar | How an attacker evades it | Cost to them |
|---|---|---|
| Reuse graph (P3) | Never reuse an image | Must source a novel image per claim — which is the win, not a loss |
| Forensics (P2) | Submit a clean camera capture | Requires actually photographing something |
| Behaviour (P4) | Fresh identity per claim | Defeated by canonicalisation *if* details overlap; genuinely effective if they do not (see `docs/LIMITATIONS.md`) |
| Provenance (P1) | Strip C2PA | Trivial — and normal, which is why absence is `UNKNOWN`, never adverse |

The honest summary: **P3 is hard to evade without abandoning image
reuse, and P4 is easy to evade with disciplined operational security.**
We state this rather than implying uniform robustness.

### No generative capability

There is no image generator anywhere in the dependency tree — check
`pyproject.toml`. CLIP is used only for `encode_image`; there is no
decoder and no path from an embedding back to pixels. See
[`SAFETY.md`](../SAFETY.md).

---

## Attacks on PRAMAAN itself

The federated index is a shared, writable data structure. That makes it
an attack surface, and four rules defend it. **Each is measured, not
merely asserted** — the measurements are from
`tests/test_audit_federated_monitoring.py`.

### Rule 1 — First-seen immunity

*Attack:* I submit a rival's genuine photo **first**, so that when they
file their real claim it looks like reuse and gets denied.

*Rule:* the earliest identity on a cluster is never penalised by that
cluster.

*Measured:* `test_first_claimant_is_immune_to_their_own_cluster` — the
first claimant queries as `is_first_seen=True`, `actionable=False`, with
the suppression reason recorded on the result.

**But first-seen immunity alone does not stop this attack**, and it would
be dishonest to imply otherwise. If the *attacker* submits first, the
attacker gains the immunity and the victim is the one flagged. That is
why rule 2 exists.

### Rule 2 — k-independence

*Attack:* as above, or a merchant manufacturing a ring to damage a
competitor's customer.

*Rule:* a cluster contributes risk only once it holds **≥2 distinct
identity groups from ≥2 distinct merchants**.

*Measured:* `test_first_seen_immunity_blocks_the_poisoning_attack` runs
the full attack — attacker publishes the victim's photo first, victim
files a genuine claim — and the victim's query returns
`actionable=False` with `"k-independence not met"`. The same test file
shows the attack **succeeding** when the rule is relaxed to
`min_groups=1` (`test_attack_succeeds_when_k_independence_is_disabled`),
which is what makes the defence demonstrated rather than assumed.

`test_k_independence_requires_two_merchants_too` confirms a single
merchant cannot manufacture a ring alone.

### Rule 3 — Rate limits and an append-only signed log

*Attack:* flood the index to degrade it, or edit history to hide a
poisoning attempt.

*Rule:* per-merchant submission limits in a rolling window, plus a
hash-chained log where each entry commits to the previous one.

*Measured:* `test_rate_limit_rejects_a_flood` (20 submissions, 5
accepted at a limit of 5); `test_editing_the_audit_log_is_detected`
(mutating one entry breaks `verify_log()`).

The log does not *prevent* poisoning. It makes it **detectable after the
fact**, which is a weaker and more honest claim.

### Rule 4 — Decay

*Attack:* poison once, benefit indefinitely.

*Rule:* cluster evidence half-lives at 180 days; clusters below a weight
floor are pruned.

*Measured:* `test_decay_reduces_weight_over_time` (weight halves at
exactly 180 days); `test_expired_clusters_are_pruned`.

### What crosses the merchant boundary

Salted perceptual-hash bands (HMAC-SHA256 under a consortium-rotated
key) and differentially-private counts (Laplace, ε tracked and reported
per `stats()`). **Never** images, names, phone numbers, addresses, or
claimant identifiers — the index holds *identity group* ids, so it can
apply k-independence without learning who anyone is.

Asserted by `test_index_never_stores_raw_identifiers`.

The salt is load-bearing: raw pHash bands are invertible enough to be a
privacy problem across a consortium, and rotating the key means a
departing participant cannot keep querying yesterday's index.

---

## Attacks on the guarantee itself

### Certificate expiry under shift

*Attack:* wait. Shift the distribution — a new generator, a new
compression pipeline — and the certificate silently stops holding while
the system keeps auto-denying under a bound that no longer applies.

*Defence:* `monitoring/drift.py`'s `GuaranteeWatchdog` recomputes the
realised false-denial rate on each matured label batch and **widens the
abstention band automatically** when it crosses α, rather than merely
alerting. An alert leaves the unbounded behaviour running while it waits
for a human.

*Measured:* `test_watchdog_widens_the_band_on_breach`.

### Calibration-split reuse

*Attack:* tune against the calibration split, then certify on it. The
certificate becomes a statement about data already optimised against.

*Defence:* `FusionModel.fit` raises on calibration or test rows, and the
split's content hash is committed and re-checked in CI. See
[`GUARANTEE.md`](GUARANTEE.md) caveat 3.

---

## Regulatory framing

India's DPDP Rules were notified **13 Nov 2025**; the Consent Manager
framework goes live **13 Nov 2026**; substantive compliance lands
**13 May 2027**. A hash-only index with DP-noised counts is designed
against that deadline rather than retrofitted to it — there is no
personal data in the shared structure to consent to the sharing of.

There is domestic precedent for collaborative fraud intelligence at
national scale: RBIH's **MuleHunter.AI** is onboarded across 29 banks,
and **DPIP** pilots ecosystem-wide model training. The merchant-side
analogue is a familiar shape rather than a novel legal argument.

---

## What we have not defended against

Stated because a threat model that lists only solved problems is
marketing:

- **A determined attacker with disciplined opsec.** Fresh phone, fresh
  email, fresh address, novel image per claim defeats every pillar. The
  system raises the cost of fraud; it does not make it impossible.
- **A malicious consortium insider.** HMAC signing is symmetric, so a
  party holding the key can forge log entries. It protects against
  tampering and accident, not against an insider with the key.
- **Collusion across the k-independence bar.** Two genuinely independent
  attackers across two merchants clear rule 2. The bar raises
  coordination cost; it does not eliminate the attack.
- **Adversarial perturbation of the reuse graph.** We have not tested
  images crafted specifically to sit just outside the pHash and CLIP
  thresholds. That is a real gap and a natural next piece of work.
