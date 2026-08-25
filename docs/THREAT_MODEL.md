# Threat model

> **Status: skeleton.** Finalised in Phase 7, subject to the federation
> kill-gate (see the implementation plan): the federated index section
> below is either backed by measured results (on/off ablation, a real
> poisoning-attack self-test) or shrinks to a documented-design-only
> section if those measurements aren't clean within two implementation
> attempts.

## What will go here

### Attacks *using* PRAMAAN (evasion)
- Why the API returns tiered decisions + templated reason codes only,
  never raw scores/gradients/SHAP values directly (no black-box evasion
  oracle).
- Why no generative model exists anywhere in the dependency tree
  (`SAFETY.md`).

### Attacks *on* PRAMAAN itself (the federated index is itself an attack
surface — §4 L6)
1. **First-seen immunity** — the earliest claimant on an image cluster is
   never penalised by that cluster.
2. **k-independence** — a cluster only contributes risk once it has ≥2
   claimants from ≥2 merchant-independent identity groups.
3. **Rate limits + append-only signed log** — poisoning attempts
   detectable after the fact.
4. **Decay** — cluster evidence half-lives at 180 days.

Each rule gets a measured result: a first-seen-poisoning attack run
against the index, reported as blocked-with-the-rule / succeeds-without-it.

### Regulatory framing
DPDP Rules timeline (notified 13 Nov 2025; Consent Manager framework live
13 Nov 2026; substantive compliance 13 May 2027), and the RBIH
MuleHunter.AI / DPIP precedent for collaborative fraud intelligence.
