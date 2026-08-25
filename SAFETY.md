# Safety

> **Status: placeholder.** Written properly in Phase 2, once the evidence
> pillars exist and this document can point at the actual dependency tree
> (`pyproject.toml`) to verify the claims below rather than assert them.

PRAMAAN is defense-only by construction. This document will cover, with
verifiable evidence for each claim:

- No generative model in the repository or its dependencies.
- Every adversarial/synthetic image example is drawn from pre-existing
  public academic benchmarks (GenImage, AI-GenBench) — no fake-damage
  generator was built, including for evaluation.
- Robustness/shift testing perturbs only our own held-out inputs to
  measure our own degradation (§6 shift matrix), not third-party systems.
- The API (`src/pramaan/api/`) returns a tiered decision and reason codes
  only — never raw scores or gradients — so it cannot be used as a
  black-box evasion oracle.
- Auto-deny requires corroboration beyond the image pillars alone; absent
  provenance (C2PA) is encoded as `UNKNOWN`, never as adverse.
- The federated index (`src/pramaan/federated/`) stores salted
  perceptual-hash bands and DP-noised counts only — never images, names,
  phone numbers, or addresses.

See `docs/THREAT_MODEL.md` for the full analysis, including attacks
against PRAMAAN itself (the federated index's anti-poisoning rules).
