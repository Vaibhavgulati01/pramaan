# Data card

> **Status: skeleton.** Filled in as `benchmarks/build_bench.py` lands in
> Phase 1 (sources actually used, manifest schema) and finalised in
> Phase 9 with licence/attribution text.

## What will go here

- **Source table**: GenImage, AI-GenBench, Amazon Berkeley Objects (CC
  BY-NC 4.0 — **non-commercial**, attribution required; this repo does not
  redistribute ABO images, only samples them at build time and records
  provenance/SHA256 in the manifest), IEEE-CIS Fraud Detection (optional,
  behavioural-feature validation only, not the primary corpus), plus each
  source's exact licence terms and a link.
- **Composition table** (fraud prevalence ~15%, per-class shares) as
  actually realised in `dev` vs `full`, with any source that was
  unreachable at build time flagged explicitly rather than silently
  dropped.
- **The simulated component, declared plainly**: there is no public
  dataset of refund claims with claimant history, so Pillar 4 and the
  claim ledger (`simulate_ledger.py`) are simulated, parameters anchored
  to published rates. Declared here, in the README, and in a code comment
  at the top of `simulate_ledger.py` — three places, per the spec.
- **Sensitivity sweep** (tornado chart) over the least-certain simulator
  parameters.
- **Licensing gate**: what `scripts/check_image_licenses.py` enforces and
  why (no CC BY-NC-derived image files committed to this public repo —
  see the implementation plan's licensing fix).
