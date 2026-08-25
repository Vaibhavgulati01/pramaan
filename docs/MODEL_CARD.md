# Model card

> **Status: skeleton.** Filled in as the fusion model lands in Phase 3
> and calibration/risk control in Phase 4.

## What will go here

- Intended use (one loss class: refund/damage-claim evidence
  adjudication) and explicit out-of-scope uses.
- LightGBM fusion architecture, the ~60-dim feature schema (versioned),
  and the 9 monotone-constrained features with their a-priori direction
  and rationale.
- Mondrian isotonic calibration groups (`{category x price-band}`) and
  the thin-cell shrinkage fallback.
- Training/calibration/test data provenance, with the same DEV/full
  labeling discipline as everywhere else in the repo.
- Known failure modes, pointing at `docs/LIMITATIONS.md` and the shift
  matrix rather than duplicating them.
