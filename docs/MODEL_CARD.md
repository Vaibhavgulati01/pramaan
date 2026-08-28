# Model card

## Intended use

Adjudicating **evidence attached to a refund, damage, or non-receipt
claim** for an Indian e-commerce merchant. One class of loss, deliberately
(§10 refuses a second): the model scores whether a claim's *evidence* is
fraudulent, and a selective policy converts that into
`APPROVE / REVIEW / DENY`.

### Out of scope

- **Any decision without a human in the loop above the abstention band.**
  The REVIEW tier exists because the model is not good enough to act
  alone on uncertain claims, and at current cost parameters it routes
  ~83% of claims to a human.
- **Auto-denial without a live certificate.** If Learn-then-Test
  certifies nothing, the policy disables auto-deny outright. This is
  enforced in code (`policy/selective.py`, `api/app.py`), not by
  convention.
- **Claimant-level judgements.** The system scores claims, not people. A
  claimant flagged twice is two flagged claims, not a verdict about them.
- **Any market whose cost structure differs.** The abstention band is
  *derived* from `configs/costs.yaml`. Different costs mean a different
  band, and the numbers here do not transfer.

## Architecture

LightGBM (binary objective) over a **48-feature versioned schema**
(`FEATURE_SCHEMA_VERSION = 1.1.0`) assembled from four evidence pillars
plus ring features, followed by Mondrian (group-conditional) isotonic
calibration.

| Pillar | Features | Generator-agnostic? |
|---|---|---|
| P1 provenance (C2PA) | 6 | yes |
| P2 container forensics | 14 | yes |
| P3 reuse graph | 9 | **yes — the moat** |
| P4 claimant behaviour | 13 | yes |
| Ring features | 6 | yes |

### Monotone constraints

**11 features** carry a monotone constraint, each with a written
justification in `fusion/schema.py`. Two are structural rather than
statistical:

- `ring_is_first_seen` is forced non-increasing, making first-seen
  immunity **inviolable by the model** — it cannot learn to penalise the
  earliest claimant on a cluster no matter what a thin data slice
  suggests.
- `reuse_best_hamming` is constrained opposite to
  `reuse_max_clip_similarity`, so two measures of the same thing cannot
  disagree.

Features whose direction is genuinely ambiguous are left unconstrained,
and the omissions are documented with reasons. A wrong constraint is
worse than none.

### Determinism

Single-threaded, fixed seed, `deterministic=true`, `force_row_wise=true`.
NumPy exact search in P3 rather than FAISS (see
`docs/EVALUATION_PROTOCOL.md`).

## Training data

- **Train split only.** The calibration split is reserved for
  Learn-then-Test and the test split is sealed; `FusionModel.fit` raises
  if handed either.
- The calibrator is fitted on **K-fold out-of-fold predictions** from
  train, never in-sample scores and never the calibration split.
- Images are real (ABO + a GenImage-derived set). **The claim ledger is
  simulated** — see `docs/DATA_CARD.md`.

## Performance (`dev` tier — not a held-out result)

| Metric | Value |
|---|---|
| PR-AUC | 0.512 [0.453, 0.569] |
| Brier | 0.0856 |
| ECE (10 equal-mass bins) | 0.0080 |
| MCE | 0.0219 |
| Best baseline (CLIP probe) | 0.382 |

Calibration is excellent globally and **~7× worse on high-price cells**
(ECE ~0.07 vs 0.008) — precisely the failure Mondrian calibration exists
to surface. A single global reliability curve would hide it.

The `full`-tier numbers are the reportable ones and do not exist yet
(`docs/VM_HANDOFF.md`).

## Known failure modes

Each links to the measurement rather than being asserted:

1. **Gain-based importance is misleading here.** Forensics takes ~71% of
   gain but has near-zero *label* effect size (d ≈ 0.1) against a strong
   *source-dataset* effect (d ≈ 1.6). Read ablations, not importances.
2. **The corpus retains an irreducible source confound.** Synthetic
   fraud is necessarily GenImage-sourced, which carries 2.58× the fraud
   rate of ABO. Every ablation is therefore reported twice.
3. **Screen re-photography is essentially undetected** — our simulation
   produces weaker artifacts than reality.
4. **P4 is evaluated entirely on simulated behaviour.**
5. **Entity resolution is fuzzy in both directions.** False splits (a
   claimant varying every detail) are unmeasured and probably
   non-zero.

All expanded in `docs/LIMITATIONS.md`.

## Ethical considerations

- **Adverse-action reasoning is templated and deterministic**, never
  model-generated, so a denial can be re-derived byte for byte months
  later.
- **Absent evidence is never adverse.** Missing C2PA is `UNKNOWN`; a PNG
  has no quantisation tables; neither counts against a claimant.
- **The cost model prices a wrongly-denied honest claimant above a missed
  fraud** at every plausible order value, so the policy is structurally
  biased toward not denying.
- **No raw scores are exposed**, preventing both evasion and the use of
  this system as a scoring service about individuals.

## Maintenance

`monitoring/drift.py` tracks PSI/KS per feature and runs a
`GuaranteeWatchdog` that recomputes realised false-denial rate on each
matured label batch and **widens the abstention band automatically** when
it crosses α. A certificate that cannot notice its own expiry is a
certificate only until the first shift.
