# Evaluation protocol

**Committed before the `full` test set was unsealed.** Together with
[`PREREGISTRATION.md`](PREREGISTRATION.md), this is what makes unsealing
a ceremony with a paper trail rather than an assertion after the fact.

## The three-tier scale discipline

| Tier | Where | Size | Purpose | Ever a reported result? |
|---|---|---|---|---|
| `smoke` | CI, every push | ~200 claims | proves the pipeline executes | **No** |
| `dev` | Local dev machine | 3,000 claims | proves every mechanism works | **No** |
| `full` | VM, run once | 35,000 claims | the frozen test set | **Yes — the only one** |

`pramaan eval --scale {smoke,dev,full}` writes to `reports/{tier}/` and
nowhere else.

### The reporting contract

This is the rule that keeps two scales from being quietly mixed:

- **`dev` tables live only in `reports/dev/` and are linked from the
  README, never inlined into it.** Every heading carries `DEV — not a
  held-out result`.
- **The README carries exactly one results section**, sourced from
  whichever tier actually ran, with the tier named in the text.
- Once both exist, `reports/SCALE_CONCORDANCE.md` compares `dev` and
  `full` on the headline quantities. **Disagreement between scales is a
  reported finding, not an inconsistency to reconcile away** — a
  shift-matrix cell flipping ✅→❌ between scales is informative.

## Splits

Four constraints, all verified by `benchmarks/splits.py` and enforced in
CI by `.github/workflows/leakage.yml`:

1. **Generator-family holdout** — train {SD15, BigGAN} · calibrate
   {GLIDE} · test {Midjourney, ADM, VQDM, Wukong}. (The spec names SD 1.4;
   the public mirror declares that label but contains no rows, so SD 1.5
   is substituted — see `docs/DATA_CARD.md`.)
2. **Temporal** — all test claims strictly after all calibration claims,
   strictly after all train claims.
3. **Entity-disjoint** — zero canonical-identity overlap
   (`eval/entity_leakage_audit.py`).
4. **Ring-disjoint** — no `image_group_id` spans two splits.

Where the four cannot all hold simultaneously, `reconcile_splits()` drops
the offending claims and **reports the count** rather than silently
relaxing a constraint.

### Which split is used for what

| Split | Used for | Never used for |
|---|---|---|
| `train` | fitting LightGBM; fitting the calibrator on K-fold out-of-fold predictions | — |
| `calibration` | Learn-then-Test, **once** | model fitting, threshold tuning, hyperparameters |
| `test` | final evaluation, unsealed **once** | anything before Phase 9 |

Enforced in code, not by memory: `FusionModel.fit` raises
`SplitDisciplineError` on calibration or test rows, and the calibration
split's content hash is committed and re-checked in CI
(`docs/GUARANTEE.md`, caveat 3).

## Metrics, at four granularities

| Granularity | Metrics |
|---|---|
| Image | PR-AUC, per-generator recall |
| Claim | PR-AUC, P@90R, R@95P, Brier, ECE (10 equal-mass bins), MCE, per-group calibration |
| Claimant | precision/recall on repeat abusers |
| Ring | cluster precision/recall, purity, time-to-detection |

System-level, reported together and never separately:

- **₹ loss per 1,000 claims** against every baseline
- **review load %** — reported *alongside* every ₹ figure, because a
  cheap policy that reviews everything is not automation
  (`docs/LIMITATIONS.md`)
- certified α with n, realised FDR, and the HB p-value
- mean compute ms and the stage-exit distribution
- **bootstrap 95% CIs, 2,000 resamples, on everything**

## Baselines

| Baseline | Why it is here |
|---|---|
| Approve-all / deny-all | trivial floors; deny-all also exposes the cost asymmetry |
| Rules engine (claim count + value threshold) | what merchants actually run today |
| CLIP linear probe | "just use a foundation model" |
| ResNet-50 features + linear head, *in the spirit of CNNSpot* | the standard synthetic-image detector. **Not** a faithful CNNSpot reimplementation, and never labelled as one |
| Behaviour-only LightGBM | "just use tabular fraud ML" |
| PRAMAAN (full) | |

## Ablations

Leave-one-pillar-out (P1–P4), rings-off, cascade-off, calibration-off,
monotone-constraints-off, risk-control-off (naive F1 threshold),
federated-index-off.

**Each ablation is reported twice: on the full corpus and on the
ABO-only subset.** The full-corpus version is confounded by source
dataset (`docs/LIMITATIONS.md`) and is an *upper bound* on the pixel
pillars; the ABO-only version holds source constant. Reporting only the
first would overstate forensics by roughly 3×.

## Negative controls

These validate the methodology itself, so they are run and reported at
`dev` scale freely — they are not pre-registered hypotheses:

- **Label shuffle** — retrain on permuted labels; PR-AUC must collapse to
  prevalence.
- **Random features** — replace the vector with noise; same expectation.
- **Temporal constraint disabled** — re-run P3 without the
  strictly-earlier rule and report the inflated number it *would* have
  produced. This demonstrates the common bug is understood because it was
  measured, not merely avoided.

## Shift matrix — two experiments, not one

For each of the eight conditions, both are run:

1. **Frozen calibration** — the certificate computed in-distribution,
   applied to shifted data. Answers: *does the guarantee survive the
   shift?*
2. **Re-certified** — Learn-then-Test re-run on shifted calibration data.
   Answers: *could we recover a guarantee if we knew about the shift?*

These have different answers and different operational implications, and
the spec conflates them. The gap between them is reported explicitly.

## Determinism

Two consecutive `dev` runs must produce **byte-identical**
`metrics.json`. Guaranteed by: `OMP_NUM_THREADS=1`, NumPy exact search in
P3 (FAISS reserved for `full` on Linux), LightGBM
`deterministic=true`/`force_row_wise=true`/single-threaded, and fixed
seeds recorded in `run_manifest.json`.

## What CI enforces

| Check | Workflow |
|---|---|
| lint, types, unit tests, `smoke` pipeline end-to-end | `ci.yml` |
| split verification across multiple seeds | `leakage.yml` |
| calibration-seal integrity | `leakage.yml` |
| committed README matches `inject_metrics.py` output | `ci.yml` |
| no NC-licensed image derivative committed | `ci.yml` |

CI runs `smoke` only. Nobody should expect CI to reproduce the science;
it exists to prove the pipeline executes and the invariants hold.
