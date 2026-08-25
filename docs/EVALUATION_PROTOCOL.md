# Evaluation protocol

> **Status: skeleton.** Committed for real at the end of Phase 5, before
> the `full` test set is ever unsealed — that ordering is the point: this
> document exists so unsealing is a ceremony with a paper trail, not an
> after-the-fact assertion. Extended, never contradicted, by later phases.

## The three-tier scale discipline

| Tier | Where | Purpose | Ever reported as a result? |
|---|---|---|---|
| `smoke` | CI, every push | pipeline executes, <5 min budget | No |
| `dev` | Local dev machine | every mechanism proven correct | **No.** Lives in `reports/dev/`, linked not inlined, headed `DEV — not a held-out result` |
| `full` | VM, user-run | the frozen test set, unsealed exactly once | **Yes — the only source of the README's results section** |

`pramaan eval --scale {smoke,dev,full}` writes to `reports/{tier}/` and
nowhere else. Once `full` exists, `reports/SCALE_CONCORDANCE.md` compares
`dev` vs `full` on the headline quantities; disagreement is a reported
finding, not a silent inconsistency.

## Splits (all enforced by CI at `full` scale, tested against `dev` at build time)

1. Generator-family holdout — train {SD 1.4, BigGAN} · calibrate {GLIDE}
   · test {Midjourney, ADM, VQDM, Wukong}.
2. Temporal — all test claims strictly after all train claims.
3. Entity-disjoint — zero canonical-identity overlap
   (`eval/entity_leakage_audit.py`).
4. Ring-disjoint — no test ring member in training.

## Metrics (four granularities)

Image (PR-AUC, per-generator recall) · Claim (PR-AUC, P@90R, R@95P,
Brier, ECE, per-group calibration) · Claimant (precision/recall on repeat
abusers) · Ring (cluster precision/recall, purity, time-to-detection).

Plus system-level: ₹ loss / 1,000 claims vs. baselines, review load %,
certified α with n and HB p-value, mean compute ms + stage-exit
distribution, bootstrap 95% CIs (2,000 resamples) on everything.

## Pre-registration discipline

Pillar/cascade/calibration/monotone/risk-control/federated ablations are
**pre-registered** in `docs/PREREGISTRATION.md` before the `full` test set
is unsealed. At `dev` scale they are run as mechanism checks only (does
the code execute, is the number coherent) — never interpreted as a
result, so the pre-registration stays meaningfully blind. Negative
controls (label-shuffle, random-feature, temporal-constraint-disabled)
are a different thing — they validate the evaluation methodology itself,
not a pre-registered hypothesis — and are run and reported at `dev` scale
freely.

## What Phase 6 will add here once it lands

Baselines list, ablation list, the two-experiment shift-matrix design
(frozen-calibration-under-shift vs. re-certified-on-shifted-calibration),
and the determinism-test configuration.
