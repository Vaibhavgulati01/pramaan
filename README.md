# PRAMAAN

**Risk-controlled selective adjudication of refund-claim evidence.**

[![CI](https://github.com/Vaibhavgulati01/pramaan/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaibhavgulati01/pramaan/actions/workflows/ci.yml)
[![Leakage audits](https://github.com/Vaibhavgulati01/pramaan/actions/workflows/leakage.yml/badge.svg)](https://github.com/Vaibhavgulati01/pramaan/actions/workflows/leakage.yml)

> Every refund desk runs on one assumption: a photograph is proof.
> That assumption broke in 2026.

PRAMAAN adjudicates the *evidence* attached to a refund, damage, or
non-receipt claim and returns one of `APPROVE / REVIEW / DENY` — with a
**distribution-free, finite-sample guarantee on the false-denial rate**,
and an abstention band whose width is derived from the cost of a wrong
decision rather than chosen by hand.

---

**Status: under active build — Phase 1 of 9 complete** (ingest,
canonicalisation, benchmark builder, leakage audits; see
[`PROGRESS.md`](PROGRESS.md)). This README is a living document,
generated in its final numeric form by `scripts/inject_metrics.py` from
`reports/{tier}/metrics.json`. Nothing here is hand-typed once Phase 6
lands. Every number in this repo is scale-labeled per
[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md):

| Tier | Purpose | Is it a result? |
|---|---|---|
| `smoke` | CI, proves the pipeline executes | No |
| `dev` | Local, proves every mechanism works | **No — never.** See `reports/dev/`, always headed `DEV — not a held-out result` |
| `full` | VM-run, the one frozen test set | **Yes — the only source of the headline below** |

---

## Contents
[Problem](#the-problem) · [Approach](#approach) · [Results](#results) ·
[The guarantee](#the-guarantee) · [Shift & robustness](#shift--robustness) ·
[Ablations](#ablations) · [Reproduce](#reproduce) ·
[Architecture](#architecture) · [Data](#data) ·
[Limitations](#limitations) · [Safety](#safety)

## The problem

Fraud systems output a score. A score is not a decision, and a decision
made from an uncalibrated score under distribution shift is an unbounded
liability. PRAMAAN adjudicates refund/damage-claim evidence and produces
three-tier decisions with a certified bound on the false-denial rate,
built from evidence pillars designed to survive the shift that actually
happens in this domain: the attacker upgrading their image generator.

*(Sourced figures — return-fraud prevalence, India RTO context, etc. —
land here in Phase 9 alongside the full problem framing;
`PRAMAAN_v2_architecture.md` §1 has the source paragraph this expands on.)*

## Approach

**What this is not:** an AI-image detector. Those generalise poorly to
unseen generators — see [Ablations](#ablations).

**What this is:** four independent evidence pillars — cryptographic
**provenance**, container **forensics**, a temporal **reuse graph**, and
claimant **behaviour** — fused, group-conditionally calibrated, and
converted into a certified three-tier decision via Learn-then-Test.
Three of the four pillars are generator-agnostic by construction, which
is what lets the statistical guarantee survive generator shift.

*(Architecture diagram — Phase 9.)*

## Results

<!-- BEGIN:results -->
> **⚠️ These are `dev`-scale numbers, not a held-out result.**
> They evaluate the *train* split's out-of-fold predictions and exist to
> prove the mechanisms work. The reportable certificate and results come
> from the `full` tier — see [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md).

Evaluated on the **train** split of the `dev` corpus — 1,813 claims, 14.5% fraud prevalence.

| System | PR-AUC (95% CI) |
|---|---|
| approve all | 0.145 [0.128, 0.162] |
| deny all | 0.145 [0.128, 0.162] |
| rules engine | 0.240 [0.201, 0.283] |
| clip probe | 0.382 [0.330, 0.436] |
| resnet style pixel probe | 0.174 [0.148, 0.204] |
| behaviour only gbm | 0.171 [0.141, 0.204] |
| **PRAMAAN (full)** | 0.512 [0.453, 0.569] |

**₹35,437 per 1,000 claims** at **83.0% review load**.

| Policy | ₹ per 1,000 claims |
|---|---|
| approve all | ₹632,451 |
| deny all | ₹4,433,044 |
| review all | ₹40,000 |
| **PRAMAAN** | **₹35,437** |

Calibration: Brier **0.0856**, ECE **0.0080**, MCE **0.0219** (10 equal-mass bins).

Cascade: **147.9 ms/claim** mean, stage-exit 0% / 0% / 100% (stages 1/2/3).
<!-- END:results -->

### Why false positives cost more than false negatives

*(Phase 5/9. At typical order values,
`C_FP = order_value + ₹250 + 0.35×₹3,000` exceeds `C_FN = order_value +
₹180` — see `configs/costs.yaml`. This asymmetry is what derives the
width of the abstention band.)*

## The guarantee

<!-- BEGIN:guarantee -->
> **⚠️ These are `dev`-scale numbers, not a held-out result.**
> They evaluate the *train* split's out-of-fold predictions and exist to
> prove the mechanisms work. The reportable certificate and results come
> from the `full` tier — see [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md).

**No α/δ rung certified.** Every rung of the pre-committed ladder was
attempted and each failed; that is the published result rather than a
loosened bound. See [`docs/GUARANTEE.md`](docs/GUARANTEE.md).

| α | δ | Outcome |
|---|---|---|
| 0.03 | 0.1 | failed — no threshold reached 222 denials, the minimum that can certify alpha=0.03 at delta=0.1 even with zero errors |
| 0.05 | 0.1 | failed — no threshold reached 131 denials, the minimum that can certify alpha=0.05 at delta=0.1 even with zero errors |
| 0.1 | 0.1 | failed — no threshold reached 64 denials, the minimum that can certify alpha=0.1 at delta=0.1 even with zero errors |
| 0.1 | 0.2 | failed — no threshold reached 45 denials, the minimum that can certify alpha=0.1 at delta=0.2 even with zero errors |
<!-- END:guarantee -->

Full statement, the three caveats that qualify it, and the power analysis
that sized the corpus: [`docs/GUARANTEE.md`](docs/GUARANTEE.md).

## Shift & robustness

*(Phase 6/9 — two experiments per condition: certificate frozen at
in-distribution calibration tested under shift, vs. re-certified using
shifted calibration data. Real ❌s reported, not hidden.)*

## Ablations

<!-- BEGIN:ablations -->
> **⚠️ These are `dev`-scale numbers, not a held-out result.**
> They evaluate the *train* split's out-of-fold predictions and exist to
> prove the mechanisms work. The reportable certificate and results come
> from the `full` tier — see [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md).

Each ablation is reported twice. The **full corpus** column is confounded
by source dataset — every synthetic-fraud claim comes from GenImage, which
carries 2.58× the fraud rate of ABO — so it is an *upper bound* on the
pixel pillars. The **ABO-only** column holds source constant.

| Ablation | Δ PR-AUC (full corpus) | Δ PR-AUC (ABO-only) |
|---|---|---|
| no provenance | +0.0000 | +0.0000 |
| no forensics | -0.1024 | -0.0335 |
| no reuse | -0.0220 | -0.0273 |
| no behaviour | -0.0090 | +0.0007 |
| no rings | -0.0013 | -0.0083 |

**Negative controls** (these validate the methodology, not a hypothesis):

| Control | PR-AUC | Prevalence | Pass? |
|---|---|---|---|
| label shuffle | 0.1543 | 0.1445 | ✅ |
| random features | 0.1620 | 0.1445 | ✅ |
<!-- END:ablations -->

Predictions were registered in advance
([`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md)); Phase 9 reports
actuals against them, including the misses.

## Reproduce

```bash
git clone https://github.com/Vaibhavgulati01/pramaan && cd pramaan
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
python -m pramaan.cli setup
python -m pramaan.cli all --scale smoke   # <5 min, proves the pipeline runs (CI does this on every push)
python -m pramaan.cli all --scale dev     # local mechanism validation, not a result
```

`make <target>` works identically wherever `make` is available (CI, the
VM, Linux/Mac); on Windows without `make`, use the `python -m
pramaan.cli` form above directly. Full-scale reproduction
(`make data-full && make train && make eval SCALE=full`) is documented in
[`docs/REAL_DATA_ONRAMP.md`](docs/REAL_DATA_ONRAMP.md) and is run on a
machine with real compute, not in CI.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Data

**PRAMAAN-Bench-v1** is built from two real public sources — [Amazon
Berkeley Objects](https://registry.opendata.aws/amazon-berkeley-objects/)
(CC BY-NC 4.0) for real product photos, and a
[GenImage](https://github.com/GenImage-Dataset/GenImage)-derived set
(CC BY-NC-SA 4.0) for AI-generated images across 8 generator families.

**The claim ledger is simulated, and that is stated in three places** —
here, in [`docs/DATA_CARD.md`](docs/DATA_CARD.md), and in the module
docstring of `benchmarks/simulate_ledger.py`. There is no public dataset
of refund claims with claimant history, so every claimant identity, order
record, timestamp, and device fingerprint is synthetic, with parameters
anchored to published rates (~15% fraud among returns, ~23% India RTO).
Only the images are real. Pillar 4 (claimant behaviour) is therefore
evaluated entirely on simulated data and its contribution is reported
separately from the image pillars, never blended into a single headline
number.

**No dataset images are redistributed here** — `data/` is gitignored in
full. The repo ships the *builder* and a manifest recording each claim's
upstream source, licence, source SHA256, transform, and output SHA256, so
the corpus is rebuildable byte-identically without this repo carrying
NC-licensed images. See [`docs/REAL_DATA_ONRAMP.md`](docs/REAL_DATA_ONRAMP.md)
for the exact schema that replaces the simulator with real merchant data.

## Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — kept long and specific
on purpose; growing throughout the build rather than written only at the
end.

## Safety

See [`SAFETY.md`](SAFETY.md) and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
