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

**Status: Phases 0–8 complete; Phase 9 blocked on the full-scale run.**
All mechanisms are built, tested (514 tests) and exercised end-to-end at
`dev` scale. What remains is the `full`-scale run on a larger machine —
the runbook is [`docs/VM_HANDOFF.md`](docs/VM_HANDOFF.md), progress is
tracked in [`PROGRESS.md`](PROGRESS.md). This README is a living document,
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

### The scale, from named sources

| Figure | Value | Source |
|---|---|---|
| US merchandise returned, 2025 | **$849.9B** (15.8% of retail sales) | NRF / Happy Returns, *2025 Retail Returns Landscape*, 15 Oct 2025 |
| Online sales returned | **19.3%** | same |
| Returns that are fraudulent | **9%** | same |
| Cost of fraudulent returns & claims, 2024 | **$103B** | Appriss Retail / Deloitte, *Consumer Returns in the Retail Industry*, 7th annual, Dec 2024 |
| Returns that are fraudulent | **15.14%** | same |

**Those last two rows disagree, and we are not going to average them.**
Two credible industry surveys put fraudulent returns at 9% and 15.14%
respectively — a 68% relative gap, from different panels, definitions of
"fraudulent", and years. Anyone quoting a single confident number for
this quantity is choosing one and not telling you. The honest reading is
that return fraud is somewhere in the high single digits to mid teens as
a share of returns, and that **the base rate is uncertain enough that a
system acting on it should be able to abstain** — which is the design
this repository argues for.

### Why India makes it harder

Roughly 60–65% of Indian e-commerce orders are cash-on-delivery, and
COD orders are returned-to-origin at rates that logistics vendors put
between **20% and 35%**, against low single digits for prepaid.

**Source quality note:** unlike the two rows above, these India figures
come from logistics-vendor blogs (Shipway, GoKwik, Qikink and similar),
not from a primary survey with a published methodology. We could find no
primary, methodologically-documented source for Indian RTO or
return-fraud rates. They are directionally useful and quantitatively
unreliable, and we mark them as such rather than laundering a marketing
figure into a statistic. The one hard Indian data point we did find is a
prosecution, not a rate: a seller-side ring defrauded Meesho of
**₹5.5 crore over seven months** by exploiting the returns policy
(*Deccan Herald*).

This matters for the architecture: high return volume plus a
photograph-as-proof workflow is exactly the setting where a scoring
model gets deployed as a decision-maker without anyone bounding its
error rate.

### What follows from that

A refund desk that must act on ~15% of orders coming back, with a fraud
base rate no one can pin down inside a factor of two, cannot safely run
on a threshold someone picked off a PR curve. It needs a decision
procedure that (a) knows what it does not know, (b) carries a bound on
how often it wrongly denies an honest customer, and (c) says so out loud
when that bound stops holding.

## Approach

**What this is not:** an AI-image detector. Those generalise poorly to
unseen generators — see [Ablations](#ablations).

**What this is:** four independent evidence pillars — cryptographic
**provenance**, container **forensics**, a temporal **reuse graph**, and
claimant **behaviour** — fused, group-conditionally calibrated, and
converted into a certified three-tier decision via Learn-then-Test.
Three of the four pillars are generator-agnostic by construction, which
is what lets the statistical guarantee survive generator shift.

```mermaid
flowchart TD
    A["<b>L0 · INGEST</b><br/>image bytes · order · claimant · claim text<br/>identity canonicalisation — phone / email / address / device"]

    A --> S1["<b>Stage 1</b> ~2 ms<br/>P1 provenance (C2PA) + P4 behaviour<br/><i>cached aggregates, keyed by canonical identity</i>"]
    S1 -->|"p &lt; τ_lo or p &gt; τ_hi"| EX(["early exit<br/><i>claim still enters the reuse index</i>"])
    S1 --> S2["<b>Stage 2</b> ~42 ms<br/>P2 container forensics<br/>QT tables · thumbnail desync · ELA · DCT · FFT"]
    S2 -->|early exit| EX
    S2 --> S3["<b>Stage 3</b> ~120 ms<br/>P3 reuse graph — <b>the moat</b><br/>pHash ≤ 2 → LSH bands → CLIP ≥ 0.98<br/><i>query_then_add: temporal leak is structurally impossible</i>"]

    S3 --> F["<b>L2 · FUSION + CALIBRATION</b><br/>LightGBM, 48-feature versioned schema, 11 monotone constraints<br/>→ Mondrian isotonic per {category × price band}<br/>outputs p̂ with <i>per-group</i> reliability"]

    F --> R["<b>L3 · RISK CONTROL — Learn-then-Test</b><br/>Hoeffding–Bentkus p-values · fixed-sequence testing<br/>→ certified set Λ̂(α, δ)<br/><b>P(FDR_deny ≤ α) ≥ 1 − δ</b>"]

    R --> P{"<b>L4 · COST-OPTIMAL POLICY</b><br/>argmin ₹ <i>within</i> Λ̂"}
    P -->|"p̂ &lt; t_approve"| AP([APPROVE])
    P -->|"between"| RV([REVIEW])
    P -->|"p̂ &gt; t_deny"| DN([DENY])

    R -.->|"<b>nothing certified</b>"| OFF["auto-deny <b>disabled</b><br/><i>enforced in code, not by convention</i>"]
    OFF -.-> RV

    EX --> F

    style S3 fill:#1a4d2e,stroke:#2d7a4a,color:#fff
    style R fill:#1e3a5f,stroke:#3d6fa5,color:#fff
    style OFF fill:#5c1a1a,stroke:#a33,color:#fff
    style DN fill:#5c1a1a,stroke:#a33,color:#fff
    style AP fill:#1a4d2e,stroke:#2d7a4a,color:#fff
```

**Reading the diagram.** Two arrows carry most of the argument. The
dotted one — *nothing certified → auto-deny disabled* — is the path the
system actually took at `dev` scale, and it is enforced in
[`policy/selective.py`](src/pramaan/policy/selective.py) rather than left
to operator discipline. The `early exit → reuse index` arrow exists
because omitting it was a real bug: claims that exited cheaply were
never indexed, silently blinding the reuse graph to exactly the
high-volume attacker it is meant to catch. It is now asserted by a test
that was verified by re-introducing the bug.

Stage timings are measured on this machine, not estimated.

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

The dataflow diagram is in [Approach](#approach) above. Module-by-module
detail, the feature schema, and the design decisions behind each pillar
are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
anchored to the published rates in [The problem](#the-scale-from-named-sources)
(~15% fraud among returns, per Appriss/Deloitte; ~23% India RTO, from the
vendor-reported range whose weakness is flagged there). Anchoring a
simulator to a number we have just called uncertain is a real limitation,
not a detail: it means P4's measured contribution inherits that
uncertainty.
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

## Sources

Figures quoted in [The problem](#the-problem), with their provenance so a
reader can check them rather than trust them:

- National Retail Federation & Happy Returns, [*2025 Retail Returns Landscape*](https://nrf.com/research/2025-retail-returns-landscape) (15 Oct 2025) — $849.9B returned, 15.8% of sales, 19.3% online return rate, 9% of returns fraudulent.
- Appriss Retail & Deloitte, [*Consumer Returns in the Retail Industry*, 7th annual](https://apprissretail.com/news/appriss-retail-annual-research-fraudulent-returns-and-claims-cost-retailers-103b-in-2024/) (Dec 2024) — $103B fraudulent returns and claims, 15.14% of returns fraudulent; 60+ US retailers, US Census data, 150 executives, 1,000 consumers.
- *Deccan Herald*, [seller-side returns-policy fraud ring, ₹5.5 crore](https://www.deccanherald.com/india/karnataka/bengaluru/gang-registers-as-seller-with-e-commerce-company-exploits-returns-policy-swindles-rs-5-5-crore-3302465).
- India RTO rates: **no primary source located.** The 20–35% COD range circulates across logistics-vendor blogs without a published methodology behind it, and is marked as vendor-reported wherever this repo uses it.

## Citation

See [`CITATION.cff`](CITATION.cff).
