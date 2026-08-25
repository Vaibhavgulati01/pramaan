# PRAMAAN

**Risk-controlled selective adjudication of refund-claim evidence.**

[![CI](https://github.com/Vaibhavgulati01/pramaan/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaibhavgulati01/pramaan/actions/workflows/ci.yml)

> Every refund desk runs on one assumption: a photograph is proof.
> That assumption broke in 2026.

PRAMAAN adjudicates the *evidence* attached to a refund, damage, or
non-receipt claim and returns one of `APPROVE / REVIEW / DENY` — with a
**distribution-free, finite-sample guarantee on the false-denial rate**,
and an abstention band whose width is derived from the cost of a wrong
decision rather than chosen by hand.

---

**Status: under active build (Phase 0 of 9 — see the implementation
plan referenced below).** This README is a living document, generated in
its final numeric form by `scripts/inject_metrics.py` from
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

*(Populated by `scripts/inject_metrics.py` from `reports/full/metrics.json`
once the VM run lands — Phase 9. Until then, see `reports/dev/` for
DEV-labeled mechanism-validation numbers, linked not inlined, per the
tier discipline above.)*

### Why false positives cost more than false negatives

*(Phase 5/9. At typical order values,
`C_FP = order_value + ₹250 + 0.35×₹3,000` exceeds `C_FN = order_value +
₹180` — see `configs/costs.yaml`. This asymmetry is what derives the
width of the abstention band.)*

## The guarantee

See [`docs/GUARANTEE.md`](docs/GUARANTEE.md) — filled in through Phases
0.5, 4, and 9.

## Shift & robustness

*(Phase 6/9 — two experiments per condition: certificate frozen at
in-distribution calibration tested under shift, vs. re-certified using
shifted calibration data. Real ❌s reported, not hidden.)*

## Ablations

*(Phase 9 — actuals reported against
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md), including any
misses.)*

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

See [`docs/DATA_CARD.md`](docs/DATA_CARD.md) — which parts are real
public data, which are simulated (declared here, in that doc, and in a
code comment at the top of `simulate_ledger.py`), and the sensitivity
sweep over simulator parameters.

## Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — kept long and specific
on purpose; growing throughout the build rather than written only at the
end.

## Safety

See [`SAFETY.md`](SAFETY.md) and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
