# VM handoff — running the `full` tier

Everything in this repository has been built and verified at `smoke` and
`dev` scale on a CPU-only Windows laptop. **The one thing that machine
cannot produce is the reportable result**: the `full` tier needs a
35,000-claim corpus, which means ~35k CLIP embeddings and a much larger
source download than 44 GB of free disk comfortably allows.

This document is the complete set of commands to run on the VM (64 GB
RAM, 16 GB VRAM), and — just as important — what to look for when they
finish.

---

## 0. Why `full` matters, in one paragraph

At `dev` scale **no α/δ rung certified**, and that was the correct
answer: the calibration split held 580 claims and only 18 high-confidence
denials, far below the 45–222 the power analysis requires. The
architecture's headline claim — *"at most α% of auto-denied claims are
legitimate, with 1−δ confidence"* — is currently **unproven, not
disproven**. `full` is the run that settles it.

---

## 1. Setup (~10 minutes)

```bash
git clone https://github.com/Vaibhavgulati01/pramaan && cd pramaan
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,provenance]"
python -m pramaan.cli setup          # verifies every dependency imports
```

**On Linux, switch the vector backend to FAISS.** The default is NumPy
because `faiss-cpu` and `torch` each bundle an OpenMP runtime and abort
on import together on Windows. That conflict does not exist on Linux, and
at 35k vectors FAISS's `IndexHNSWFlat` is worth having:

```yaml
# configs/model.yaml — already set for the full tier
reuse_index:
  full: { type: hnsw_flat, ef_construction: 200, m: 32 }
```

> ⚠️ **Determinism caveat.** HNSW is an approximate index and its
> construction is order-sensitive. Using it means the `full` run is *not*
> byte-reproducible in the way `dev` is. If exact reproducibility matters
> more than speed, force `type: flat_ip` — at 35k vectors an exact
> matmul is still perfectly tractable on this hardware.

---

## 2. Build the corpus (~2–4 hours, mostly network)

```bash
python -m pramaan.cli data-full 2>&1 | tee logs/data_full.log
```

**What it does:** simulates 35,000 claims, fetches ABO and GenImage
sources (cached under `data/raw/`, resumable), applies per-class
transforms, reconciles and verifies all four split constraints, and
writes a manifest with per-claim provenance.

**Check when it finishes:**

- [ ] `split verification: OK (generator-holdout, temporal, entity, ring)`
- [ ] fraud prevalence ≈ 0.15
- [ ] `reconciliation: N claim(s) dropped` — N should be well under 0.1%.
      A large number means the entity matcher is over-merging at scale
      and is worth investigating before trusting anything downstream.
- [ ] Disk: expect ~15–25 GB under `data/`.

**If the GenImage fetch stalls:** it reads ~100 MB parquet row groups
over HTTP and retries transient failures. It is resumable — re-run the
same command and it continues from the cache.

---

## 3. Train (~1–2 hours, GPU-accelerated for CLIP)

```bash
python -m pramaan.cli train --scale full 2>&1 | tee logs/train_full.log
```

Feature extraction dominates (~35k images through CLIP + forensics).
CLIP uses CUDA automatically when available.

**Check:**

- [ ] `features: 48 (schema 1.1.0)`
- [ ] `monotone constraints: 11`
- [ ] OOF Brier improves after calibration (it was 0.1013 → 0.0856 on dev)
- [ ] Worst-calibrated cells are reported — expect high-price cells to be
      worst, as on dev

---

## 4. Certify — **the moment that matters** (~5 minutes)

```bash
python -m pramaan.cli certify --scale full 2>&1 | tee logs/certify_full.log
```

This consumes the calibration split **once**, seals its content hash, and
walks the pre-committed α/δ ladder from
[`PREREGISTRATION.md`](PREREGISTRATION.md).

**Three possible outcomes, all publishable:**

| Outcome | What to do |
|---|---|
| **α=0.03 certifies** | The headline claim holds. Proceed. |
| **A lower rung certifies** (0.05, or 0.10) | Report that rung *and every failed attempt above it*. The ladder exists precisely so this is not a retreat. |
| **Nothing certifies** | Publish that, with the required n. `GUARANTEE.md` already contains the honest framing; do **not** extend or loosen the ladder. |

**Prediction on record:** α=0.03 certifies if the near-zero realised FDR
seen at dev (0.000 at t=0.90) holds at scale; otherwise α=0.05.
Projected denials at `full` are ~217, which clears α=0.03's zero-error
floor of 76 but not its 0.3α floor of 222. Genuinely uncertain.

⚠️ **Commit `reports/full/calibration_seal.json` immediately.** It is
what makes "the calibration split was used once" checkable rather than
asserted.

---

## 5. Evaluate (~2–4 hours; ablations refit the model repeatedly)

```bash
python -m pramaan.cli eval --scale full 2>&1 | tee logs/eval_full.log
```

Runs baselines, both ablation variants, negative controls, bootstrap CIs
(2,000 resamples), and writes `reports/full/metrics.json`.

**Check — and these are the ones that would invalidate everything:**

- [ ] **Negative controls PASS.** Label-shuffle and random-features must
      collapse to prevalence (~0.15). A failure here means something is
      leaking and *no other number in the run can be trusted*.
- [ ] The ablation table has both `all` and `abo_only` rows.
- [ ] PR-AUC CI does not overlap the best baseline's — if it does, the
      architecture has not demonstrated an improvement, and that is the
      finding.

If `eval` runs out of memory, `--skip-slow` omits ablations and negative
controls; run those separately afterwards.

---

## 6. Generate the README and commit

```bash
python -m pramaan.cli report --scale full
python -m pramaan.cli report --scale full --check   # must pass
git add reports/full README.md
git commit -m "Full-scale results"
```

`--check` is what CI runs. Every README figure comes from
`metrics.json`; nothing is hand-typed.

---

## 7. What I still owe, after the numbers land

These are deliberately left until `full` exists, because doing them
earlier would mean writing about results that do not yet exist:

1. **`PREREGISTRATION.md` → actuals section.** Report each prediction
   against what happened, *including the misses*. The predictions
   explicitly bet against the spec's own headline claim; if the
   controlled ablation puts forensics clearly ahead of reuse, that gets
   written down plainly.
2. **`reports/SCALE_CONCORDANCE.md`.** Compare `dev` and `full` on the
   headline quantities. Disagreement is a reported finding, not
   something to reconcile away.
3. **The shift matrix.** `eval/shift_matrix.py` has all eight conditions
   and the two-experiment design; it needs a driver that applies each
   perturbation to the frozen test split and fills the table.
4. **README problem statement** with the sourced figures, and the
   architecture diagram.
5. **A tagged release.**

---

## Known risks, stated in advance

| Risk | Signal | Response |
|---|---|---|
| Deny rate lower than assumed | Fewer denials than ~217 | Expected: dev measured 3.1% against a 0.075 assumption. Walk the ladder; report honestly. |
| Ablation still confounded | `all` and `abo_only` columns far apart | Already documented in `LIMITATIONS.md`; report both, lead with `abo_only`. |
| Review load stays ~80%+ | `review_load_pct` in metrics.json | At ₹40/review vs ₹3,300/false-positive this is *economically rational*, not a defect. Already in `LIMITATIONS.md`. |
| HNSW breaks determinism | Two runs differ | Switch to `flat_ip` and re-run, or state the run is not byte-reproducible. |

---

## Current state at handoff

| | |
|---|---|
| Phases complete | 0 through 8 |
| Tests | 514, all passing |
| Lint / types | ruff + mypy clean |
| CI | `ci.yml` and `leakage.yml` green |
| `dev` corpus | 3,000 claims, all four split constraints verified |
| `dev` certificate | nothing certified — correctly, on power grounds |
| Blocking on | this document |
