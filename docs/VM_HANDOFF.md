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

## Why `full` matters, in one paragraph

At `dev` scale **no α/δ rung certified**, and that was the correct
answer: the calibration split held 580 claims and only 18 high-confidence
denials, far below the 45–222 the power analysis requires. The
architecture's headline claim — *"at most α% of auto-denied claims are
legitimate, with 1−δ confidence"* — is currently **unproven, not
disproven**. `full` is the run that settles it.

---

## 0. What to copy from the dev machine

**Nothing.** This is worth stating explicitly, because "transfer the
data to the VM" is the natural assumption and it would be wasted effort.

| Thing | How the VM gets it | Transfer? |
|---|---|---|
| Source code, configs, `Makefile` | `git clone` — the repo is public | no |
| `reports/dev/*` (metrics, certificate, seal) | committed in the repo | no |
| ABO product photos | fetched from Hugging Face by `benchmarks/sources.py` | no |
| GenImage-derived images | same, public HF repo | no |
| CLIP ViT-B/32 weights | `open_clip` downloads on first use (~350 MB) | no |
| `data/` (corpus, features, models) | **rebuilt** by `pramaan data-full` | no |
| Credentials / API keys | none exist — no `.env`, no HF token needed | n/a |

`data/` is gitignored in full and every byte of it is derived: the
manifest records each claim's upstream source, licence, source SHA256,
transform and output SHA256, so the corpus is rebuildable rather than
shippable. That is a deliberate property — see `docs/DATA_CARD.md` — and
it is also why no NC-licensed image is ever committed to this public
repo. Copying `data/` across would move ~700 MB of files the VM will
regenerate anyway, at a different and larger scale.

**One optional shortcut.** Copying `data/raw/` (602 MB) pre-seeds the
download cache, saving the VM from re-fetching the ~5,300 source images
`dev` already pulled. It is resumable and additive, so this only helps if
bandwidth is scarce; the `full` build still has to fetch roughly six
times that much on top. Skip it unless the connection is slow.

**What comes back the other way**, once the run finishes: `reports/full/`
(metrics, certificate, calibration seal, figures), the regenerated
`README.md`, and the run logs. Those are small and they are the actual
product of the VM run. Do not copy `data/full/` back.

---

## 1. Setup (~10 minutes)

```bash
git clone https://github.com/Vaibhavgulati01/pramaan && cd pramaan
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,provenance]"
pramaan setup          # verifies every dependency imports
```

**No backend switch is needed, and an earlier version of this document
was wrong to ask for one.** It told you to edit `configs/model.yaml` to
select an `hnsw_flat` index for the `full` tier. Three things were wrong
with that: nothing loads `configs/model.yaml`, no HNSW backend exists,
and at this scale none is warranted. Following the instruction would have
changed nothing, silently, while leaving you believing the index had been
swapped.

What is actually true: two backends exist, both **exact**, and they
return identical results (`test_numpy_backend_matches_faiss_exactly`).
NumPy is the default and needs no native library. FAISS `IndexFlatIP` is
opt-in via `TemporalReuseIndex(vector_backend="faiss")` and is safe on
Linux, where the `faiss-cpu`/`torch` OpenMP collision that forced the
NumPy default on Windows does not arise.

Neither is a bottleneck. Exact search over 35,000 × 512 vectors, grown
one claim at a time in the `query_then_add` pattern the index actually
uses, measures at **~0.8 minutes** end to end. Leave it on the default.

> This was worth measuring rather than assuming. The same benchmark, run
> against the index as it stood before Phase 9, extrapolated to **~47
> minutes** — the backend kept its rows in a list and re-stacked the
> whole index on every query, which is quadratic in exactly this access
> pattern and was invisible at `dev` scale. `eval`'s ablations rebuild
> the index repeatedly, so that cost would have been paid many times
> over. Fixed, and pinned by a structural test rather than a timing one.

---

## 2. Build the corpus (~2–4 hours, mostly network)

```bash
pramaan data-full 2>&1 | tee logs/data_full.log
```

**What it does:** simulates 35,000 claims, fetches ABO and GenImage
sources (cached under `data/raw/`, resumable), applies per-class
transforms, reconciles and verifies all four split constraints, and
writes a manifest with per-claim provenance.

**What it will actually ask the network for.** Pool sizes are derived
from *distinct image groups*, not claim count — sources are drawn
without replacement, so under-provisioning would manufacture duplicate
"legit" claims that P3 then reports as reuse. Computed by running the
ledger simulation for the `full` tier (no network needed to reproduce
this table — it is `simulate_ledger` plus the sizing arithmetic in
`build_bench.py`):

| Quantity | `dev` (built) | `full` (to build) |
|---|---|---|
| Claims | 3,000 | 35,000 |
| Distinct non-synthetic image groups | 2,626 | **30,662** |
| ABO images downloaded | 2,889 | **33,703** |
| GenImage "Real" images | 1,213 | **12,284** |
| GenImage per AI family (min) | 75 | **975** × 7 families |
| Synthetic-fraud claims | 144 | 2,052 |

So the `full` build needs roughly **12× the source imagery** of `dev`.
`data/raw/` currently holds 602 MB for `dev`; budget **6–8 GB** for the
`full` pools and **15–25 GB** for `data/` overall once the corpus and
cached features are written.

The ~33,700 ABO fetches are individually small (~16 KB each) but numerous
— this is the step most likely to be slow on a poor connection, and it is
the reason the estimate below is dominated by network rather than CPU.

**The pre-fetch stages have already been run at `full` scale on the dev
machine**, up to the point where the download starts. So these are
measured expectations, not guesses — anything materially different is
worth stopping for:

```
INFO simulating ledger: n_claims=35000 merchants=3 seed=1337
INFO reconciling splits
INFO Reconciliation: dropped 39 claim(s) (0.111%) from 28 cross-split
     component(s); 35000 -> 34961.
INFO split verification: OK: all 4 split constraints hold.
INFO fetching source images (abo=33703, genimage>=975/family)
```

**Check when it finishes:**

- [ ] `split verification: OK: all 4 split constraints hold.`
- [ ] `Reconciliation: dropped 39 claim(s) (0.111%)` — expect exactly
      this, since the ledger is seeded. An earlier draft of this document
      told you to expect "well under 0.1%", which the real run would have
      tripped on a non-problem. A *much* larger number would mean the
      entity matcher is over-merging at scale; 39 is the irreducible
      residue described in `docs/LIMITATIONS.md`.
- [ ] `abo=33703, genimage>=975/family` — the derived pool sizes.
- [ ] final corpus is 34,961 claims, fraud prevalence ≈ 0.15
- [ ] Disk: expect ~15–25 GB under `data/`.

**If the GenImage fetch stalls:** it reads ~100 MB parquet row groups
over HTTP and retries transient failures. It is resumable — re-run the
same command and it continues from the cache.

---

## 3. Train (~1–2 hours, GPU-accelerated for CLIP)

```bash
pramaan train --scale full 2>&1 | tee logs/train_full.log
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
pramaan certify --scale full 2>&1 | tee logs/certify_full.log
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
pramaan eval --scale full 2>&1 | tee logs/eval_full.log
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
pramaan report --scale full
pramaan report --scale full --check   # must pass
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
| Tests | 557, all passing |
| Lint / types | ruff + mypy clean |
| CI | `ci.yml` and `leakage.yml` green |
| `dev` corpus | 3,000 claims, all four split constraints verified |
| `dev` certificate | nothing certified — correctly, on power grounds |
| Entry point | installed `pramaan` script, verified from outside the source tree |
| `pramaan all` | runs all five stages (it did not, until Phase 9 — see `LIMITATIONS.md`) |
| Clean install | wheel built and installed into an empty venv; every subcommand runs from a directory with no source tree |
| Determinism | full `dev` rebuild → retrain → recertify → re-eval reproduced `metrics.json` and `certificate.json` byte-identically, apart from wall-clock |
| Reuse index at scale | measured, not assumed: ~0.8 min for 35,000 × 512 exact, `query_then_add` |
| Entity resolution at scale | ~5 s at 35,000 (was ~13 min); `dev` corpus rebuilds byte-identically |
| `full` pre-flight | ledger, reconciliation and split verification **actually run at 35,000**; only the download is untried |
| Blocking on | this document |

### Pre-transfer sweep

Everything below was run against this commit before handoff. At `full`
scale, everything up to the source download has been executed too — the
ledger simulation, split reconciliation and all four split-constraint
checks — so the only genuinely untried part is the download itself and
what follows it.

| Check | Result |
|---|---|
| `pramaan all --scale dev`, end to end | exit 0, all five stages |
| `dev` determinism | `metrics.json` + `certificate.json` byte-identical across a full rebuild |
| `dev` corpus determinism | manifest byte-identical after the entity-resolution rewrite |
| `pramaan all --scale smoke` | exit 0, README correctly left untouched |
| Wheel contents | ships `pramaan`, `benchmarks`, `eval` + console script |
| Clean-venv install from wheel | all runtime modules import outside the repo |
| Config ↔ code consistency | pinned by `tests/test_config_matches_code.py` |
| Dockerfile ↔ declared packages | pinned by `tests/test_entry_point.py` |
| Linux portability | no platform branches, no `os.path.join`, pathlib throughout |
| Doc links and anchors | 63 resolve |
| Image licence allowlist | clean |
| Tests | 557, 0 failures |

**Five defects this sweep found and fixed** are written up in
[`LIMITATIONS.md`](LIMITATIONS.md): a config file nothing read that this
runbook told you to edit, two separate quadratics (the reuse index and
entity resolution, together ~60 minutes of avoidable work at `full`
scale), a documented `make` chain that trained at the wrong scale and
skipped `certify` entirely, and a Dockerfile that silently rebuilt the
packaging bug. Four would have cost VM time; one would have cost trust.

**Before the first full run, install from a clean environment and check
the console script works there**, rather than relying on a checkout:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,provenance]"
cd /tmp && pramaan setup && cd -   # must pass from outside the repo
```

This is not ceremony. The wheel shipped without `benchmarks` and `eval`
for eight phases, and nothing noticed because every invocation happened
from the repo root.
