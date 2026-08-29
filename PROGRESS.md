# Build progress

> Dev-facing build tracker, not a judged deliverable — updated as each
> piece lands. For the polished submission docs, see `README.md` and
> `docs/`. For the full phase rationale (why things are ordered/scoped
> this way), see `PRAMAAN_v2_architecture.md` and the phase descriptions
> below, which summarize the approved implementation plan.

**Repo**: https://github.com/Vaibhavgulati01/pramaan (public, CI green)
**Environment**: Windows dev machine, Python 3.13 venv (`.venv/`), CPU-only.
A separate VM (64GB RAM/16GB VRAM) is used later for the `full`-scale run
only (Phase 9) — everything built here targets `smoke`/`dev` scale and is
portable to that VM unmodified.

## Phase status

| Phase | Status | Summary |
|---|---|---|
| 0 — Scaffold + push | ✅ done | Repo tree, Typer CLI, GH Actions CI, Docker, docs skeletons |
| 0.5 — Power analysis | ✅ done | `risk/hb_pvalue.py`, `risk/power_analysis.py`; sized `full`=35,000 claims (α=0.03/δ=0.10 conservative), `dev`=3,000 |
| 1 — Foundations | ✅ done | Canonicalisation, leakage audit, benchmark builder, simulated ledger, verified splits |
| 2 — Pillars (P3→P2→P4→P1) | ✅ done | All four pillars + rings + cost-ordered cascade; temporal-leak test verified against an injected bug |
| 3 — Fusion & calibration | ✅ done | LightGBM + monotone constraints, Mondrian isotonic, reliability diagrams |
| 4 — Risk control | ✅ done | LTT + certified sets, calibration seal, GUARANTEE.md with all three caveats |
| 5 — Policy & OPE | ✅ done | Cost model, selective policy, DR-OPE, ε-exploration, label maturity; PREREGISTRATION + EVALUATION_PROTOCOL committed |
| 6 — Evaluation infra (smoke+dev) | ✅ done | Baselines, ablations (×2 subsets), negative controls, bootstrap CIs, README generator, 2 CI gates |
| 7 — Audit, federation, monitoring | ✅ done | Reason codes, signed evidence packs, federated index with all 4 anti-poisoning rules measured, guarantee watchdog. **Kill-gate passed.** |
| 8 — Product surface | ✅ done | FastAPI (/adjudicate /explain /healthz), hardened multi-stage container |
| 9 — Docs, VM handoff, README | 🚧 **blocked on the VM run** | All docs written; `docs/VM_HANDOFF.md` is the next action |

## What's implemented right now

- **CLI** (`src/pramaan/cli.py`): every command is real —
  `setup / data / data-full / train / certify / eval / report / serve / all`.
- **Ingest / canonicalisation** (`src/pramaan/ingest/`): phone, email,
  address (transliteration + PIN-bucketed fuzzy + numeric-token rule),
  device (feature only, not identity), identity resolution, and
  forensics-preserving image normalisation.
- **Benchmark builder** (`benchmarks/`): real ABO + GenImage sourcing
  with on-disk caching, per-fraud-class image transforms, the simulated
  claim ledger, split reconciliation + verification, and a manifest
  carrying per-claim provenance.
- **Leakage audit** (`eval/entity_leakage_audit.py`): CLI + library,
  fails loudly on cross-split identity overlap.
- **Risk core**: `src/pramaan/risk/hb_pvalue.py` (Hoeffding-Bentkus
  p-value), `src/pramaan/risk/power_analysis.py` (corpus sizing from the
  certification power requirement). Property-tested + hand-checked.
- **Configs**: `configs/{data,model,costs,risk}.yaml` — `data.yaml` and
  `risk.yaml` now carry real, derived numbers (not placeholders).
- **CI**: `.github/workflows/ci.yml` — lint (ruff) + type (mypy) + tests
  (pytest) + CLI smoke, all green on every push.
- **Tests**: 542 across ingest, benchmarks, splits, pillars, cascade, fusion, risk, policy, eval, audit, federation, monitoring, API and CLI.

## Phase 1 checklist (current)

- [x] `src/pramaan/ingest/phone.py` — India phone canonicalisation
- [x] `src/pramaan/ingest/email.py` — Gmail-family canonicalisation
- [x] `src/pramaan/ingest/address.py` — NFKC + transliteration + abbrev + PIN-bucketed fuzzy match
- [x] `src/pramaan/ingest/device.py` — UA/screen/timezone/font hash, instability caveat documented
- [x] `src/pramaan/ingest/identity.py` — cross-signal entity resolution (identity clustering), on `src/pramaan/common/union_find.py`
- [x] `src/pramaan/ingest/image.py` — raw_bytes/decoded_array/exif_blob kept separate, no re-encode round-trip
- [x] `eval/entity_leakage_audit.py` — asserts zero canonical-identity overlap across splits, fails loudly
- [x] `benchmarks/sources.py` — real data acquisition (ABO via HF streaming; Tiny-GenImage via `hf://` parquet row-group reads, no full-file download), local caching under `data/raw/` (gitignored, never committed)
- [x] `benchmarks/transforms.py` — fraud-class image transforms (messaging-app degradation, recycle crop/rotate/recolour, screen-rephotograph moiré, metadata-inconsistent thumbnail/QT desync)
- [x] `benchmarks/simulate_ledger.py` — simulated claim ledger + Pillar 4 source, declared-simulation docstring, cohort-consistent-by-construction split design
- [x] `benchmarks/splits.py` — `verify_splits()` (all 4 constraints) + `reconcile_splits()` (structural enforcement); see the design note below
- [x] `benchmarks/build_bench.py` — orchestrates ledger+sources+transforms into a corpus, manifest with per-claim provenance/licence/source-SHA256/output-SHA256
- [x] `pramaan data --scale {smoke,dev}` and `pramaan data-full` wired to `build_bench.py`
- [x] `benchmarks/loaders.py` — timestamp-ordered corpus reader, so the temporally-correct iteration order is the default one
- [x] Real `smoke` build end-to-end (200 claims, 3m27s cold / 7s warm), split verification green on actual output
- [x] Real `dev` build (3,000 claims, 14.8% fraud prevalence), split verification green
- [x] **Gate**: leakage audits green — enforced in CI by `.github/workflows/leakage.yml` (5 seeds, network-free) and by `verify_splits()` hard-failing the build

### Three real bugs Phase 1 surfaced

Recording these because each was invisible until something concrete ran,
and each would have quietly corrupted downstream numbers:

1. **Fuzzy address matching merged different households** (caught by
   `verify_splits`) — fixed with a numeric-token rule, plus structural
   reconciliation for the irreducibly ambiguous residue. See below.
2. **Image size predicted the label** (caught by the first real corpus
   build) — only the legit class was being resized, leaving synthetic
   fraud at 512px against ~256px elsewhere. Fixed by splitting
   construction into fraud-transform + class-independent transport.
   Guarded by `test_image_dimensions_carry_no_label_signal`.
3. **`pramaan all` passed Typer sentinels as argument values** (caught by
   CI) — invisible locally because `data` was only ever invoked through
   the CLI, and invisible before Phase 1 because downstream commands
   ignored their arguments.

Also: the spec's SD 1.4 generator family does not exist in the public
mirror (it declares the label but carries no rows). Substituted SD 1.5
and declared it in `docs/DATA_CARD.md` / `docs/LIMITATIONS.md`.

### Design note: the 4-way splits, and a real bug this caught

`simulate_ledger.py` assigns each claim's split cohort *first* and
generates its timestamp/identity-reuse/image-reuse to be consistent with
that cohort — so the corpus is leakage-free **by construction**.
`splits.py` was written to verify that rather than trust it.

It did not hold. Verification found entity-leakage violations caused by
`addresses_match` merging two genuinely different households that shared
a street and locality inside one PIN code (`H.No. 97 Chanda Marg,
Faridabad` ≈ `H.No. 51 Garg Marg, Faridabad`). `token_set_ratio` is
deliberately forgiving about differing tokens, which is exactly wrong
for addresses where the house number is the whole distinguishing signal.

Two fixes, and the second is the important one:

1. `addresses_match` now also requires numeric tokens to intersect when
   both sides have them (regression-tested against the observed pairs).
2. That is **not sufficient** — a residual case (`H.No. 06 Tata Marg` vs
   `H.No. 06 Sarraf`, same PIN, same number, different street) is
   genuinely ambiguous and no threshold fixes it. So `reconcile_splits()`
   now enforces one-split-per-entity+ring-component structurally and
   reports the claims it drops (~0.03% at n=3000) in the manifest.

Verified clean across 7 seeds at `dev` scale. Written up in
`docs/LIMITATIONS.md` and `docs/DATA_CARD.md` rather than quietly fixed.

## Phase 2 checklist (current)

- [x] `src/pramaan/pillars/p3_reuse.py` — pHash + LSH banding, CLIP semantic stage, strictly time-ordered index
- [x] `src/pramaan/pillars/clip_embed.py` — CLIP ViT-B/32 embeddings (quickgelu variant, matching the OpenAI weights)
- [x] **The temporal-leak test** (`tests/test_pillars_p3_temporal.py`) — and verified it fails on a deliberately leaked index, rather than trusting a green run
- [x] Ring detection (`rings.py`) — temporal bipartite `claimant ↔ image_cluster` graph with first-seen immunity; 116 rings found in the dev corpus
- [x] `p2_forensics.py` — QT tables, thumbnail consistency, ELA, DCT, FFT (41.5 ms/claim, inside the ~40ms stage-2 budget)
- [x] `p4_behaviour.py` — claimant aggregates over the simulated ledger, keyed by **canonical identity** so a fresh account per claim doesn't read as N first-timers
- [x] `p1_provenance.py` — C2PA with graceful `UNKNOWN` (400/400 real dev claims report UNKNOWN, as the WhatsApp reality demands)
- [x] `SAFETY.md` written properly — every claim names the file or command that verifies it
- [x] `cascade/` — cost-ordered orchestration with early exit, and feature assembly (all four pillars + rings → one fixed-schema vector per claim)

### Cascade: machinery verified, early exit inert until Phase 3

Measured over the full 3,000-claim dev corpus:

| | |
|---|---|
| Mean compute | **48.5 ms/claim** (median 46.8, p95 80.3) |
| Stage-exit distribution | **0% / 0% / 100%** — nothing exits early |
| Index completeness | 3000/3000 |
| Rings detected | 110 |

The 0% is honest, not a bug. The interim scorer is a hand-specified
placeholder (a trained model doesn't exist until Phase 3); it anchors at
0.5 and moves in bounded steps, so ordinary claims land mid-band and
never clear the exit thresholds. **I deliberately did not narrow the band
to manufacture exits** — early exit is only safe when the interim score
is well-calibrated, which is precisely what Phase 3 supplies and Phase 5
tunes against the cost model. So 48.5 ms/claim is the no-early-exit upper
bound, and `0/0/100` is the number to report until then.

Two design points the cascade had to get right, both silent when wrong:

- **Early-exiting claims are still indexed.** Skipping the *use* of a
  claim's reuse features must never skip *recording* it, or a fraudster
  whose first submission exits early makes their own recycled follow-up
  undetectable. Indexing uses pHash only (~12ms) and skips the CLIP
  embedding (~90ms) — a stated trade, not a silent one. Guarded by a test
  I verified fails when the bug is re-introduced.
- **Skipped pillars leave `NaN`, not `0.0`.** A zero is a measurement;
  LightGBM reads NaN as genuinely missing.

### P3 thresholds are measured against ground truth, and measuring overturned two guesses

Legit claims contain no reuse by construction, so any match on one is a
false positive — which makes the legit flag rate directly measurable on
the 3,000-claim dev corpus:

| rule | legit FP | reuse recall |
|---|---|---|
| pHash≤6 **or** CLIP≥0.92 *(first guess)* | 20.2% | 72.7% |
| pHash≤2 **or** CLIP≥0.96 | 4.1% | 48.7% |
| **pHash≤2 or CLIP≥0.98** *(chosen)* | **1.5%** | **42.0%** |
| pHash≤2 alone | 1.3% | 39.9% |
| CLIP≥0.98 alone | 0.5% | 38.2% |

Two things this overturned, both wrong when guessed from pairwise
statistics:

1. **pHash, not CLIP, was the dominant false-positive source.** `pHash≤6`
   alone flags 6.6% of the legit class. ABO is a product catalogue and
   pHash is a low-frequency hash, so different white-background products
   hash alike. Per-pair rates look negligible then compound across ~1,500
   priors per claim.
2. **CLIP needs a far higher bar than semantic intuition suggests.** Two
   different white sneakers sit near 0.92; *instance* identity needs 0.98.

Both stages still earn their place — together 42.0% where each alone
reaches ~39%.

### P3 emits graded evidence, not just a verdict

The binary flag is tuned for precision, so crop/rotate/recolour reuse
mostly fails it (4.8% flagged). But that reuse is *not* invisible:

| class | median best-Hamming | median best-CLIP |
|---|---|---|
| `fraud_catalog_photo` | 0 | 0.996 |
| `fraud_recycled_prior_claim` | 12 | **0.923** |
| `legit_real_photo` | 16 | **0.864** |

Recycled claims are clearly separated from legit ones on the continuous
features even where no threshold cleanly divides them. Reporting only a
boolean would have thrown that away, so `ReuseFeatures` carries
`best_hamming` / `best_clip_similarity` over every candidate examined and
lets the fusion model (Phase 3) combine them with the other pillars.

### Vector backend: NumPy by default, FAISS opt-in

`IndexFlatIP` is a brute-force matmul, so NumPy is algorithmically
identical (verified: same top-k, 2.4e-07 max difference, same speed) —
but `faiss-cpu` and `torch` each bundle an OpenMP runtime and abort on
import together on Windows (`OMP: Error #15`), and the documented
workaround is one its own authors say may "silently produce incorrect
results". FAISS stays opt-in for `full` scale on Linux, where HNSW
actually matters. A test asserts both backends agree.

## Phase 3 checklist (current)

- [x] `fusion/schema.py` — 48-feature typed schema, versioned, with 11 monotone constraints each carrying a written justification
- [x] `fusion/calibration.py` — Mondrian isotonic with shrinkage, equal-mass ECE/MCE, per-group reporting
- [x] `fusion/model.py` — LightGBM (single-threaded, deterministic) + cross-fitted calibration, split discipline enforced in code
- [x] `fusion/pipeline.py` + `pramaan train` — cascade over the corpus, cached feature matrix
- [ ] Reliability diagrams rendered to `reports/dev/`

### Split discipline: why isotonic is cross-fitted

Isotonic calibration needs held-out predictions, but the calibration
split is reserved for Learn-then-Test and spending it here would void the
guarantee. So the calibrator is fitted on **K-fold out-of-fold
predictions from the train split**, leaving `calibration` untouched for
Phase 4 and `test` sealed until Phase 6. `FusionModel.fit` *raises* if
handed rows from either — enforced by code, not by memory.

### Mondrian calibration reproduces the failure the spec predicted

Measured on dev out-of-fold predictions:

| | ECE |
|---|---|
| Overall | **0.0099** |
| `sports\|high` | 0.0754 |
| `beauty\|high` | 0.0702 |
| `electronics\|high` | 0.0694 |

Calibration looks excellent globally and is ~7× worse on every high-price
cell — precisely §4 L2's "well-calibrated overall and badly calibrated on
high-value electronics is a system that loses money exactly where money
is." Per-group reporting is what makes that visible; a single global
reliability curve would hide it entirely.

Calibration also does real work: out-of-fold Brier improves 0.1066 →
0.0884 (−17%).

### Two problems found by inspecting the fitted model

1. **A feature measuring the corpus, not the claim.**
   `reuse_n_candidates_examined` counted prior claims in the index, so it
   grew 691 → 1597 → 2041 across train/calibration/test — a near-perfect
   proxy for split membership, drawing 10.2% of gain. Removed (schema
   1.1.0). It was *already* flagged in a comment as "an artifact of index
   size, not of the claim"; noting a hazard is not the same as excluding
   it.
2. **The corpus could not test its own central claim.** Forensics took
   **72.6%** of gain — the opposite of the spec's prediction — with
   `fft_peak_ratio` top at 16.7% despite synthetic images scoring *lower*
   on it than legit (0.66×). Tracing that led to the real problem: every
   non-synthetic claim came from ABO and every synthetic one from
   GenImage, so the GenImage-sourced set and the AI-generated set were
   **literally identical**. "Is it AI-generated?" and "did it come from
   GenImage?" were the same question, and the forensics pillar was
   free to separate the classes by reading encoding provenance.

   §6's headline ablation was therefore **untestable** — any answer would
   have been an artifact. Fixed by mixing the real-photo pool so
   non-synthetic claims draw from ABO *and* GenImage's own real class,
   decorrelating source from label. Guarded by
   `test_source_dataset_does_not_predict_the_label`; written up in
   `docs/LIMITATIONS.md`.

   This one is worth dwelling on: the corpus passed every leakage audit,
   every split constraint, and every unit test. It was only inspecting
   *which features the fitted model leaned on* that exposed it.


## Phase 3 result: the ablation that changed what we can claim

Retraining on the decorrelated corpus **did not** move forensics gain
(72.6% -> 70.9%). I had predicted it would drop; it did not, and chasing
that down produced the most important finding in the project so far.

Three measurements, each contradicting the previous:

1. **Gain says forensics dominates** — `forensics 70.9% / reuse 19.2%`.
2. **Effect sizes say forensics has no label signal** — `d(fraud vs
   legit)` is −0.01 to 0.16, while `d(GenImage vs ABO)` reaches 1.83.
   These are dataset detectors.
3. **The ablation reconciles them, and quantifies the confound:**

   | ablation | full corpus | ABO-only (source constant) |
   |---|---|---|
   | − forensics | **−0.1024** PR-AUC | **−0.0335** |
   | − reuse | −0.0220 | −0.0273 |

   With source held constant the pillars contribute *comparably*.
   Roughly two-thirds of forensics' apparent lead is it detecting
   GenImage, which carries **2.58×** the fraud rate of ABO.

**The residual confound is structurally irreducible.** Writing `q` for
the GenImage share among non-synthetic claims,
`P(GenImage|fraud) = (144 + 301q)/445` and `P(GenImage|legit) = q` are
equal only at `q = 1` — i.e. only by abandoning ABO entirely. Any corpus
sourcing AI fraud from one dataset and real photos partly from another
inherits this.

**Consequences, now written into `docs/LIMITATIONS.md`:**

- Gain-based importance is a diagnostic here, **not** evidence of
  predictive contribution. A feature that partitions subpopulations helps
  tree structure without predicting the label. This is exactly why §6
  mandates ablations over importance scores — we hit the reason directly.
- The headline ablation is an **upper bound** on the pixel pillars.
  Phase 6 will report full, ABO-only, and generator-holdout variants
  rather than one number.

## Phase 4 note: a power failure that would have silenced the certificate

Fixed-sequence LTT tests thresholds most-conservative-first, but those
deny the fewest claims and so have the *least* power. Observed: a good
model failed at t=0.99 on 60 denials at 3.3% empirical FDR — comfortably
below alpha=0.10 — because 60 observations cannot establish it. The naive
sequence stops there and certifies nothing, despite thresholds further
down denying 600 claims at 1.2%.

Fixed by treating insufficient power as a *skip*, not a failure, with the
floor from `min_n_for_rhat` at a pre-committed planning rate — the same
Phase 0.5 analysis that sized the corpus. This is legitimate because
`n_denied` depends only on predicted probabilities, never on labels, so
filtering the grid by it uses no information about the risk being tested.


## Phase 4 result: the certificate correctly refused to exist

`docs/GUARANTEE.md` was written **before** any certified α was computed —
visible in the git history, and the point: caveats written after seeing a
pleasing number are marketing.

Then `pramaan certify --scale dev` walked the whole pre-committed ladder
and **certified nothing at any rung** (α=0.03, 0.05, 0.10 at δ=0.10, and
0.10 at δ=0.20). That is the correct answer, not a defect.

The dev calibration split (n=580) looks like this:

| threshold | denied | realised FDR |
|---|---|---|
| 0.90 | 18 | **0.000** |
| 0.80 | 24 | 0.167 |
| 0.50 | 34 | 0.382 |

The model has a genuinely clean high-confidence region — zero false
denials at t=0.90 — but only 18 claims reach it, far below every power
floor (45–222). Loosening the threshold buys denials at rapidly worse
FDR. **Certification is blocked by sample size, not model quality**,
which is exactly the distinction `certified_set.py`'s power floor exists
to draw, and exactly what Phase 0.5 predicted would happen at dev scale.

### One planning assumption now needs revising, flagged before the full run

Phase 0.5 sized the corpus assuming `deny_rate = 0.075` (half of
prevalence). The **observed** rate in the high-confidence region is
`18/580 = 3.1%` — about **2.4× lower**. Projected onto `full` (~7,000
calibration claims) that gives ~217 denials at t≈0.90, which clears
α=0.03's zero-error floor of 76 but not its 0.3α floor of 222.

So α=0.03 at full scale is reachable if the near-zero FDR holds and
marginal if it does not. Genuinely open — and flagged now rather than
discovered at Phase 9.

### Calibration-split single-use is enforced, not asserted

- `FusionModel.fit` raises `SplitDisciplineError` on calibration/test rows.
- The split's content hash is committed (`reports/dev/calibration_seal.json`)
  and re-checked in CI; `write_seal` refuses to overwrite a changed seal.
- Verified end-to-end: an unchanged split passes, a single flipped label
  is rejected, and a seal copied between tiers is caught.


## HANDOFF POINT — everything local is done

Phases 0–8 are complete. Phase 9 needs the `full`-scale run, which needs
the VM. **[`docs/VM_HANDOFF.md`](docs/VM_HANDOFF.md) is the complete
runbook** — commands, expected runtimes, what to check at each step, and
the three publishable outcomes of certification.

### Why the VM is genuinely required

Not convenience. At `dev` scale the calibration split held 580 claims and
only 18 high-confidence denials, against the 45–222 the power analysis
requires. **Nothing certified, and that was correct.** The headline claim
is currently *unproven, not disproven*, and `full` (35,000 claims, ~7,000
calibration) is the run that settles it.

### What the VM run produces

| Step | Command | ~Time |
|---|---|---|
| Corpus | `pramaan data-full` | 2–4 h |
| Train | `pramaan train --scale full` | 1–2 h |
| **Certify** | `pramaan certify --scale full` | 5 min |
| Evaluate | `pramaan eval --scale full` | 2–4 h |
| README | `pramaan report --scale full` | seconds |

### What I still owe afterwards

Genuinely blocked on `full` — each needs numbers that do not exist yet:

1. `PREREGISTRATION.md` actuals — including the misses
2. `reports/SCALE_CONCORDANCE.md` — dev vs full, disagreement as a finding
3. The shift-matrix driver (all eight conditions + the two-experiment
   design exist; wiring targets the frozen test split, which `full`
   creates)
4. A tagged release, once the README carries `full` numbers

**Needs a human, not compute** — these were always out of scope for an
agent and are recorded here so they are not a silent omission:

5. **Hugging Face Spaces deploy.** The FastAPI app (`pramaan serve`) and
   the `Dockerfile` are ready; deploying needs an HF account and a
   token. Nothing in the codebase blocks it.
6. **A UI screen capture.** The terminal recording at the top of the
   README is automated and committed; a browser capture of the served
   interface needs someone at a screen. The demo GIF is deliberately a
   terminal session and not a mocked-up UI.

**Done since, and no longer waiting on the VM:**

- **`pramaan all` now actually runs all five stages.** It had been
  skipping certify, eval and report since Phase 6, via a leftover
  "not implemented yet" stub, with CI green throughout. See
  [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — grouped there with two
  other bugs of the same shape.
- **The installed package works.** The wheel had omitted `benchmarks`
  and `eval`, so `pramaan data` failed for anyone who installed rather
  than cloned. CI now runs the console script from outside the source
  tree so the checkout cannot stand in for the distribution.
- **The test suite no longer starts a web server.** One test invoked
  `pramaan serve`, which blocks on `uvicorn.run()`; whether the suite
  finished depended on port 8000 being occupied. The same test ran a full
  real `dev` pipeline (and rewrote the README) on any machine that had a
  corpus.
- **Demo GIF** at the top of the README: a real recording of
  `pramaan all --scale smoke`, captured and rendered by two committed
  scripts (`vhs`/`asciinema` both need a Unix pty and do not run here).

- README problem statement, with figures attributed to NRF/Happy Returns
  and Appriss/Deloitte — including the fact that those two sources
  **disagree** on the fraud rate (9% vs 15.14%), reported rather than
  averaged away. India RTO figures are labelled vendor-reported because
  no primary source exists for them.
- Architecture diagram (mermaid, renders natively on GitHub), covering
  the cascade, the certificate, and the dotted *nothing-certified →
  auto-deny-disabled* path the system actually took at `dev`.
- `scripts/check_doc_links.py` + CI gate: every relative link and anchor
  in the repo's markdown resolves. It found a dead anchor I had written
  into `FAQ.md`, pointing at a section of `LIMITATIONS.md` that did not
  exist.

## Phases 5–8 summary

**Phase 5** — cost model proving FP > FN at every order value (2.65× at
₹500), selective policy where *the certificate constrains and cost
chooses within it*, DR-OPE, ε-exploration with rupee cost reported,
120-day label maturity. Two bugs found by running it: the cost-optimal
search could not reach its own baselines, and `hb_pvalue` turned out not
to be monotone in n — Hypothesis found the counterexample to a property
test I had written asserting the false property.

**Phase 6** — six baselines, ablations reported on both the full corpus
and the source-controlled subset, negative controls (both PASS), bootstrap
CIs, and a README generator whose output CI diffs against the committed
file.

**Phase 7** — templated reason codes, HMAC-signed evidence packs, the
federated index with all four anti-poisoning rules **measured** (the
poisoning attack is run in both directions), and the guarantee watchdog
that widens the band rather than merely alerting.

**Phase 8** — FastAPI with a response boundary that is a security
control, asserted by `test_response_never_leaks_a_score`. Writing those
tests caught that five of fifteen were silently skipping because
`TestClient` only fires lifespan startup inside a context manager.
