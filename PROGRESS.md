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
| 1 — Foundations | 🚧 in progress | Canonicalisation, leakage audit, benchmark builder, simulated ledger, splits |
| 2 — Pillars (P3→P2→P4→P1) | 🚧 in progress | P3 (reuse graph) done incl. the mandatory temporal-leak test; P2/P4/P1 next |
| 3 — Fusion & calibration | ⬜ not started | |
| 4 — Risk control | ⬜ not started | `certified_set.py`/`ltt.py` on top of Phase 0.5's `hb_pvalue.py` |
| 5 — Policy & OPE | ⬜ not started | Ends with `PREREGISTRATION.md` + `EVALUATION_PROTOCOL.md` committed |
| 6 — Evaluation infra (smoke+dev) | ⬜ not started | |
| 7 — Audit, federation, monitoring | ⬜ not started | Federation is kill-gated (2 attempts, concrete criteria) |
| 8 — Product surface | ⬜ not started | API, Docker, `vhs`/`asciinema` GIF |
| 9 — Docs, VM handoff, README | ⬜ not started | Blocked on the user running `full` scale on the VM |

## What's implemented right now

- **CLI** (`src/pramaan/cli.py`): `setup`, `data`, `data-full` are real.
  `train/eval/report/serve` exist and are invocable but say "not yet —
  lands in Phase N" (no crashes, no missing commands).
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
- **Tests**: 130+ across ingest, benchmarks, splits, risk, and CLI.

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
- [ ] Real `dev` build (3,000 claims) — running
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
- [ ] Ring detection — temporal bipartite `claimant ↔ image_cluster` graph, claim- and ring-level outputs
- [ ] `p2_forensics.py` — QT tables, thumbnail consistency, ELA, DCT, FFT
- [ ] `p4_behaviour.py` — claimant aggregates over the simulated ledger
- [ ] `p1_provenance.py` — C2PA with graceful `UNKNOWN`
- [ ] `cascade/` — cost-ordered orchestration with early exit
- [ ] `SAFETY.md` written properly (pillars now exist to point at)

### P3 thresholds are measured, not guessed

| | pHash (Hamming) | CLIP (cosine) |
|---|---|---|
| Chosen | **6** | **0.92** |
| Catches | exact / near-exact reuse | crop + rotate + recolour |
| At the chosen value | 0 FPs in 20,000 unrelated pairs | 73.3% recall, 0 FPs in 3,540 unrelated pairs |
| Why not looser | threshold 10 flagged **19.1% of the legit class** — per-pair FP of 0.03% compounds across ~1,500 priors per claim | 0.90 gives 90% recall but 0.06%/pair, which compounds badly |

The two stages are complementary by construction: crop/rotate reuse has a
median pHash distance of 15 and is unreachable at any useful precision,
which is exactly why the spec's design has a semantic second stage.

### Vector backend: NumPy by default, FAISS opt-in

`IndexFlatIP` is a brute-force matmul, so NumPy is algorithmically
identical (verified: same top-k, 2.4e-07 max difference, same speed) —
but `faiss-cpu` and `torch` each bundle an OpenMP runtime and abort on
import together on Windows (`OMP: Error #15`), and the documented
workaround is one its own authors say may "silently produce incorrect
results". FAISS stays opt-in for `full` scale on Linux, where HNSW
actually matters. A test asserts both backends agree.
