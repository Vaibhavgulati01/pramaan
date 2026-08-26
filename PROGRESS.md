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
| 2 — Pillars (P3→P2→P4→P1) | ⬜ not started | |
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
- [ ] Run the real `smoke` and `dev` builds end-to-end and confirm split verification is green on actual output
- [ ] `benchmarks/loaders.py` — a `datasets`-compatible loader over the built corpus (deferred to Phase 2, where the pillars are its first consumer)
- [ ] **Gate**: leakage audits green before any Phase 2 model code

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

## Next up after Phase 1

Phase 2 (pillars), P3 (reuse graph) first per the plan — pHash + LSH
banding, CLIP+FAISS with a strictly time-ordered index, and the
temporal-leak unit test the spec calls out as a must-have.
