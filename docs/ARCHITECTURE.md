# Architecture

> **Status: skeleton.** Filled in as each layer lands (Phases 1-7); the
> layer diagram and full walkthrough are finalised in Phase 9.

Mirrors `PRAMAAN_v2_architecture.md` §3-4. Will contain, per layer:

- **L0 Ingest & canonicalisation** — `src/pramaan/ingest/` ✅ Phase 1
- **L1 Evidence pillars** (P1-P4) + cascade — `src/pramaan/pillars/`,
  `src/pramaan/cascade/` (Phase 2)
- **L2 Fusion + calibration** — `src/pramaan/fusion/` (Phase 3)
- **L3 Risk control** — `src/pramaan/risk/` (Phase 0.5, Phase 4)
- **L4 Cost-optimal policy + OPE** — `src/pramaan/policy/` (Phase 5)
- **L5 Audit trail** — `src/pramaan/audit/` (Phase 7)
- **L6 Federated hash index** — `src/pramaan/federated/` (Phase 7)
- **L7 Monitoring** — `src/pramaan/monitoring/` (Phase 7)
- **API surface** — `src/pramaan/api/` (Phase 8)

Each section will link the module(s) that implement it and the tests that
verify its non-negotiables (e.g. the temporal-leak test for P3).

## L0 — Ingest & canonicalisation (landed, Phase 1)

The layer that decides whether L3's guarantee means anything: bad
canonicalisation silently leaks entities across splits, and a
contaminated test set makes the certificate fiction.

| Module | Responsibility |
|---|---|
| `ingest/phone.py` | `+91`/leading-`0`/separator stripping → 10-digit key |
| `ingest/email.py` | lowercase; Gmail-family dot- and `+tag`-stripping |
| `ingest/address.py` | NFKC → Devanagari transliteration → abbreviation expansion → stopword removal; fuzzy match **blocked within an exact PIN bucket** and requiring numeric-token overlap |
| `ingest/device.py` | UA+screen+timezone+font-list hash — a P4 *feature*, deliberately **not** an identity signal (see `docs/LIMITATIONS.md`) |
| `ingest/identity.py` | union-find over phone/email/address edges → canonical identity per claim |
| `ingest/image.py` | keeps `raw_bytes` / `decoded_array` / `exif_blob` strictly separate; never re-encodes a claim's own bytes |

Verified by `eval/entity_leakage_audit.py` (zero canonical-identity
overlap across splits) and `benchmarks/splits.py` (all four split
constraints), both of which the corpus build hard-fails on.

## Benchmark construction (Phase 1)

`benchmarks/` is the corpus builder, not part of the serving path:

```
simulate_ledger  ->  sources (ABO / GenImage)  ->  transforms
                                                       |
        manifest.json + claims.csv  <-  verify  <-  reconcile
```

- `sources.py` — real image acquisition, cached, never redistributed
- `transforms.py` — per-fraud-class image construction
- `simulate_ledger.py` — the synthetic claim ledger (declared simulation)
- `splits.py` — `reconcile_splits()` then `verify_splits()`
- `build_bench.py` — orchestration + manifest
