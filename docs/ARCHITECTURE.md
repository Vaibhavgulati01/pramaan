# Architecture

> **Status: skeleton.** Filled in as each layer lands (Phases 1-7); the
> layer diagram and full walkthrough are finalised in Phase 9.

Mirrors `PRAMAAN_v2_architecture.md` §3-4. Will contain, per layer:

- **L0 Ingest & canonicalisation** — `src/pramaan/ingest/` (Phase 1)
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
