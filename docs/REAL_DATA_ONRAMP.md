# Real data onramp

> **Status: skeleton.** Filled in as `simulate_ledger.py` (Phase 1) and
> the claim/behavioural schema stabilise.

## What will go here

The exact CSV/table schema a merchant would hand over to replace
`simulate_ledger.py`'s simulated claim ledger with real data, plus a
one-command switch (`configs/data.yaml: source: real|simulated`) to use
it. Column-by-column: type, nullability, and which pillar/feature
consumes it, so the swap is mechanical rather than a re-architecture.
