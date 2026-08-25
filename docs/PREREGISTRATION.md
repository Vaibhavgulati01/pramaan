# Pre-registration

> **Status: skeleton — not yet committed for real.** This document is
> only meaningful once written *before* results are observed. It is
> filled in for real at the end of Phase 5 and MUST NOT be edited after
> `full`-scale results exist, except to append a dated "actuals vs.
> prediction" section. `dev`-scale mechanism checks (Phase 6) do not
> count as "results observed" for this document's purposes — see
> `docs/EVALUATION_PROTOCOL.md`'s pre-registration discipline section for
> why that line is drawn where it is.

## What will be pre-registered here (Phase 5)

1. **Ablation predictions** — predicted direction and rough magnitude for
   every ablation (leave-one-pillar-out x4, cascade-off, calibration-off,
   monotone-off, risk-control-off, federated-index-off). The spec's own
   expectation — pixel-model ablation costs little, reuse-graph ablation
   costs a lot — is a hypothesis at this point, not a result; it gets
   registered here like any other prediction.
2. **The α/δ decision ladder**, pre-committed before the `full` denial set
   is known: attempt (α=0.03, δ=0.10); if uncertifiable, walk (0.05,
   0.10) -> (0.10, 0.10) -> (0.10, 0.20); report the first pair that
   certifies plus every failed attempt. If nothing certifies, that is the
   published result, with the required n stated (see
   `src/pramaan/risk/power_analysis.py`, Phase 0.5). If the power
   analysis implies `full` would need to exceed ~60k claims to certify
   α=0.03, α=0.05 is the pre-declared primary target instead.
3. **The federation kill-gate criteria** (bounded to two implementation
   attempts in Phase 7): a directionally-correct non-zero Δ in ring
   recall from the on/off ablation, and a poisoning self-test where
   first-seen immunity blocks the attack and the same attack succeeds
   with the rule disabled.

## Actuals vs. prediction (Phase 9, after the one `full`-scale unsealing)

*(Appended, never edited into the predictions above.)*
