# The guarantee

> **Status: skeleton.** Power analysis lands in Phase 0.5 (this doc gets
> the power curve + sizing rationale first). Illustrative `DEV`-scale
> numbers land in Phase 4. The one real, `full`-scale certified statement
> lands in Phase 9, after the VM run, resolved via the α/δ decision ladder
> in `docs/PREREGISTRATION.md`.

## What will go here

1. **Plain-English statement**, then the formal one: control of the
   false-denial rate among auto-denied claims via Learn-then-Test
   (Hoeffding-Bentkus p-values, fixed-sequence testing over a monotone
   threshold grid).
2. **The power curve**: minimum certifiable denial-set size n vs. α, at
   fixed δ=0.10 (`src/pramaan/risk/power_analysis.py`). This is what the
   corpus was sized from — not the other way round.
3. **Three honesty caveats**, stated before any certified α is reported
   anywhere else in the repo:
   - The selective-risk ratio caveat (conditional `FDR_deny` vs. the
     clean unconditional `E[(1-y)*1{deny}] <= alpha'` variant — both
     reported).
   - The exchangeability assumption, and the pointer to §6's shift matrix
     for exactly how badly it breaks under generator shift.
   - Calibration-set single-use, enforced by a recorded file hash + CI
     assertion that it never changes.
4. **The α/δ decision ladder** actually walked at `full` scale, and which
   rung certified (or the honest "nothing certified at n=X, here is the
   required n" result if none did).
5. Reliability diagrams, global and per-`{category x price-band}` group.

## DEV-scale numbers (mechanism validation only)

*(Phase 4 fills this in. Explicitly labelled `DEV — not a held-out
result`; never a substitute for the `full`-scale statement above.)*
