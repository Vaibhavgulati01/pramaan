# The guarantee

> **Status: partial.** Phase 0.5 (power analysis / sizing) is done and
> real — see below. Illustrative `DEV`-scale numbers land in Phase 4. The
> one real, `full`-scale certified statement lands in Phase 9, after the
> VM run, resolved via the α/δ decision ladder in
> `docs/PREREGISTRATION.md`.

## What will go here (Phases 4, 9)

1. **Plain-English statement**, then the formal one: control of the
   false-denial rate among auto-denied claims via Learn-then-Test
   (Hoeffding-Bentkus p-values, fixed-sequence testing over a monotone
   threshold grid).
2. **Three honesty caveats**, stated before any certified α is reported
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
3. **The α/δ decision ladder** actually walked at `full` scale, and which
   rung certified (or the honest "nothing certified at n=X, here is the
   required n" result if none did).
4. Reliability diagrams, global and per-`{category x price-band}` group.

## The power curve (Phase 0.5 — done)

*The dataset was sized from the guarantee's power requirement, not the
other way round.* `src/pramaan/risk/power_analysis.py` (property-tested
in `tests/test_risk_power_analysis.py`) answers: for a target (α, δ), how
many *denied* claims does a test set need to contain before certification
is even mathematically possible, and — given assumptions about deny rate
and test-split fraction — how big does the full corpus need to be to
plausibly produce that many?

**Minimum certifiable denial-set size, δ=0.10 fixed:**

| α | r̂=0 (zero-error floor) | r̂=0.3·α (model comfortably clears the bar) | r̂=0.5·α (model just barely clears the bar) |
|---|---|---|---|
| 0.03 | 76 | 222 | 489 |
| 0.05 | 45 | 131 | 288 |
| 0.10 | 30 | 64 | 134 |
| 0.15 | 30 | 41 | 88 |
| 0.20 | 30 | 30 | 59 |

(The r̂=0 column is the closed-form floor `n ≥ log(δ)/log(1-α)` — see
`min_n_zero_errors`; the other two are from `min_n_for_rhat`, found by
search since Hoeffding-Bentkus isn't algebraically invertible in general.)

**Corpus sizing** (`fraud_prevalence=0.15`, `assumed_deny_rate=0.075` —
half of prevalence, since DENY is the confident tail and REVIEW absorbs
the uncertain middle — `test_split_fraction=0.20`, all in
`configs/data.yaml`/`configs/risk.yaml`): at the primary target
**α=0.03, δ=0.10**, sizing to the *conservative* scenario (r̂=0.5·α, since
undershooting the one real test set is exactly the failure mode this
exercise exists to avoid) requires a denial set of **489** claims, hence
a test split of **6,520** claims, hence a full corpus of **32,600**
claims — rounded up to **35,000** for headroom. This is comfortably
within the spec's own 25k–40k target and the `max_practical_full_n=60000`
practical cap in `configs/risk.yaml`: the ladder's first rung (α=0.03)
is achievable and does not fall back to α=0.05. (The optimistic scenario,
r̂=0.3·α, would only have required 14,800 — the gap between the two is
exactly the uncertainty this power analysis exists to plan around.)

`dev` scale is sized with the *optimistic* assumption instead (45 denied
claims → 3,000 total), deliberately — `dev` is never a reported result
(see `docs/EVALUATION_PROTOCOL.md`), so it should stay small and fast
rather than conservative.

Regenerate this analysis at any time with `python
scripts/run_power_analysis.py`.

## DEV-scale numbers (mechanism validation only)

*(Phase 4 fills this in. Explicitly labelled `DEV — not a held-out
result`; never a substitute for the `full`-scale statement above.)*
