# The guarantee

> **Status.** The machinery, the caveats, and the power analysis are
> real and complete. The one *reportable* certified statement is
> deliberately absent: it comes from the `full`-scale run on the VM
> (Phase 9), resolved through the pre-committed α/δ ladder in
> [`PREREGISTRATION.md`](PREREGISTRATION.md). `DEV`-scale figures below
> are mechanism validation and are labelled as such throughout.
>
> **This document's caveats were written before any certified α was
> computed anywhere in the repository.** That ordering is visible in the
> git history, and it is the point: caveats written after seeing a
> pleasing number are marketing.

## What is being claimed

**In plain English:** among the claims PRAMAAN auto-denies, the fraction
that were actually legitimate is bounded — and the bound comes with a
confidence level rather than a hope.

**Formally:** for a denial threshold `t`, with `FDR_deny(t) = P(y = 0 |
p̂ ≥ t)`,

```
P( FDR_deny(t) ≤ α )  ≥  1 − δ
```

obtained by Learn-then-Test: a Hoeffding–Bentkus p-value for the null
`H₀: FDR_deny > α` at each threshold, evaluated in a fixed sequence from
the most conservative threshold downward, stopping at the first failure.
Fixed-sequence testing is what controls the family-wise error rate
without a multiplicity correction — each test is only performed if every
prior one passed.

The output is a *set* of certified thresholds, not a single number.
Phase 5 selects the cost-minimising member of that set
(`configs/costs.yaml`), which is a policy decision made separately from
the statistical one.

Implementation: [`src/pramaan/risk/certified_set.py`](../src/pramaan/risk/certified_set.py),
on the Hoeffding–Bentkus primitive in
[`hb_pvalue.py`](../src/pramaan/risk/hb_pvalue.py).

---

## The three caveats

### 1. The selective-risk ratio caveat

`FDR_deny` is a ratio of two random quantities. Both the numerator
(legitimate claims denied) and the denominator (claims denied at all)
depend on which claims happened to fall above the threshold. The clean
Learn-then-Test guarantee is stated for a **bounded loss with a fixed
denominator**, and that is not what a conditional rate is.

What we actually do: condition on the realised denial set and treat
`n_denied` as given. That is an approximation, and it is the one place
where the guarantee is weaker than the notation suggests.

So the **unconditional** form is reported alongside it for every
threshold:

```
E[ (1 − y) · 1{deny} ]  ≤  α′
```

This has a fixed denominator — every claim — and is covered by the clean
guarantee with no caveat at all. It is the more defensible statement and
the less useful one: a merchant asks "of the claims we denied, how many
were honest?", not "what fraction of all claims were wrongly denied?".

Both appear in every `ThresholdResult`. The README quotes the conditional
form because it is operationally meaningful, and links here.

### 2. Exchangeability — and it does not hold

Learn-then-Test requires calibration and deployment data to be
exchangeable. **In this domain they are not, and the attacker is the
reason.** A fraudster upgrading their image generator deliberately
changes the distribution the certificate was computed on. This is not an
edge case; it is the expected trajectory of the threat.

Three of the four pillars are built generator-agnostic precisely so the
guarantee has a chance of surviving that shift — the reuse graph in
particular does not care how an image was produced. But "has a chance"
is not "does", so §6's shift matrix **measures** where the certificate
still holds and where it stops, and publishes the cells where it fails.

A guarantee whose failure conditions are unmeasured is not a guarantee.

### 3. Calibration-split single-use

The calibration split is used **once**, for Learn-then-Test, and never
for model selection, hyperparameter choice, or threshold tuning. Two
mechanisms enforce this rather than asking anyone to remember:

- `FusionModel.fit` **raises `SplitDisciplineError`** if handed rows from
  the calibration or test split. Isotonic calibration, which also needs
  held-out data, is instead fitted on K-fold *out-of-fold* predictions
  from the train split — see
  [`fusion/model.py`](../src/pramaan/fusion/model.py).
- The calibration split's content hash is recorded when LTT first
  consumes it, and CI asserts it has not changed since
  (`scripts/check_calibration_seal.py`, `.github/workflows/`).

---

## What Phase 9 adds

The α/δ ladder actually walked at `full` scale, which rung certified,
**and every rung that failed** — reporting only the one that worked would
misrepresent how hard the guarantee was to obtain. If nothing certifies,
that is published as the result, together with the denial-set size that
would have been required.

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
`min_n_zero_errors`; the other two are from `min_n_for_rhat`.)

**A subtlety worth stating, because it nearly produced wrong numbers
here.** `hb_pvalue` is *not* monotone in n. It is the minimum of a
Hoeffding and a Bentkus bound, and the Bentkus term contains
`binom.cdf(ceil(n·r̂), n, α)` — as n rises by one, `ceil(n·r̂)` can jump
and step the p-value **upward**. Across 30 (α, r̂) combinations, 22 showed
at least one such increase, the largest being 0.096.

`min_n_for_rhat` originally binary-searched on the assumption of
monotonicity, which is unsound for a non-monotone function. It now scans
for the **stable** floor: the smallest n such that certification holds at
n *and at every larger n*. Isolated values that certify while n+1 does
not are artifacts of binomial discreteness, and quoting one as "the
sample size you need" would be advice that stops being true when one more
claim arrives.

The table above is unchanged by the correction — 489 was already the
stable floor — and the figures are now pinned by
`test_published_sizing_numbers_are_unchanged` so a future change to the
power code cannot silently invalidate this document.

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

## DEV-scale run: nothing certified, and that is the correct answer

> `DEV — mechanism validation only, not a held-out result.`

Running `pramaan certify --scale dev` walks the full pre-committed ladder
and **certifies nothing at any rung**:

| α | δ | outcome |
|---|---|---|
| 0.03 | 0.10 | failed — no threshold reached the 222-denial floor |
| 0.05 | 0.10 | failed — no threshold reached the 131-denial floor |
| 0.10 | 0.10 | failed — no threshold reached the 64-denial floor |
| 0.10 | 0.20 | failed — no threshold reached the 45-denial floor |

This is not a defect. It is the power analysis from Phase 0.5 coming true
on real data, and the mechanism refusing to issue a certificate it cannot
support. The calibrated scores on the 580-claim dev calibration split
look like this:

| threshold | claims denied | realised FDR |
|---|---|---|
| 0.90 | 18 | **0.000** |
| 0.80 | 24 | 0.167 |
| 0.60 | 28 | 0.250 |
| 0.50 | 34 | 0.382 |
| 0.30 | 82 | 0.598 |

The model *does* have a clean high-confidence region — zero false denials
at t=0.90 — but only 18 claims reach it, well short of every floor.
Loosening the threshold does not help: it buys denials at rapidly worse
FDR. Certification is blocked by **sample size, not by model quality**,
which is precisely the distinction the power floor in
`certified_set.py` exists to make.

**Why this is the outcome the repo wanted from `dev`.** A system that
produced a confident-looking α here would be producing it from 18
observations. `dev` exists to prove the mechanism runs; the reportable
certificate comes from `full` (see
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md)).

### Forward projection to `full` — and one assumption that needs revising

The dev run gives the first real measurement of a quantity Phase 0.5 had
to assume. Sizing used `assumed_deny_rate = 0.075` (half of the 15% fraud
prevalence). The **observed** deny rate in the high-confidence region is
`18/580 = 3.1%` — roughly **2.4× lower than assumed**.

Projecting the observed behaviour onto the 35,000-claim `full` corpus
(20% calibration ⇒ ~7,000 calibration claims):

- expected denials at t≈0.90: `7,000 × 3.1% ≈ 217`
- α=0.03 at δ=0.10 needs **76** denials if the realised FDR stays near
  zero, or **222** if it sits at `0.3α`

So α=0.03 is reachable at `full` scale if the near-zero FDR holds, and
marginal if it does not. That is a genuine open question rather than a
formality, and it is exactly why the ladder is pre-committed: if 0.03
fails, 0.05 is attempted and **both outcomes are published**.

We are flagging the optimistic deny-rate assumption now, before the
`full` run, rather than discovering it afterwards.
