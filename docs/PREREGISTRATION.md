# Pre-registration

**Committed before the `full` test set was built or unsealed.** Nothing
below was written with knowledge of a `full`-scale result. The git commit
date is the evidence, and it is the only thing that makes this document
worth anything.

## What this is for

The repository's headline finding — that the reuse graph is load-bearing
while the pixel model is nearly disposable — is currently a *hypothesis
taken from the spec*, not a result. Reporting a hypothesis as a discovery
after the fact is the most common way a repository like this becomes
untrustworthy. So the predictions are written down first, and Phase 9
reports actuals against them **including the misses**.

## Honesty note on blinding

These predictions were registered before any `full`-scale run. They were
**not** blind to `dev`-scale behaviour: Phase 3 and 5 ran ablations on
`dev` while building the evaluation machinery, and some outcomes were
visible.

That is stated rather than glossed. Specifically, we already know from
`dev` that:

- forensics takes ~71% of model gain, the opposite of the spec's
  prediction, but ~2/3 of that is a source-dataset artifact
  (`docs/LIMITATIONS.md`);
- with source held constant, forensics and reuse contribute comparably
  (−0.034 vs −0.027 PR-AUC);
- no α/δ rung certified at `dev` scale.

The predictions below are made *with* that knowledge, which makes them
weaker than a truly blind pre-registration and stronger than a
post-hoc narrative. Where a prediction contradicts the spec, it is
because `dev` evidence already pointed that way — and saying so is the
honest position.

---

## 1. Ablation predictions

Reported on the `full` test split, PR-AUC unless noted. "Direction" is
the prediction that matters; magnitudes are stated to make the
prediction falsifiable rather than because we expect precision.

| Ablation | Predicted direction | Predicted Δ PR-AUC | Confidence |
|---|---|---|---|
| − reuse graph (P3) | worse | −0.03 to −0.08 | high |
| − forensics (P2) | worse | −0.02 to −0.10 | **low** — see below |
| − behaviour (P4) | slightly worse | −0.005 to −0.02 | medium |
| − provenance (P1) | ~no change | 0.000 to −0.005 | high |
| − rings | ~no change | 0.000 to −0.005 | medium |
| cascade off (all stages always) | no change or slightly better | 0.000 to +0.01 | high |
| calibration off | PR-AUC unchanged, ECE much worse | ECE ×3 or more | high |
| monotone constraints off | PR-AUC unchanged or marginally better, worse under shift | ±0.01 | medium |
| risk control off (naive F1 threshold) | better PR-AUC, **worse realised FDR** | FDR above α | high |

**We explicitly predict the spec's headline claim will NOT reproduce
cleanly.** The spec expects removing the pixel model to barely move the
ensemble while removing the reuse graph guts it. On `dev`, with source
held constant, the two are comparable. We predict the `full` result will
show **both pillars mattering, with reuse ahead of forensics only once
the source confound is controlled** — and that the uncontrolled ablation
will overstate forensics.

If the controlled ablation shows forensics clearly ahead of reuse, the
architecture's central argument is wrong and we will say so.

## 2. Negative-control predictions

These validate the evaluation methodology, not a hypothesis:

| Control | Prediction |
|---|---|
| Label shuffle | PR-AUC collapses to base rate (~0.15), ±0.03 |
| Random features | PR-AUC collapses to base rate |
| Temporal constraint disabled in P3 | PR-AUC **inflates** measurably (+0.02 or more) |

The third is the interesting one: it measures the size of the bug we
guarded against, and a *small* inflation would be a mildly embarrassing
result worth reporting (it would mean the temporal guard buys less than
claimed).

## 3. The α/δ decision ladder

Walked in this order, stopping at the first rung that certifies. Every
rung attempted is reported, including failures.

1. **α = 0.03, δ = 0.10** — the primary target
2. α = 0.05, δ = 0.10
3. α = 0.10, δ = 0.10
4. α = 0.10, δ = 0.20

**If nothing certifies, that is the published result**, together with the
denial-set size that would have been required. We do not extend the
ladder, loosen it, or search beyond it.

**Prediction:** α=0.03 certifies at `full` scale *if* the near-zero
realised FDR observed at `dev` (0.000 at t=0.90, on 18 denials) holds at
scale; otherwise α=0.05. Projected denials at `full` are ~217, which
clears α=0.03's zero-error floor of 76 but not its 0.3α floor of 222.
This is genuinely uncertain, which is why the ladder exists.

We also flag in advance: Phase 0.5 sized the corpus assuming a deny rate
of 0.075; the observed rate is **3.1%**, about 2.4× lower. If `full`
certifies nothing, this assumption is the first place to look.

## 4. Shift-matrix predictions

Two experiments per condition (frozen calibration vs re-certified on
shifted data). Predicted certificate validity under **frozen**
calibration:

| Condition | Certificate holds? |
|---|---|
| In-distribution | ✅ |
| Unseen generator families | ✅ (reuse is generator-agnostic) |
| JPEG Q60 recompression | ✅ |
| JPEG Q40 recompression | ❌ |
| All metadata stripped | ✅ |
| 90% centre crop | ❌ (defeats pHash; CLIP partially survives) |
| Screenshot round-trip | ✅ |
| Colour jitter + 2° rotation | ❌ |

We predict **re-certification recovers some but not all** of these — and
that the gap between frozen and re-certified is the most interesting
number in the matrix.

## 5. The federation kill-gate

Bounded to **two implementation attempts**. Success requires both:

- the federated on/off ablation shows a directionally-correct, non-zero Δ
  in ring recall at `dev` scale, **and**
- a first-seen poisoning attack is blocked with the immunity rule and
  succeeds with it disabled.

If either fails after two attempts, federation is cut to a documented
design in `THREAT_MODEL.md` and the code is deleted. **Prediction:**
passes — first-seen immunity is already implemented and tested in
`pillars/rings.py`, so the poisoning half is largely done.

---

## Actuals vs prediction

*(Appended in Phase 9 after the single `full`-scale unsealing. Never
edited into the predictions above.)*
