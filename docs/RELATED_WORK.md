# Related work

> **Status: skeleton.** Drafted in Phase 9, but noted here from Phase 0
> since it's cheap to research incrementally: the goal is showing PRAMAAN
> knows the landscape it's claiming novelty in, not a literature dump.

## What will go here

**Commercial fraud/returns-abuse platforms** — Forter, Riskified,
Signifyd: what they claim to do, and specifically what they do *not*
claim (a distribution-free finite-sample guarantee on a decision rate,
published per-shift-condition validity). This is where PRAMAAN's actual
differentiation gets stated precisely instead of asserted.

**Synthetic-image detection literature** — CNNSpot and successors,
GenImage/AI-GenBench's own baselines: why PRAMAAN treats the pixel model
as one disposable-by-design pillar among four rather than the whole
system, and what the ablation in `reports/full/` shows about that choice.

**Selective classification / conformal prediction** — Learn-then-Test
(Angelopoulos et al.), the coverage-risk literature underlying §4 L3;
positioning PRAMAAN's cost-derived abstention band against
fixed-coverage selective classifiers.

**Selective labeling / off-policy evaluation in fraud** — why this is
almost never addressed in fraud-detection work despite being how these
systems actually fail in production (§4 L4).
