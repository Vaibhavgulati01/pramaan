# Related work

Positioning, so the novelty claim is specific rather than implied.

## Commercial platforms

**Forter, Riskified, Signifyd** dominate e-commerce fraud decisioning.
They are considerably more mature than this repository on data scale,
network effects, and integration breadth — that is not the comparison
being drawn.

What they do **not** publicly claim, and what this repository is built
around:

| | Commercial platforms | PRAMAAN |
|---|---|---|
| Output | a score, or a decision from an internal threshold | a decision with a **distribution-free finite-sample bound** on the false-denial rate |
| Threshold choice | tuned, often per-merchant | the **certified set** from Learn-then-Test; cost chooses within it |
| When the guarantee breaks | not published | **published per shift condition**, including the cells that fail |
| Behaviour with no certificate | n/a | auto-deny **disabled** rather than falling back |

Several offer chargeback guarantees, which is a *commercial* guarantee —
the vendor absorbs the loss — not a *statistical* one about the decision
rate. Both are useful; they are different objects, and conflating them
would be the easy overclaim here.

## Synthetic-image detection

**CNNSpot** (Wang et al., 2020) established that a ResNet-50 trained on
one generator transfers surprisingly well across others, and the
follow-up literature has largely been a story of that transfer eroding as
generators improved. **GenImage** (2023) and **AI-GenBench** exist
precisely to measure the erosion under held-out-generator protocols.

PRAMAAN's position is that this literature is solving the wrong problem
for this application. A detector is in a direct arms race with generator
progress. The reuse graph is not: it asks *"has this image been submitted
before?"*, which no generator improvement answers.

Our baseline is named **"ResNet-50 features + linear head, in the spirit
of CNNSpot"** and never "CNNSpot" — it is not a faithful
reimplementation, and mislabelling it would be the kind of small
overclaim that makes a reader doubt the larger ones.

**Where our own evidence complicates the story:** on the
source-controlled ablation, forensics and reuse contribute comparably.
We do not claim the pixel signal is disposable; we claim it is not the
pillar that will survive generator shift, and the shift matrix is the
test.

## Selective classification and conformal prediction

**Learn-then-Test** (Angelopoulos, Bates et al.) generalises
conformal-style risk control to arbitrary bounded losses with
fixed-sequence testing — the machinery in `risk/certified_set.py`.
**Bates et al. (2021)** on risk-controlling prediction sets supplies the
Hoeffding–Bentkus bound.

Two departures worth naming:

1. **The abstention band is cost-derived, not coverage-targeted.** Most
   selective classifiers fix a coverage target and report the risk.
   Here the band's width falls out of the rupee cost matrix — a claim
   goes to REVIEW exactly when paying a human beats acting on the model
   in expectation.
2. **We report the caveat the notation hides.** `FDR_deny` is a ratio of
   two random quantities, so the clean LTT guarantee does not apply to it
   directly; we condition on the realised denial set and report the
   unconditional variant alongside. See `GUARANTEE.md`.

## Selective labeling and off-policy evaluation

The **selective-labeling problem** — you only observe outcomes for
claims you approved — is well studied in credit (Björkegren et al.) and
in the bandit-feedback literature, but is almost never addressed in
applied fraud work, which routinely trains tomorrow's model on logs
censored by today's policy.

`policy/exploration.py` and `policy/dr_ope.py` implement the standard
remedies: ε-exploration with logged propensities, and doubly-robust
off-policy evaluation (Dudík, Langford & Li). Neither is novel; applying
them in a refund-fraud repository and **reporting the exploration cost in
rupees** is the contribution.

## Federated / privacy-preserving fraud intelligence

Domestic precedent: RBIH's **MuleHunter.AI** (29 banks) and **DPIP**
demonstrate collaborative fraud intelligence at national scale in India,
which makes the merchant-side analogue a familiar shape rather than a
novel legal argument.

Technically, salted-hash membership with DP counts is standard private
set intersection territory. The part that is less standard is the
**anti-poisoning rules** — first-seen immunity and k-independence — and
running the poisoning attack against our own index, in both directions,
rather than asserting the defence works.

## What is genuinely new here

Not the components. The composition, and one discipline:

1. **Certificate → cost, in that order.** The certified set constrains;
   cost optimises within it. When nothing certifies, the system refuses
   to auto-deny rather than falling back.
2. **The failure conditions are published.** A guarantee whose breaking
   points are unmeasured is not a guarantee, and the shift matrix is
   meant to contain real ❌s.
3. **The evaluation is instrumented against itself** — negative
   controls, an injected-bug check on the temporal guard, ablations
   reported both confounded and source-controlled, and a corpus flaw we
   found and documented rather than shipped quietly.
