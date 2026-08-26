# Limitations

> **Status: skeleton.** This is deliberately the last document to be
> considered "done," even though it starts here in Phase 0 — every phase
> should add to it as real limitations are discovered, and Phase 9 makes
> a final pass. Per the spec: under repo-only judging this is arguably the
> highest-ROI file in the repository. It should be long and specific, and
> unflattering where warranted — a reviewer who reads a thorough
> limitations section stops looking for the flaws you hid, because you've
> demonstrated you already found them.

## Found while building (real, specific)

**Entity resolution is fuzzy, and fuzzy means wrong sometimes — in both
directions.** Building Phase 1 surfaced this concretely rather than
theoretically:

- *False merges.* At a `token_set_ratio ≥ 85` threshold, two genuinely
  different households merged into one canonical identity because they
  shared a street and locality inside one PIN code (`H.No. 97 Chanda
  Marg, Faridabad` vs `H.No. 51 Garg Marg, Faridabad`). `token_set_ratio`
  is deliberately forgiving about differing tokens, which is exactly
  wrong for addresses, where the house number is often the *entire*
  distinguishing signal. `addresses_match` now additionally requires
  numeric tokens to intersect when both sides have them.
- *Irreducible ambiguity.* That fix does not eliminate the problem. A
  residual case — `H.No. 06 Tata Marg, Sri Ganganagar` vs `H.No. 06
  Sarraf, Sri Ganganagar` (same PIN, same house number, different
  street) — is genuinely ambiguous; a human would hesitate too. **No
  threshold tuning drives this to zero**, which is why the corpus build
  enforces one-split-per-entity-component structurally and reports the
  claims it drops, rather than trusting the matcher.
- *False splits are the more dangerous direction and are harder to see.*
  A claimant who uses a fresh phone, a fresh email, and a slightly
  different address on each claim will not be linked at all. Every
  entity-disjointness guarantee in this repo is a guarantee **about the
  canonicalisation we implemented**, not about ground-truth humans. On
  real merchant data with adversarial claimants deliberately varying
  their details, the true entity-leakage rate is unmeasured and is
  probably higher than zero.

**Benchmark construction can leak the label through file properties that
have nothing to do with fraud, and it did.** An early build of
PRAMAAN-Bench-v1 applied the messaging-app degradation pipeline only to
the legit class, as §5 literally describes. The consequence: ABO-sourced
claims came through near 256×256 while GenImage-sourced synthetic-fraud
claims stayed at 512×512, so **image dimensions alone were a near-perfect
detector of the synthetic-fraud class** — and any pixel model trained on
that corpus would have looked excellent for entirely spurious reasons,
directly undermining the ablation the architecture argument rests on.

Fixed by splitting construction into a *fraud transform* (what the
fraudster did) and a *transport* step (how the image reached the
merchant) applied to **every** claim with parameters drawn independently
of class. Guarded by `test_image_dimensions_carry_no_label_signal`.

The general lesson is worth more than the specific bug: **every property
that differs systematically between source pools is a potential leak** —
resolution, compression history, colour profile, file size, even encoder
version. We have closed the ones we found and tested for. We cannot claim
to have found them all, and any residual leak inflates the pixel model's
apparent contribution specifically.

**Our screen-rephotograph simulation is weaker than the real thing.**
Measured on the dev corpus, `fraud_screen_rephotograph` claims are
essentially indistinguishable from legit ones on the frequency features
meant to catch them (FFT peak ratio 3545 vs 3571 for legit), and the
transform barely moves the perceptual hash either (median Hamming 0).
Real screen re-photography produces pronounced moiré, reduced dynamic
range, glare, and keystone distortion; ours applies only a faint
multiplicative grid. **This class is therefore easier than reality in one
sense (it is nearly a clean copy, so reuse detection finds it) and harder
in another (the forensic artifacts that would catch it in production are
not present).** We have deliberately not tuned the transform to make our
own detector look better — the honest reading is that results on this
class say little either way.

**The forensics pillar dominates the fitted model, and that is a warning
sign rather than a result.** Training on the dev corpus, gain splits
roughly `forensics 64% / reuse 28% / behaviour 8% / provenance ~0% /
ring ~0%` — the *opposite* of what §6 predicts (reuse load-bearing, the
pixel-adjacent signals nearly disposable). Two of the top forensics
features look actively suspect on inspection: `fft_peak_ratio` draws 14%
of gain although synthetic-fraud images score *lower* on it than legit
ones (0.66×), and `blockiness` draws 14.5% at a class separation of only
1.06×. A model leaning that hard on features with so little class
separation is fitting structure we have not explained.

The most likely explanation is the encoding-artifact concern immediately
below: GenImage and ABO images carry different compression histories, and
the forensics pillar may be reading the *source dataset* rather than
anything about fraud. The generator-holdout split and the recompression
rows of the shift matrix (Phase 6) are the tests that decide it. **Until
they run, no claim about pillar importance in this repository should be
taken at face value**, including the spec's own predicted finding.

**A feature that measured the corpus rather than the claim reached the
model.** `reuse_n_candidates_examined` counted how many prior claims the
reuse index held, so it grew monotonically with corpus position — 691 →
1597 → 2041 across train/calibration/test. Because splits here are
temporal, that made it a near-perfect proxy for *which split a claim was
in*, and the model drew 10.2% of its gain from it. Removed in feature
schema 1.1.0. It was already flagged in the schema as "an artifact of
index size, not of the claim", which is the lesson: noting a hazard in a
comment is not the same as excluding it.

**The corpus was, for a time, unable to test its own central claim.**
This is the most consequential thing we found, and it was invisible until
the fitted model was inspected.

The first `dev` corpus drew every non-synthetic claim from ABO and every
synthetic-fraud claim from GenImage. Checked directly, the GenImage-sourced
claim set and the AI-generated claim set were **literally identical** (144
claims, same members). "Is this image AI-generated?" and "did this image
come from GenImage?" were therefore the *same question*, and the two
datasets differ in encoding history for reasons that have nothing to do
with AI generation.

The consequence is not a minor caveat. Under that structure:

- The forensics pillar could separate the classes perfectly by reading
  compression provenance, and no evaluation on the corpus could
  distinguish that from genuine detection.
- Observed in the fitted model: forensics took **72.6%** of total gain,
  with `fft_peak_ratio` the single largest feature at 16.7% — despite
  synthetic images scoring *lower* on it than legit ones (0.66×), and
  `blockiness` at 16.0% on a class separation of only 1.06×. A model
  leaning that hard on features with so little separation is reading
  something other than the label's actual cause.
- §6's headline ablation — "removing the pixel model barely moves the
  ensemble, while removing the reuse graph guts it" — was **untestable**.
  Any answer would have been an artifact.

**Fixed** by mixing the real-photo pool: non-synthetic claims now draw
from ABO *and* GenImage's own real-photo class, so source dataset no
longer predicts label and a forensics feature can only earn gain by
finding genuine AI-vs-camera structure. Guarded by
`test_source_dataset_does_not_predict_the_label`.

**What remains uncertain even after the fix.** Decorrelating the pools
removes the confound but does not prove the residual forensics signal is
real. Error-level analysis (0.60 vs 0.28) and DCT high-frequency ratio
(0.109 vs 0.046) still separate synthetic images ~2×, and that could be
genuine AI-image structure or a subtler artifact we have not isolated.
The generator-holdout split and the recompression rows of the shift
matrix (Phase 6) are the tests that decide it. **Until those run, treat
every pillar-importance number in this repository as provisional** —
including the spec's own prediction.

**One generator family in the spec does not exist in the public data.**
§6's split names SD 1.4 as a train family; the public Tiny-GenImage
mirror declares an `SD14` label but carries no SD14 rows. We substituted
SD 1.5 (`docs/DATA_CARD.md`). The holdout structure is preserved, but any
claim about generalising *from SD 1.4 specifically* is not supported by
this corpus.

**Device fingerprints are not identity.** UA strings change on browser
update, font lists change on app install, timezone changes on travel.
`ingest/device.py` computes the fingerprint but it is deliberately **not**
used as an entity-linking signal for the leakage audit — only as a P4
behavioural feature. Two matching fingerprints mean "same device
configuration observed," not "same person."

## Known limitations to document as each phase lands

- **Scale**: `DEV`-scale numbers anywhere in this repo are
  mechanism-validation only, not statistically meaningful — see the
  three-tier discipline in `docs/EVALUATION_PROTOCOL.md`. If `full` never
  ran, say so here plainly, with what that means for every number in the
  README.
- **Simulated behavioural data**: Pillar 4 and the claim ledger are
  simulated (no public dataset of refund claims with claimant history
  exists). Sensitivity sweep results and their implications go here.
- **Exchangeability**: the certificate's core assumption breaks under
  generator shift by construction (the attacker's whole strategy). The
  shift-matrix results — including the real ❌s — get discussed here in
  prose, not just tabulated.
- **The selective-risk ratio caveat** (§4 L3 honesty note #1) and what it
  does and doesn't let us claim.
- **C2PA / provenance**: near-universal absence in the India context
  (WhatsApp strips it); what that means for how much this pillar can
  contribute.
- **Federation**: if the federation kill-gate triggers (see the
  implementation plan), document exactly why the measured version didn't
  land and what would be needed to make it real.
- **Device canonicalisation instability** (§4 L0).
- Anything else found while building — this list is expected to grow.
