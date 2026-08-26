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

**P2 separates AI-generated images, and we cannot yet fully attribute
why.** Error-level analysis (0.60 vs 0.28) and DCT high-frequency ratio
(0.109 vs 0.046) both separate `fraud_synthetic_image` from every other
class by roughly 2x. Two explanations are consistent with that, and they
have very different implications:

1. *Genuine.* AI-generated images really do have different frequency
   content and compression response. This would generalise.
2. *Dataset artifact.* GenImage images arrive with a different encoding
   history than ABO images (generated, then packaged, then re-encoded by
   us), and some of that survives our uniform transport step. This would
   **not** generalise, and would inflate the pixel-adjacent pillars
   exactly where §6's shift matrix expects them to be weakest.

We do not currently distinguish these. The generator-holdout split and
the recompression rows of the shift matrix (Phase 6) are the tests that
will, and this note stays until they have run.

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
