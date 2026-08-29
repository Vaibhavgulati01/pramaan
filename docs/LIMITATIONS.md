# Limitations

> **Status: current through Phase 9's local work.** This file was started
> in Phase 0 and added to as real limitations were found, rather than
> written at the end. Under repo-only judging it is arguably the
> highest-ROI file here: it should be long, specific, and unflattering
> where warranted — a reviewer who reads a thorough limitations section
> stops looking for the flaws you hid, because you've demonstrated you
> already found them. The `full`-tier sections cannot close until the VM
> run happens ([`VM_HANDOFF.md`](VM_HANDOFF.md)).

## Found while building (real, specific)

### Four bugs from one root cause: the environment stood in for the product

Found late, while recording the demo GIF, and grouped together because
they are the same mistake wearing three hats. In each case the
*development environment* supplied something the *shipped artifact* did
not, and every gate we had was running inside that environment.

**1. `pramaan all` silently skipped three of its five stages.** Long
after Phase 6 delivered working `certify`, `eval` and `report`, the `all`
command still called a Phase-1 stub that printed *"not implemented yet —
it lands in Phase 6"* and returned. CI ran `pramaan all --scale smoke` on
every push for four phases and stayed green the entire time, because the
stub wrote to stderr and exited zero. The flagship command was a partial
no-op and **the failure mode was that nothing failed**. Caught only
because the last frame of the demo recording had the stub text in it.

**2. The wheel omitted `benchmarks` and `eval`.** Installed library code
imports both at module level, but `pyproject.toml` shipped only
`src/pramaan`, so `pramaan data` raised `ModuleNotFoundError` for anyone
who installed the package rather than cloning it. Invisible because every
invocation went through `python -m pramaan.cli` *from the repo root*, and
`python -m` puts the working directory on `sys.path` — the source tree
was quietly substituting for the distribution.

**3. `httpx` was undeclared.** Required by starlette's `TestClient`,
present locally as a transitive dependency, absent from a clean install.
CI failed at collection, which took down the whole suite rather than one
module.

**What we changed, beyond the three fixes.** Gates that run only inside
the development environment cannot see this class of bug, so:

- CI now invokes the **installed console script from outside the source
  tree** (`cd $RUNNER_TEMP`), not `python -m` from the checkout.
- `tests/test_entry_point.py` imports the runtime modules from a
  temporary directory. Verified by reverting the packaging fix and
  confirming exactly those tests fail.
- `tests/test_cli_all_runs_every_stage.py` asserts the composition of
  `all` directly, because an exit code demonstrably does not.

**A fourth, found while fixing the first three.** The test suite itself
had the same disease. `test_commands_are_invocable_and_fail_informatively`
invoked every command with no `--scale`, so they defaulted to `dev` — and
on a developer machine, where a `dev` corpus exists, the "unit test" ran a
real train, certify, eval and **rewrote the tracked README**. On CI, with
no `data/`, it took the fast error path it was actually written to test.
Same test, two entirely different behaviours, decided by the environment.

Worse, that loop invoked `serve`, which calls `uvicorn.run()` and blocks
forever. Suite runtime therefore depended on **whether port 8000 happened
to be free**: occupied gave a quick `SystemExit` and a green run, free
hung the suite indefinitely. It passed for weeks and then stopped, with
no change to the code under test. Both are fixed — the repo root is
redirected to an empty directory so the no-corpus branch is
deterministic, and `uvicorn.run` is stubbed.

**What we are not claiming.** This class is not closed. These three were
found by accident, not by a systematic audit, and the honest lesson is
that a green CI badge measured our environment rather than our artifact
for most of the build. Anything else installed-but-untested could still
be broken the same way.

**One test in this group is deliberately weaker than it looks**, and it
is labelled as such in the file: `pramaan <cmd> --help` passes against
the broken wheel, because Typer renders help without executing the
command body. The import tests are what actually catch it.


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

**At the spec's cost parameters, reviewing almost everything is
economically rational — which narrows what automation can be worth.**
This is a property of the cost model, not a defect in the model, and it
is worth stating plainly because it bounds the whole system's value.

With review at ₹40 and a false positive at roughly ₹3,300, **human
review is ~82× cheaper than one wrong denial**. Working out when
auto-approving beats reviewing:

| order value | approve beats review only if P(fraud) < | i.e. certainty needed |
|---|---|---|
| ₹500 | 0.059 | >94% sure legitimate |
| ₹2,000 | 0.018 | >98% sure legitimate |
| ₹8,000 | 0.005 | >99.5% sure legitimate |

On the dev calibration split the cost-optimal policy lands at **₹39,931
per 1,000 claims against ₹40,000 for reviewing everything** — a 0.2%
improvement, auto-approving 0.2% of claims. The model is not confident
enough, often enough, to beat a cheap human.

Three honest readings, and we do not yet know which dominates:

1. *The model is not good enough yet* — plausible at dev scale, where
   nothing certified either.
2. *₹40 per review is optimistic* — it is the spec's figure; a real
   adjudication including escalation and dispute handling may cost more,
   which would widen the automation gap.
3. *The value is in triage, not replacement* — the system's contribution
   may be concentrating review effort rather than removing it, which is a
   different and less headline-friendly claim than the one the
   architecture opens with.

We report the review rate alongside every ₹-per-1,000 figure precisely so
this cannot be hidden by a favourable cost number.

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

### Roughly two-thirds of the forensics signal is source, not fraud

**Roughly two-thirds of the forensics pillar's apparent contribution is
an artifact of which dataset an image came from.** This is quantified,
not suspected, and it qualifies the repository's central ablation claim.

Three measurements, in the order they were made:

1. *Gain-based importance says forensics dominates.* `forensics 70.9% /
   reuse 19.2% / behaviour 9.8%` — the opposite of §6's prediction.
2. *Effect sizes say forensics has almost no label signal.* Standardised
   mean differences on the dev corpus:

   | feature | d(fraud vs legit) | d(GenImage vs ABO) |
   |---|---|---|
   | `dct_high_freq_ratio` | −0.01 | **1.61** |
   | `ela_mean` | 0.10 | **1.29** |
   | `fft_peak_ratio` | −0.03 | **−1.19** |

   These are dataset detectors, not fraud detectors.
3. *The ablation reconciles the two, and shows how much is confounded.*

   | ablation | full corpus | ABO-only (source held constant) |
   |---|---|---|
   | − forensics | **−0.1024** PR-AUC | **−0.0335** |
   | − reuse | −0.0220 | −0.0273 |

   With source held constant the two pillars contribute *comparably*.
   Forensics' 3× apparent lead on the full corpus is mostly it acting as
   a GenImage detector — and GenImage claims carry **2.58×** the fraud
   rate of ABO ones (24.8% vs 9.6%).

**The residual confound is structurally irreducible, not an oversight.**
Mixing the real-photo pool (see above) cut the association but cannot
remove it. Writing `q` for the GenImage share among non-synthetic claims,
`P(GenImage | fraud) = (144 + 301q)/445` while
`P(GenImage | legit) = q`; these are equal only at `q = 1`, i.e. only if
ABO is abandoned entirely. Any corpus that sources AI-generated fraud
from one dataset and real photographs partly from another inherits this.

**What follows for how this repo's numbers should be read:**

- **Gain-based feature importance is not evidence of predictive
  contribution here** and is reported only as a diagnostic. A feature
  that cleanly partitions subpopulations is useful for tree structure
  without predicting the label. This is precisely why §6 mandates
  ablations rather than importance scores, and we ran into the reason
  first-hand.
- The headline ablation must be read as an **upper bound** on the pixel
  pillars, with the source-controlled variant reported beside it.
- Phase 6 will report both the full and ABO-only ablations, plus the
  generator-holdout result, rather than a single number.

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
