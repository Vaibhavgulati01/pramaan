# FAQ

Written adversarially — as a hostile reviewer would ask. Under repo-only
judging this replaces the Q&A round, so the uncomfortable questions are
answered first and the answers point at code or measurements rather than
at claims.

---

### 1. Your synthetic-image detector will fail on the next generator.

Correct, and we designed around it rather than denying it. Three of the
four pillars are generator-agnostic by construction — the reuse graph in
particular does not care how an image was made.

But we go further than the spec here, because our own measurements
undercut its framing. On the source-controlled ablation, forensics
contributes **−0.0335** PR-AUC and reuse **−0.0273** — *comparable*, not
"pixel model disposable". On the uncontrolled corpus forensics looks 3×
more important, and roughly two-thirds of that is it detecting *which
dataset an image came from*. See
[`LIMITATIONS.md`](LIMITATIONS.md#roughly-two-thirds-of-the-forensics-signal-is-source-not-fraud).

So: the honest claim is that reuse is the pillar we expect to survive
generator shift, and the shift matrix measures whether it does.

### 2. Missing C2PA is normal in India — WhatsApp strips it.

Yes, which is why absent provenance is `UNKNOWN` and **never adverse**.
Every P1 feature is a *positive* assertion, so a missing manifest yields
an all-zero vector rather than risk. Measured: **400/400** real dev
claims report `UNKNOWN`. Asserted by
`test_unknown_contributes_nothing_in_either_direction`.

The pillar contributes ~0% of model gain, exactly as expected. It is
kept because a signed manifest declaring AI generation is decisive when
it *does* appear, and a pillar that fires rarely but is nearly always
right is worth having — provided nothing mistakes its silence for a
verdict.

### 3. Your behavioural data is simulated.

Declared in three places: the README, [`DATA_CARD.md`](DATA_CARD.md), and
the module docstring of `benchmarks/simulate_ledger.py`. There is no
public dataset of refund claims with claimant history.

Consequences we accept: Pillar 4's measured contribution is evidence
about our simulator, not about real claimants. It is reported separately
and never blended into a headline. [`REAL_DATA_ONRAMP.md`](REAL_DATA_ONRAMP.md)
specifies the exact schema that replaces it.

Only the **images** are real, and they are the pillars we lead with.

### 4. Is this really better than a rules engine?

On dev, PR-AUC **0.512 [0.453, 0.569]** against **0.240** for a
hand-specified rules engine, with bootstrap CIs on both. The rules engine
is deliberately *not* fitted — fitting its thresholds would produce a
small logistic model wearing a rules costume, and beating that would
prove nothing about beating rules.

The `full`-scale comparison is the one that counts, and the CI overlap
test is stated in advance in [`VM_HANDOFF.md`](VM_HANDOFF.md): if the
intervals overlap, the architecture has not demonstrated an improvement,
and that is the finding.

### 5. What's the false-positive cost?

Higher than a false negative at every plausible order value, once churn
is priced. Measured: **2.65×** at ₹500, **1.14×** at ₹8,000. The churn
term alone (₹1,050) is ~6× the entire fixed cost of a false negative.

This is the load-bearing argument for the abstention band, and it is
asserted by `test_false_positives_cost_more_at_every_plausible_order_value`
rather than left as prose.

**The uncomfortable corollary**, which we report rather than bury: at
₹40 per review against ~₹3,300 per false positive, human review is ~82×
cheaper than one wrong denial, so reviewing most claims is economically
rational. Our cost-optimal policy sits at **83% review load**. Whether
that is "automation" is a fair challenge, and
[`LIMITATIONS.md`](LIMITATIONS.md) offers three readings without picking
one.

### 6. Could this be misused offensively?

No generative model exists anywhere in the dependency tree — check
`pyproject.toml`. CLIP is used only for `encode_image`; there is no
decoder and no path from an embedding back to pixels.

The API returns tiered decisions and templated reason codes, never
scores or gradients, so it cannot serve as a black-box evasion oracle.
`test_response_never_leaks_a_score` asserts this against a
forbidden-key set, so a refactor that helpfully adds `p_hat` breaks the
build rather than the promise.

Full analysis, including the dual-use trade-off of publishing which
signals we rely on: [`SAFETY.md`](../SAFETY.md).

### 7. Is the shared index legal?

It stores salted perceptual-hash bands (HMAC under a rotating
consortium key) and DP-noised counts. Never images, names, phone
numbers, addresses, or claimant identifiers — it holds *identity group*
ids, so it applies k-independence without learning who anyone is.
Asserted by `test_index_never_stores_raw_identifiers`.

DPDP Rules were notified 13 Nov 2025; Consent Manager goes live 13 Nov
2026; substantive compliance lands 13 May 2027. A hash-only index with DP
counts is designed against that deadline rather than retrofitted to it.
RBIH's MuleHunter.AI (29 banks) and DPIP are domestic precedent for
collaborative fraud intelligence at national scale.

### 8. Can the index itself be attacked?

Yes, and we ran the attack. Four rules are implemented **and measured**
in [`THREAT_MODEL.md`](THREAT_MODEL.md).

The honest part: **first-seen immunity alone does not stop the poisoning
attack.** If the attacker submits the victim's photo first, the attacker
gains the immunity. It is k-independence that blocks it, and there is a
test showing the attack *succeeding* when that rule is relaxed — which
is what makes the defence demonstrated rather than assumed.

### 9. Why conformal rather than a tuned threshold?

Because a tuned threshold is a number someone chose, and a certificate is
a statement with a confidence level attached. The difference shows up
exactly when it matters: at `dev` scale **nothing certified**, and the
system refused to auto-deny rather than falling back to the best
uncertified threshold. A tuned threshold would have happily produced a
confident-looking operating point from 18 observations.

[`GUARANTEE.md`](GUARANTEE.md) states all three caveats — the
selective-risk ratio, exchangeability (which does *not* hold under
generator shift), and calibration-split single-use — and they were
written before any certified α was computed. The git history shows that
ordering.

### 10. How would this survive contact with real data?

[`REAL_DATA_ONRAMP.md`](REAL_DATA_ONRAMP.md) gives the exact CSV schema,
column by column, with which pillar consumes each field and a
one-command switch. Two things are flagged as behaving differently on
real data: split construction does more work (real history is not
generated to be split-clean), and the cost constants must be replaced
with the merchant's own realised rates.

---

### 11. Why does your own README say nothing certified?

Because that is the mechanism working, and it is the single most
important thing this repository demonstrates.

At `dev` scale the calibration split yields 18 high-confidence denials
against the 45–222 the ladder requires. The power analysis in
[`GUARANTEE.md`](GUARANTEE.md) computed those bars **before** the run and
predicted this exact outcome. Faced with insufficient evidence, the
system published that fact and disabled auto-deny — the alternative being
to quote a confident-looking bound derived from 18 observations, which is
precisely what a hand-tuned threshold would have done without telling
anyone.

A guarantee that has never declined to be issued is not a guarantee; it
is a formatting choice. This one declines, on the record, under
conditions it predicted in advance. `full` is sized to clear the bar, and
the same machinery will report whatever it finds there.

### 12. How much of this is actually verified versus asserted?

557 tests. The ones worth naming:

- The **temporal-leak test** was verified by deliberately re-introducing
  the leak and confirming it fails.
- The **cascade index-hole guard** likewise.
- The **poisoning attack** is run in both directions — blocked with the
  rule, succeeding without it.
- **Negative controls** (label-shuffle, random-features) confirm the
  pipeline finds nothing where nothing exists.
- The **packaging tests** import the runtime modules from a temporary
  directory, and were likewise verified by reverting the fix.
- CI enforces leakage audits, the calibration seal, an image-licence
  allowlist, doc links and anchors, and that the committed README matches
  `metrics.json`.

Where something is asserted rather than measured, it is labelled as such.

**And the honest counterweight:** this count was 514 while `pramaan all`
was silently skipping three of its five stages, while the installed
package could not import its own modules, and while one test in this very
suite was starting a real web server and hanging or not depending on
whether a port was free. Both had been broken for
several phases with CI green throughout, because the gates ran inside the
development environment rather than against the shipped artifact. Test
count is not evidence of correctness; the specific tests above are.
[`LIMITATIONS.md`](LIMITATIONS.md) has the full account.
