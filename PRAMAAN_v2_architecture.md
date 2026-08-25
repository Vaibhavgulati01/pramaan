# PRAMAAN v2 — Risk-Controlled Selective Adjudication of Claim Evidence

**Definitive architecture and repository blueprint.**
Supersedes the v1 hackathon spec. Optimised for: *unlimited build time, agentic coding, repository-only evaluation, no live defence.*

---

## 0. What changed, and why the architecture changes with it

| v1 assumption | v2 reality | Consequence |
|---|---|---|
| 36 hours | Unlimited | The architecture can carry a real research contribution, not a demo |
| Live Q&A to defend choices | None | Every objection must be pre-answered **in a committed file** |
| Judges watch a demo | Judges read a repo | README is the product. Reproducibility replaces charisma. |
| Solo/small team coding by hand | Claude Code | Breadth is cheap; *coherence* is now the scarce resource |

The last row is the trap. With an agentic coder, it is trivially easy to produce a sprawling 40-module repo that does twelve things adequately. That repo loses. A repo that does **one thing with a defensible guarantee**, and proves it, wins. Every addition below has to earn its place against that standard.

---

## 1. The thesis (one paragraph — this is the top of your README)

Fraud systems output a score. A score is not a decision, and a decision made from an uncalibrated score under distribution shift is an unbounded liability. PRAMAAN is a claim-evidence verifier that produces **three-tier decisions with a distribution-free, finite-sample guarantee on the false-denial rate**, obtained by Learn-then-Test over a selective-prediction policy whose abstention band is sized by rupee cost rather than by a fixed coverage target. The guarantee is only meaningful if it survives the shift that actually happens in this domain — the attacker upgrading their image generator — so the evidence pillars are deliberately built to be generator-agnostic, and we publish the guarantee's validity **separately for each shift condition, including the ones where it degrades**.

That paragraph contains the whole contribution: *a certified decision, and an honest account of when the certificate stops being worth anything.*

---

## 2. The novel core, stated precisely

Three pieces that compose into something no one else in the track will have:

**(a) Selective adjudication.** The output space is `{APPROVE, REVIEW, DENY}`, not `{fraud, legit}`. The REVIEW band is an explicit abstention region. This is selective classification, and it has a proper coverage–risk theory attached to it.

**(b) Risk control instead of threshold-picking.** Rather than choosing `t_hi` by eyeballing a PR curve, we use **Learn-then-Test (LTT)** on a held-out calibration split to return the *set* of thresholds for which the false-denial rate is statistically certified below α, then pick the cost-minimising member of that certified set. The output is a sentence of the form: *"With 90% confidence, at most 3% of auto-denied claims are legitimate."* Nothing in a hackathon repo normally looks like this.

**(c) The guarantee's Achilles heel is the point.** Conformal-style guarantees require exchangeability between calibration and deployment data. In this domain the attacker breaks exchangeability *on purpose* by switching generators. So we:
- build three of four pillars to be generator-agnostic (the reuse graph in particular is invariant to how the image was made),
- and then **empirically measure whether the certificate holds** under held-out generators, compression, metadata stripping and crop attacks — publishing a table where some cells fail.

The coherence of (a)+(b)+(c) is the thing. Each piece alone is a nice touch; together they are a research argument.

---

## 3. Full system architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L0  INGEST & CANONICALISATION                                            │
│     image bytes · order record · claimant record · claim text            │
│     identity canonicalisation (phone/email/address) · image normalisation │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│ L1  EVIDENCE PILLARS — cost-ordered cascade with early exit              │
│                                                                          │
│   Stage 1  (~2 ms)   P1 Provenance      P4 Behaviour (cached aggregates) │
│                      └── if fused p < τ_exit_lo or p > τ_exit_hi → EXIT  │
│   Stage 2  (~40 ms)  P2 Container forensics (QT, thumb, ELA, DCT, FFT)   │
│                      └── early-exit check again                          │
│   Stage 3  (~120 ms) P3 Reuse graph (pHash → LSH bands → CLIP → FAISS)   │
│   Stage 4  (opt.)    P2b learned synthetic-image probe (CLIP linear)     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  feature vector (≈ 60 dims, typed schema)
┌───────────────────────────────▼──────────────────────────────────────────┐
│ L2  FUSION + GROUP-CONDITIONAL CALIBRATION                               │
│     LightGBM (monotone constraints on 9 features) → Mondrian isotonic    │
│     calibration, conditioned on {category × price band}                  │
│     outputs: p̂ ∈ [0,1] with per-group reliability                        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│ L3  RISK CONTROL  (Learn-then-Test)                                      │
│     Λ = grid of (t_lo, t_hi) · Hoeffding–Bentkus p-values ·              │
│     fixed-sequence testing → certified set Λ̂(α, δ)                       │
│     GUARANTEE: P( FDR_deny ≤ α ) ≥ 1 − δ                                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│ L4  COST-OPTIMAL SELECTIVE POLICY                                        │
│     argmin over Λ̂ of expected ₹ loss · doubly-robust off-policy          │
│     evaluation vs. incumbent policy · ε-exploration for unbiased labels  │
│     → APPROVE / REVIEW / DENY                                            │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│ L5 AUDIT      │     │ L6 FEDERATED     │     │ L7 MONITORING            │
│ SHAP → reason │     │    HASH INDEX    │     │ PSI drift · guarantee    │
│ codes ·       │     │ salted pHash     │     │ validity tracker ·       │
│ heatmap ·     │     │ bands + cuckoo   │     │ shadow-mode replay ·     │
│ evidence pack │     │ filter · DP      │     │ label-delay handling     │
│ (signed PDF)  │     │ counts · anti-   │     │                          │
│               │     │ poisoning rules  │     │                          │
└───────────────┘     └──────────────────┘     └──────────────────────────┘
```

---

## 4. Layer-by-layer specification

### L0 — Ingest & canonicalisation

The unglamorous layer that decides whether L3's guarantee means anything, because bad canonicalisation silently leaks entities across splits.

**Identity canonicalisation** (India-specific, and this is a genuine differentiator):
- Phone: strip `+91`, leading `0`, spaces, hyphens → 10-digit key.
- Email: lowercase; for Gmail-family domains strip dots and everything after `+`.
- Address: transliteration-tolerant canonical form — Unicode NFKC, Devanagari→Latin transliteration via `indic-transliteration`, abbreviation expansion (`flt`→`flat`, `opp`→`opposite`, `nr`→`near`), stopword removal, then `rapidfuzz.token_set_ratio` blocking within an exact PIN-code bucket.
- Device: UA + screen + timezone + font-list hash, with a documented instability caveat.

**Why this matters for the guarantee:** if the same human appears as two "different" claimants across the train/test boundary, your test set is contaminated and the certificate is a fiction. Ship an `entity_leakage_audit.py` that asserts zero canonical-identity overlap across splits and **fails CI** if violated. That script is worth more to a reader than any model file.

**Image normalisation:** decode without re-encoding (preserve original bytes for forensics — a shocking number of pipelines destroy their own evidence by round-tripping through PIL). Keep `raw_bytes`, `decoded_array`, and `exif_blob` as three separate artifacts.

### L1 — The four pillars

Feature specs from v1 carry over. The v2 changes:

**Cascade with early exit.** Order pillars by cost. After each stage, run a cheap interim fusion; if `p̂` is outside `[τ_exit_lo, τ_exit_hi]`, stop. Report **mean cost per adjudication** and the **fraction of claims resolved at each stage**. Crucially: *the certified guarantee must be computed on the cascade as deployed*, not on the full-feature model. Getting this right (and saying that you got it right) is a mark of real rigour.

**Monotone constraints in LightGBM** on the nine features where the direction is known a priori (e.g. `n_distinct_claimants_sharing` ↑ ⇒ risk ↑; `days_account_age` ↑ ⇒ risk ↓). Buys robustness under shift, makes reason codes coherent, and prevents the model learning a nonsense inversion from a thin data slice.

**P3 upgraded to a proper temporal index.** LSH banding over 64-bit pHash (16 bands × 4 bits) for candidate generation, CLIP ViT-B/32 in FAISS `IndexHNSWFlat` for semantic near-dup, and a **strictly time-ordered query**: a claim may only match evidence submitted *before* it. Any implementation that queries the full index including future claims has a temporal leak that will inflate every number in the repo. Make it a unit test.

**Ring detection:** temporal bipartite graph `claimant ↔ image_cluster`. Score clusters, not claims. Emit both claim-level and ring-level predictions and evaluate both.

### L2 — Fusion and group-conditional calibration

Marginal calibration is not enough when decisions are per-claim and priced per-rupee. Use **Mondrian (group-conditional) isotonic regression**: fit separate calibrators per `{category × price-band}` cell, with a shrinkage fallback to the global calibrator for thin cells.

Report:
- Brier, ECE (10 equal-mass bins), MCE
- **Per-group** reliability diagrams — a system well-calibrated overall and badly calibrated on high-value electronics is a system that will lose money exactly where money is
- Calibration under each shift condition

### L3 — Risk control (the centrepiece)

Control the **false-denial rate**: among auto-denied claims, the fraction that were legitimate.

```python
import numpy as np
from scipy.stats import binom

def hb_pvalue(r_hat, n, alpha):
    """Hoeffding–Bentkus p-value for H0: R > alpha (Bates et al., Angelopoulos et al.)."""
    if r_hat >= alpha:
        return 1.0
    hoeffding = np.exp(-n * kl_bernoulli(r_hat, alpha))
    bentkus   = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)
    return float(min(1.0, hoeffding, bentkus))

def kl_bernoulli(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12); q = np.clip(q, 1e-12, 1 - 1e-12)
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def certify_thresholds(p_hat, y, grid, alpha=0.03, delta=0.10):
    """
    Fixed-sequence testing down a monotone 1-D grid of t_hi (most→least conservative).
    Returns the certified set: thresholds where FDR_deny <= alpha w.p. >= 1-delta.
    """
    certified = []
    for t in sorted(grid, reverse=True):          # start most conservative
        denied = p_hat >= t
        n = int(denied.sum())
        if n < 30:                                # too few to certify honestly
            continue
        r_hat = float((y[denied] == 0).mean())    # legit among denied
        if hb_pvalue(r_hat, n, alpha) <= delta:
            certified.append(t)
        else:
            break                                 # fixed-sequence: stop at first failure
    return certified
```

**Three honesty notes that must appear in `docs/GUARANTEE.md`** — writing these down is what separates a repo that *uses* conformal machinery from one that *understands* it:

1. **The selective-risk ratio caveat.** `FDR_deny` is a ratio of two random quantities. The clean LTT guarantee is for a bounded loss with a fixed denominator. We control the conditional rate by conditioning on the realised denial set and reporting the finite-sample correction; state the assumption explicitly rather than papering over it, and additionally report the unconditional variant `E[(1−y)·1{deny}] ≤ α'`, which is guaranteed cleanly.
2. **Exchangeability.** The guarantee holds if calibration and deployment data are exchangeable. They are not, under generator shift. §6 measures exactly how badly.
3. **Calibration-set reuse.** The calibration split is used once, for LTT, and never for model selection. Enforce with a separate file hash and a CI assertion.

Then the headline sentence in the README becomes:

> On the frozen test set, at the cost-optimal certified threshold, **at most 3.0% of auto-denied claims are legitimate, with 90% confidence** (n_denied = 412, empirical rate 1.7%, HB p = 0.041). Under unseen-generator shift this guarantee **holds**; under Q40 recompression it **fails** — see `reports/guarantee_validity.md`.

That last clause is the most valuable sentence in the entire repository.

### L4 — Cost-optimal selective policy + off-policy evaluation

**The selective-labeling problem** — and this is the single most sophisticated thing you can put in this repo, because virtually no production fraud system handles it and no hackathon repo will mention it:

> You only observe ground truth for claims you **approved**. Denied claims never reveal whether they were fraudulent. Training tomorrow's model on today's logs therefore trains on a censored, policy-biased sample, and every subsequent evaluation is optimistic.

Handle it properly:
- **ε-exploration:** approve a random ε=1% of claims that the policy would deny, and log the propensity. This buys unbiased labels in the denial region at a small, quantified cost. Compute and report that cost in rupees.
- **Doubly-robust off-policy evaluation** of any candidate policy against the logged incumbent, so you can claim improvement without deploying.
- **Delayed labels:** chargebacks arrive up to 120 days later. Implement a label-maturity window and evaluate only on matured claims; report how many claims are censored at evaluation time.

```python
def dr_estimate(reward, action, propensity, q_hat, target_policy_probs):
    """Doubly-robust off-policy value estimate. reward in rupees (negative = loss)."""
    direct = (target_policy_probs * q_hat).sum(axis=1)
    ips_w  = target_policy_probs[np.arange(len(action)), action] / np.clip(propensity, 1e-3, None)
    correction = ips_w * (reward - q_hat[np.arange(len(action)), action])
    return float((direct + correction).mean())
```

**Cost model** (from v1, unchanged in structure, now with the asymmetry made explicit and configurable in `configs/costs.yaml`):

```yaml
false_negative:  order_value + 180        # refund + COGS + reverse freight + dead stock
false_positive:  order_value + 250 + p_churn * ltv    # p_churn 0.35, ltv 3000
review:          40
exploration:     epsilon 0.01             # cost logged and reported separately
```

The load-bearing insight, and one that should be a section heading in the README: **for typical order values, a false positive costs more than a false negative once churn is priced.** Naive fraud systems maximise recall and destroy more value than the fraud they stop. The abstention band exists precisely because of this asymmetry, and its width is derived from the cost matrix rather than chosen.

### L5 — Audit trail

Every decision emits a signed, reproducible record:

```json
{
  "claim_id": "...", "decision": "DENY", "p_hat": 0.94,
  "certified_alpha": 0.03, "certified_delta": 0.10,
  "cascade_stage_exited": 3, "compute_ms": 118,
  "reason_codes": [
    {"code": "REUSE_MULTI_CLAIMANT",
     "text": "Submitted image matches 2 prior claims from 2 distinct claimants (first seen 6 days ago).",
     "shap": 0.41},
    {"code": "CATALOG_MATCH",
     "text": "Image matches merchant catalog photo for SKU B07XYZ (cosine 0.99).",
     "shap": 0.28}
  ],
  "provenance": {"c2pa": "absent (encoded as UNKNOWN, not adverse)"},
  "model_sha": "...", "feature_schema_version": "1.2.0",
  "policy_sha": "...", "timestamp": "..."
}
```

Reason codes come from SHAP but are **rendered through a deterministic template layer**, never free-text generated. An auditable adverse-action decision cannot depend on a stochastic text generator, and saying that in the README shows you understand the compliance context you're selling into.

### L6 — Federated hash index (the DPDP layer)

The reuse graph is far more powerful across merchants than within one. Making that lawful is architecture, not paperwork.

- Merchants publish **salted pHash bands** (HMAC with a consortium-rotated key) and quantised CLIP band hashes into a **cuckoo filter** — never raw images, never PII.
- Membership queries return a boolean plus a **differentially-private count** (Laplace noise, ε budgeted and reported per epoch).
- **Anti-poisoning rules** — this is where you show adversarial maturity, because a shared index is itself an attack surface:
  1. *First-seen immunity:* the earliest claimant on an image cluster is never penalised by that cluster. Otherwise I can burn a rival's genuine claim by submitting their photo first.
  2. *k-independence:* a cluster only contributes risk once it contains ≥2 claimants from ≥2 distinct merchant-independent identity groups.
  3. *Rate limits + append-only signed log* so poisoning attempts are detectable after the fact.
  4. *Decay:* cluster evidence half-lives at 180 days.

Document all four in `docs/THREAT_MODEL.md` under "attacks on PRAMAAN itself". A repo that models attacks against its own defence reads completely differently from one that doesn't.

**Regulatory framing (one paragraph, high value):** DPDP Rules were notified 13 Nov 2025; the Consent Manager framework goes live 13 Nov 2026; substantive compliance lands 13 May 2027. A hash-only index with DP counts is designed against that deadline rather than retrofitted to it. There is domestic precedent for collaborative fraud intelligence at national scale — RBIH's MuleHunter.AI is onboarded across 29 banks and DPIP pilots ecosystem-wide model training — which makes the merchant-side analogue an easy story for an Indian panel to accept.

### L7 — Monitoring

- **Feature drift:** PSI + KS per feature, alarm thresholds in config.
- **Guarantee-validity tracker:** recompute the realised false-denial rate on each matured label batch and chart it against α. When the realised rate crosses α, the certificate has expired and the system should widen the abstention band automatically. Ship this as `monitoring/guarantee_watchdog.py`. It is the natural, correct consequence of L3 and almost nobody thinks of it.
- **Shadow mode:** replay logged claims through a candidate policy without acting, then DR-evaluate.

---

## 5. Data plan

### Real, public

| Purpose | Dataset | Link |
|---|---|---|
| Synthetic images, 8 generator families | GenImage | https://github.com/GenImage-Dataset/GenImage · https://arxiv.org/abs/2306.08571 |
| Generator-holdout protocol (train on past generators, test on future) | AI-GenBench | https://github.com/MI-BioLab/AI-GenBench |
| Real product catalog images (398k, CC BY-NC 4.0) | Amazon Berkeley Objects | https://registry.opendata.aws/amazon-berkeley-objects/ · https://amazon-berkeley-objects.s3.amazonaws.com/index.html |
| Real entity graph + behavioural feature validation | IEEE-CIS Fraud Detection | https://www.kaggle.com/c/ieee-fraud-detection |
| Leakage-aware fraud benchmark loaders | fraud-dataset-benchmark | https://github.com/amazon-science/fraud-dataset-benchmark |
| Device/IP e-commerce fraud | fraud-ecommerce | https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce |
| C2PA read/verify | c2pa-python · c2patool | https://github.com/contentauth/c2pa-python · https://opensource.contentauthenticity.org/docs/c2pa-python/ |

Worth chasing with unlimited time: Amazon Science has published a retail **visual defect detection** dataset (238k+ images). Real damaged-product photos would materially strengthen the legitimate-claim class — spend an afternoon on access.

### The corpus: `PRAMAAN-Bench-v1`

With no time limit, **make the benchmark itself a contribution.** Target 25,000–40,000 claims. Ship the *builder*, a manifest with per-file provenance, licences, and SHA256s, and a `datasets`-compatible loader. Do not ship redistributed images where licences forbid it — ship the recipe.

Composition (fraud prevalence ~15%, anchored to the published rate):

| Class | Share | Construction |
|---|---|---|
| Legit, real photo | 85% | ABO + GenImage `nature`, passed through a **messaging-app degradation pipeline** (resize, metadata strip, re-JPEG Q75–90) so the legit class isn't separable by file cleanliness alone |
| Fraud — synthetic image | 6% | GenImage `ai`; holdout families appear **only** in test |
| Fraud — recycled prior claim | 4% | duplicate under a different claimant with crop/rotate/recolour |
| Fraud — catalog photo submitted | 2.5% | the SKU's own listing image as "the damaged item" |
| Fraud — screen re-photograph | 1.5% | moiré-inducing resample or genuine re-shoot |
| Fraud — metadata-inconsistent edit | 1% | re-save through PIL/GIMP so QT and thumbnail desync |

**You never build a generator.** Every synthetic image is drawn from an existing public academic benchmark. That sentence goes in `SAFETY.md`, and it is not merely compliance — under a track that disqualifies offense-capable work, the *absence* of a generation pipeline is a positive signal a reader can verify by reading your dependency list.

### The simulated component — declare it in three places

There is no public dataset of refund claims with claimant history. So Pillar 4 and the claim ledger are simulated, with parameters anchored to published rates (~15% fraud among returns, 52% INR share, ~23% India RTO, ₹150–300 per RTO). Declare this in the README, in `docs/DATA_CARD.md`, and in a code comment at the top of `simulate_ledger.py`. Then:

- Report the image-pillar metrics on real data as the **headline**.
- Report a **sensitivity sweep** (tornado chart) over the three least-certain simulator parameters.
- Ship `docs/REAL_DATA_ONRAMP.md`: the exact CSV schema a merchant would hand you to replace the simulator, and a one-command switch to use it.

A reader who finds a hidden simulator in your code scores you last. A reader who finds it declared in three places, bounded by a sensitivity analysis, and accompanied by an onramp, concludes you are a professional.

---

## 6. Evaluation protocol

### Splits — four, all enforced by CI

1. **Generator-family holdout.** Train {SD 1.4, BigGAN} · calibrate {GLIDE} · test {Midjourney, ADM, VQDM, Wukong}.
2. **Temporal.** All test claims strictly after all train claims.
3. **Entity-disjoint.** Zero canonical-identity overlap. Asserted by `entity_leakage_audit.py`.
4. **Ring-disjoint.** No test ring member in training.

Freeze and SHA the test set. Put the hash in the README. Add a CI job that recomputes it and fails on mismatch — that converts the claim "we didn't tune on test" from an assertion into a verifiable fact, which is the difference between telling and showing.

### Metrics — four granularities

| Granularity | Metrics |
|---|---|
| Image | PR-AUC, per-generator recall |
| Claim | PR-AUC, P@90R, R@95P, Brier, ECE, per-group calibration |
| Claimant | precision/recall on repeat abusers |
| Ring | cluster precision/recall, purity, time-to-detection (claims elapsed before a ring is flagged) |

Plus, system-level: **₹ loss per 1,000 claims** vs. baselines · **review load %** · **certified α with n and HB p-value** · **mean compute ms and stage-exit distribution** · **bootstrap 95% CIs (2,000 resamples) on everything**.

### Baselines — compare outward, not just to yourself

| Baseline | Purpose |
|---|---|
| Approve-all / deny-all | trivial floors |
| Rules (claim-count + value threshold) | what merchants actually run today |
| CLIP linear probe alone | "just use a foundation model" |
| CNNSpot-style ResNet detector | the standard synthetic-image detector |
| LightGBM on behaviour only | "just use tabular fraud ML" |
| **PRAMAAN (full)** | |

Auto-generate this table into the README from `reports/`. Never hand-type a number into a README; every figure should be traceable to a committed artifact.

### Ablations

Leave-one-pillar-out (4), cascade-off, calibration-off, monotone-constraints-off, risk-control-off (naive F1 threshold), federated-index-off.

The expected finding — and the headline of your ablation section — is that **removing the learned pixel model barely moves the ensemble, while removing the reuse graph guts it.** If that is what the data says, it is the most persuasive paragraph in the repository, because it demonstrates you built an architecture rather than tuned a classifier.

### Shift & robustness matrix (this is your signature table)

| Condition | PR-AUC | Recall | Realised FDR_deny | **Certificate holds?** |
|---|---|---|---|---|
| In-distribution | | | | ✅ |
| Unseen generator families | | | | ? |
| JPEG Q60 recompression | | | | ? |
| JPEG Q40 recompression | | | | ? |
| All metadata stripped | | | | ? |
| 90% centre crop | | | | ? |
| Screenshot round-trip | | | | ? |
| Colour jitter + 2° rotation | | | | ? |

**Fill in real ❌ marks.** A table of eight green ticks is not believable and a good reader will assume you fabricated it. A table with two red crosses and a paragraph explaining the failure mode is the single most credible artifact you can publish. Perturbing your own held-out inputs to measure your own degradation is standard defensive evaluation — say so in one line in `SAFETY.md`.

---

## 7. Repository blueprint

```
pramaan/
├── README.md                    # THE deliverable — see the companion README draft
├── SAFETY.md                    # defense-only scope, no generator, dual-use analysis
├── LICENSE                      # Apache-2.0 (+ dataset licence notes)
├── CITATION.cff
├── Makefile                     # make setup | data | train | eval | report | serve | all
├── pyproject.toml               # pinned, uv/poetry lock committed
├── Dockerfile · docker-compose.yml
├── .github/workflows/
│   ├── ci.yml                   # lint, type, unit tests, smoke eval on fixtures
│   ├── leakage.yml              # entity/temporal/ring leakage audits — MUST FAIL loudly
│   └── testset-integrity.yml    # recompute frozen test SHA
├── configs/                     # hydra: data.yaml, model.yaml, costs.yaml, risk.yaml
├── src/pramaan/
│   ├── ingest/                  # canonicalisation (phone, email, address, device)
│   ├── pillars/                 # p1_provenance p2_forensics p3_reuse p4_behaviour
│   ├── cascade/                 # cost-ordered orchestration + early exit
│   ├── fusion/                  # lightgbm + mondrian isotonic
│   ├── risk/                    # ltt.py hb_pvalue.py certified_set.py
│   ├── policy/                  # cost matrix, selection, dr_ope.py, exploration.py
│   ├── audit/                   # shap → reason codes, evidence pack, signed records
│   ├── federated/               # salted bands, cuckoo filter, DP counts, anti-poisoning
│   ├── monitoring/              # psi, guarantee_watchdog, shadow replay
│   └── api/                     # FastAPI /adjudicate /explain /healthz
├── benchmarks/
│   ├── build_bench.py           # PRAMAAN-Bench-v1 builder + manifest + SHAs
│   ├── loaders.py
│   └── baselines/               # rules, clip_probe, cnnspot, tabular_only
├── eval/
│   ├── run_eval.py              # regenerates EVERY number and figure, one command
│   ├── ablations.py · shift_matrix.py · calibration.py · bootstrap.py
│   └── entity_leakage_audit.py
├── reports/                     # committed, auto-generated: *.png, *.md, metrics.json
├── docs/                        # mkdocs site
│   ├── ARCHITECTURE.md · GUARANTEE.md · THREAT_MODEL.md
│   ├── DATA_CARD.md · MODEL_CARD.md · LIMITATIONS.md
│   ├── FAQ.md                   # the Q&A round you're not having
│   └── REAL_DATA_ONRAMP.md
├── tests/                       # incl. temporal-leak test on the reuse index
└── notebooks/                   # outputs only, never the source of truth
```

**Non-negotiables for a repo-judged submission:**
- `make all` reproduces every committed number from scratch.
- `metrics.json` is the single source of truth; README numbers are injected from it by a script.
- Dependencies pinned; lockfile committed; Docker image builds in CI.
- Seeds fixed and recorded; `reports/run_manifest.json` records git SHA, config hash, seed, dataset SHAs.
- A hosted demo (HF Spaces) linked from the README, plus an inline GIF for readers who won't click.

---

## 8. Documentation strategy — you are replacing the Q&A round

With no live defence, `docs/FAQ.md` *is* your defence, and it should be written adversarially — as if by a hostile reviewer. Pre-answer at minimum:

1. Your synthetic-image detector will fail on the next generator. → §6 shift matrix; the pixel model contributes ~3% of ensemble lift.
2. Missing C2PA is normal in India, WhatsApp strips it. → encoded as UNKNOWN, near-zero SHAP weight, tested.
3. Your behavioural data is simulated. → declared in three places, sensitivity sweep, onramp doc.
4. Is this really better than a rules engine? → baselines table with CIs.
5. What's the false-positive cost? → higher than FN at typical order values; that asymmetry derives the abstention band.
6. Could this be misused offensively? → no generator, no gradient/raw-score exposure, `SAFETY.md`.
7. Is the shared index legal? → hash-only + DP counts; DPDP timeline; MuleHunter/DPIP precedent.
8. Can the index itself be attacked? → four anti-poisoning rules in `THREAT_MODEL.md`.
9. Why conformal rather than a tuned threshold? → `GUARANTEE.md`, plus the shift matrix showing when it stops holding.
10. How would this survive contact with real data? → `REAL_DATA_ONRAMP.md` with the exact schema.

`docs/LIMITATIONS.md` should be **long and specific**. Counterintuitively, this is the highest-ROI file in the repository under repo-only judging: a reviewer who reads a thorough limitations section stops looking for the flaws you hid, because you've demonstrated you already found them.

---

## 9. Build order (dependency-ordered, not time-boxed)

**Phase 1 — Foundations.** Repo scaffold, CI, config system, canonicalisation, leakage audits, benchmark builder + manifest + SHAs, frozen splits. *Do not write a model until leakage audits are green.*

**Phase 2 — Pillars.** P3 first (it's the moat and the least likely to disappoint), then P2, P4, P1. Unit-test the temporal constraint on the reuse index before trusting a single number.

**Phase 3 — Fusion & calibration.** LightGBM with monotone constraints, Mondrian isotonic, per-group reliability diagrams.

**Phase 4 — Risk control.** LTT, certified sets, `GUARANTEE.md` with all three caveats written before you report any certified α.

**Phase 5 — Policy & OPE.** Cost matrix, certified-set argmin, DR-OPE, ε-exploration, label-maturity handling.

**Phase 6 — Evaluation.** Unseal test **once**. Baselines, ablations, shift matrix, bootstrap CIs. Whatever it says is what you publish.

**Phase 7 — Audit, federation, monitoring.** Reason codes, evidence pack, cuckoo index + anti-poisoning, drift + guarantee watchdog.

**Phase 8 — Product surface.** FastAPI, hosted demo, GIF, docs site.

**Phase 9 — The README.** Written last, from `metrics.json`. Budget serious effort here; it carries more weight than any single module.

---

## 10. Scope creep to actively refuse

With unlimited time and an agentic coder, the failure mode is breadth. Say no to:

- A second loss class (RTO scoring, chargeback responder). The brief says **one class of loss**. Doing two dilutes both and signals you didn't read the brief.
- A custom-trained deepfake CNN. It'll underperform CLIP probes and swell the repo.
- Any LLM in the decision path. Latency, cost, uncalibrated scores, non-auditable adverse-action reasoning. An LLM is acceptable *only* for rendering already-computed reason codes into customer-facing prose, clearly separated from the decision.
- A blockchain for the audit log. A signed append-only log is sufficient and won't make a reviewer wince.
- Microservices. One service, one endpoint, clean modules.
- Real-time streaming infrastructure. Batch + a sync API is the honest scope.

---

## 11. What determines the outcome

1. **A verifiable frozen test set** — SHA in the README, integrity check in CI. Converts trust into proof.
2. **A certified guarantee, plus the table showing where it breaks.** The contribution and the honesty are the same artifact.
3. **The reuse graph as the empirical centrepiece**, demonstrated by an ablation showing the pixel model is nearly disposable.
4. **The rupee asymmetry argument** — that FP > FN at typical order values, and that this is why the abstention band exists. Most fraud work never states it.
5. **Selective-labeling handled explicitly.** Almost no one will mention it; doing so signals you've thought past the leaderboard to deployment.
6. **A limitations document long enough to be believed.**
7. **`make all` actually working from a clean clone.** More repos fail here than anywhere else, and a reviewer who can't reproduce your numbers will discount all of them.
