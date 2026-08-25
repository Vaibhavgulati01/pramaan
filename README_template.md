# PRAMAAN

**Risk-controlled selective adjudication of refund-claim evidence.**

[![CI](badge)](link) [![Leakage audits](badge)](link) [![Test-set integrity](badge)](link) [![License](badge)](link) [![Demo](badge)](hf-spaces-link)

> Every refund desk runs on one assumption: a photograph is proof.
> That assumption broke in 2026.

PRAMAAN adjudicates the *evidence* attached to a refund, damage, or non-receipt claim and returns one of `APPROVE / REVIEW / DENY` — with a **distribution-free, finite-sample guarantee on the false-denial rate**, and an abstention band whose width is derived from the merchant's cost of a wrong decision rather than chosen by hand.

**Headline result (frozen test set, SHA `abc123…`):**

> At the cost-optimal certified threshold, **at most 3.0% of auto-denied claims are legitimate, with 90% confidence** (n=412, empirical 1.7%, HB p=0.041).
> **₹X saved per 1,000 claims** vs. the best baseline, at **Y% human review load**.
> The guarantee **holds** under unseen-generator shift and metadata stripping; it **fails** under Q40 recompression — see [the shift matrix](#shift--robustness).

---

## Contents
[Problem](#the-problem) · [Approach](#approach) · [Results](#results) · [Guarantee](#the-guarantee) · [Shift & robustness](#shift--robustness) · [Ablations](#ablations) · [Reproduce](#reproduce) · [Architecture](#architecture) · [Data](#data) · [Limitations](#limitations) · [Safety](#safety)

---

## The problem

*(~150 words. One class of loss, named precisely. Use the sourced figures: ~15.1% of retail returns are fraudulent or abusive; >$100B annually; fictitious non-receipt is 52% of refund-abuse reports (MRC 2026); 65% of 6,200 consumers surveyed say AI made false refund claims easier (Ravelin, State of Refunds 2026); AI-generated damage imagery is the fastest-growing abuse form (Forter). India context: ~23% RTO, ₹8,000 cr/yr for D2C. One example image of a fabricated damage photo, uncaptioned.)*

## Approach

**What this is not:** an AI-image detector. Those generalise poorly to unseen generators, and §[Ablations](#ablations) shows the learned pixel model contributes only ~3% of ensemble lift here.

**What this is:** four independent evidence pillars — cryptographic **provenance**, container **forensics**, a temporal **reuse graph**, and claimant **behaviour** — fused, group-conditionally calibrated, and converted into a certified three-tier decision.

Three of the four pillars are **generator-agnostic by construction**, which is what allows the statistical guarantee to survive the shift that actually occurs in this domain: the attacker upgrading their tooling.

*(Architecture diagram here.)*

## Results

*(All tables auto-generated from `reports/metrics.json`. Never hand-typed.)*

| System | PR-AUC | R@95P | ₹/1k claims | Review load | Unseen-gen recall |
|---|---|---|---|---|---|
| Approve-all | | | | | |
| Rules engine | | | | | |
| CLIP linear probe | | | | | |
| CNNSpot-style detector | | | | | |
| Behaviour-only GBM | | | | | |
| **PRAMAAN** | | | | | |

*95% bootstrap CIs (2,000 resamples) on every figure.*

### Why false positives cost more than false negatives

*(Short, load-bearing section. At typical order values, `C_FP = order_value + ₹250 + 0.35×₹3,000` exceeds `C_FN = order_value + ₹180`. Systems that maximise recall destroy more value than the fraud they stop. This asymmetry is what derives the abstention band. Include the ₹-loss-vs-threshold curve with the operating point marked.)*

## The guarantee

*(Plain-English statement, then the formal one, then link to `docs/GUARANTEE.md`. State all three caveats up front: the selective-risk ratio caveat, the exchangeability assumption, and single-use of the calibration split. Reliability diagrams, global and per-group.)*

## Shift & robustness

| Condition | PR-AUC | Recall | Realised FDR_deny | Certificate holds? |
|---|---|---|---|---|
| In-distribution | | | | ✅ |
| Unseen generator families | | | | |
| JPEG Q60 | | | | |
| JPEG Q40 | | | | ❌ |
| Metadata stripped | | | | |
| 90% centre crop | | | | |
| Screenshot round-trip | | | | |

*(Include real failures. A paragraph explaining each ❌ is worth more than a table of green ticks.)*

## Ablations

| Configuration | PR-AUC | ₹/1k | Unseen-gen recall |
|---|---|---|---|
| Full | | | |
| − Provenance | | | |
| − Forensics | | | |
| − **Reuse graph** | | | |
| − Behaviour | | | |
| − Risk control (naive F1 threshold) | | | |
| Pixel model only | | | |

*(One-paragraph reading of the table. If the finding is "removing the pixel model costs ~3%, removing the reuse graph costs ~31%", state it plainly — that sentence is the argument for the architecture.)*

## Reproduce

```bash
git clone … && cd pramaan
make setup        # pinned deps, lockfile committed
make data         # builds PRAMAAN-Bench-v1 from public sources; verifies SHAs
make all          # train → calibrate → certify → evaluate → regenerate every figure
```

Or `docker compose up`. Runtime: ~N minutes on CPU.

Every number in this README is injected from `reports/metrics.json`. `run_manifest.json` records git SHA, config hash, seed, and dataset SHAs. The frozen test set hash is checked by CI on every push — see the integrity badge above.

## Architecture

*(Diagram + one line per layer, then link to `docs/ARCHITECTURE.md`.)*

## Data

*(Sources table with links and licences. Then, prominently: which parts are real public data and which parts are simulated, with the sensitivity sweep and a link to `docs/REAL_DATA_ONRAMP.md`. Do not bury this.)*

## Limitations

*(Link to `docs/LIMITATIONS.md`, and inline the five most important ones. Be specific and unflattering. This is the highest-trust section in the document.)*

## Safety

Defense-only by construction:
- No generative model in the repository. Check `pyproject.toml`.
- All adversarial examples drawn from pre-existing public academic benchmarks (GenImage, AI-GenBench). We deliberately did not build a fake-damage generator, even for evaluation.
- Robustness testing perturbs only our own held-out inputs to measure our own degradation.
- The API returns a tiered decision and reason codes — never raw scores or gradients — so it cannot be used as a black-box evasion oracle.
- Auto-deny requires corroboration beyond the image pillars alone. Absent provenance is encoded as `UNKNOWN`, never as adverse.
- The federated index stores salted perceptual-hash bands and DP-noised counts only. Never images, names, phone numbers, or addresses.

Full analysis, including attacks against PRAMAAN itself: `docs/THREAT_MODEL.md`.

## Citation

*(CITATION.cff)*
