# FAQ

> **Status: skeleton.** This is the Q&A round the submission isn't
> getting live, so it's written adversarially — as if by a hostile
> reviewer — and finalised in Phase 9 once every section it points to
> actually exists. Placeholders below name where each answer will come
> from; see `PRAMAAN_v2_architecture.md` §8 for the source list.

1. **Your synthetic-image detector will fail on the next generator.** →
   points at the shift matrix (`reports/full/shift_matrix.*`); the pixel
   model's ablation contribution.
2. **Missing C2PA is normal in India — WhatsApp strips it.** → P1's
   `UNKNOWN` encoding, near-zero SHAP weight, tested.
3. **Your behavioural data is simulated.** → `docs/DATA_CARD.md`,
   sensitivity sweep, `docs/REAL_DATA_ONRAMP.md`.
4. **Is this really better than a rules engine?** → baselines table with
   bootstrap CIs.
5. **What's the false-positive cost?** → higher than FN at typical order
   values (`configs/costs.yaml`); the abstention band derivation.
6. **Could this be misused offensively?** → no generator, no
   gradient/raw-score exposure, `SAFETY.md`.
7. **Is the shared index legal?** → hash-only + DP counts; DPDP timeline;
   MuleHunter/DPIP precedent (`docs/THREAT_MODEL.md`).
8. **Can the index itself be attacked?** → four anti-poisoning rules,
   measured (`docs/THREAT_MODEL.md`).
9. **Why conformal rather than a tuned threshold?** → `docs/GUARANTEE.md`
   plus the shift matrix showing when it stops holding.
10. **How would this survive contact with real data?** →
    `docs/REAL_DATA_ONRAMP.md` with the exact schema.
