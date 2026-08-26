# Real data onramp

> **Status: schema is real as of Phase 1.** The one-command switch lands
> with the loader work in Phase 2/6; the schema below is already the
> exact contract `benchmarks/simulate_ledger.py` emits and every
> downstream pillar consumes, so a merchant can start preparing an export
> against it now.

## What a merchant would hand over

Two artifacts: a **claims table** (CSV or Parquet) and a **directory of
claim images**, joined on `claim_id`.

### `claims.csv`

| Column | Type | Null? | Consumed by |
|---|---|---|---|
| `claim_id` | string, unique | no | everything; also the image filename (`images/{claim_id}.jpg`) |
| `claim_timestamp` | ISO-8601 datetime | no | temporal split; P3's strictly time-ordered index; label-maturity window (L4) |
| `order_date` | ISO-8601 datetime | no | P4 (claim-to-order latency) |
| `merchant_id` | string | no | federated index (L6); merchant-independence in anti-poisoning rules |
| `claimant_id` | string | no | P4 aggregates; ring detection |
| `phone` | string | yes | `ingest/phone.py` → canonical identity |
| `email` | string | yes | `ingest/email.py` → canonical identity |
| `address` | string (free text, any script) | yes | `ingest/address.py` → canonical identity |
| `pin` | string (6-digit) | yes | address blocking bucket — **required for address matching to run at all** |
| `device_ua` | string | yes | `ingest/device.py` → P4 device-reuse feature |
| `device_screen` | string (`WxH`) | yes | as above |
| `device_timezone` | IANA tz string | yes | as above |
| `device_fonts` | comma-separated string | yes | as above |
| `category` | string | no | Mondrian calibration group (L2) |
| `price_band` | `low`/`mid`/`high` | no | Mondrian calibration group (L2) |
| `order_value_inr` | integer | no | cost model (L4) — drives the FP/FN asymmetry |
| `label` | 0 = legit, 1 = fraud | yes* | training/eval |

\* `label` may be null for claims whose outcome hasn't matured yet
(chargebacks arrive up to 120 days later). The label-maturity window in
L4 handles this explicitly and reports how many claims are censored —
**do not** forward-fill or drop them silently before handing the data
over.

Columns the simulator emits that a merchant does **not** need to supply:
`fraud_class`, `generator_family`, `image_group_id`, `split`. The first
two are benchmark-construction bookkeeping. The latter two are derived:
`image_group_id` by P3's reuse graph, `split` by whatever evaluation
protocol you choose.

### `images/{claim_id}.jpg`

**Original bytes as submitted.** Do not re-encode, resize, or strip
metadata during export — JPEG quantisation tables, the EXIF thumbnail,
and the APP1 segment are exactly what Pillar 2 reads, and a
well-meaning normalisation pass destroys the evidence before it arrives
(`src/pramaan/ingest/image.py` is built around never doing this).

## Switching over

`configs/data.yaml` gains a `source:` key (`simulated` | `real`) plus a
path to the export. With `source: real`, `benchmarks/build_bench.py`
skips `simulate_ledger.py` and the transform pipeline entirely and reads
the merchant table directly; everything downstream — canonicalisation,
split reconciliation and verification, the pillars, calibration, risk
control — is unchanged and requires no code edits.

Two things that behave differently on real data, stated up front:

- **Split construction.** The simulator generates claims already
  consistent with their split. Real history isn't, so
  `benchmarks/splits.py`'s reconciliation pass does more work and drops
  more claims. It reports the count; if that count is large, the
  splitting strategy (not the reconciler) needs revisiting.
- **Prevalence and cost.** `fraud_prevalence` in `configs/data.yaml` and
  the cost constants in `configs/costs.yaml` are anchored to published
  Indian e-commerce figures. Replace them with your own realised rates —
  the abstention band's width is derived from the cost matrix, so wrong
  costs mean a wrong operating point even with a correct model.
