"""SIMULATED claim ledger and claimant behavioural data.

**Declaration (1 of 3 - see docs/DATA_CARD.md and the README for the
other two):** there is no public dataset of refund claims with claimant
history, so every claimant identity, order record, and claim timestamp
in PRAMAAN-Bench-v1 is synthetic. Parameters are anchored to published
rates where the spec cites one (~15% fraud among returns, ~23% India
RTO) - see PRAMAAN_v2_architecture.md Sec.5. Only the *images* backing
each claim are real, sourced data (benchmarks/sources.py). Pillar 4
(claimant behaviour, Phase 2) is trained and evaluated entirely on this
simulated layer, which is why the headline pillar results are always
reported from the real image pillars first (README "Approach"), with
this ledger's contribution reported separately and with a sensitivity
sweep over its least-certain parameters (docs/DATA_CARD.md, Phase 6).

**Split-construction design decision (documented, not hidden):** rather
than solving the joint satisfiability of all 4 leakage constraints
(generator-holdout, temporal, entity-disjoint, ring-disjoint) over an
arbitrarily-connected claim graph, this simulator assigns each claim's
split *cohort* at generation time and then generates its timestamp,
identity-reuse, and image-reuse choices to be *consistent* with that
cohort - e.g. a repeat claimant's second claim is drawn from the same
cohort's identity pool, and a recycled-image claim always reuses an
image from a claim already in its own cohort. Splits are therefore
leakage-free *by construction*, verified (not just assumed) by running
`eval/entity_leakage_audit.py` and the ring-disjointness check in
`benchmarks/splits.py` against the generated output. A real merchant's
historical ledger would not allow this simplification (Phase 1 checklist
in PROGRESS.md); `docs/REAL_DATA_ONRAMP.md` covers what changes when
this simulator is swapped for one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
from faker import Faker

COHORTS = ("train", "calibration", "test")
COHORT_FRACTIONS = {"train": 0.60, "calibration": 0.20, "test": 0.20}

# Generator-holdout families (PRAMAAN_v2_architecture.md Sec.6 split #1),
# assigned by cohort at generation time so the corpus is holdout-clean by
# construction rather than needing a post-hoc reassignment pass.
#
# SUBSTITUTION, declared: the spec's split names `SD 1.4` for train. The
# public Tiny-GenImage mirror we source from declares an SD14 label in its
# schema but contains **zero** SD14 rows (verified across shards 0-2); it
# carries SD 1.5 instead. We substitute SD15 rather than silently drop the
# family, which preserves the spec's holdout structure exactly - two train
# families, one calibration family, four test families, all seven disjoint.
# See docs/DATA_CARD.md.
GENERATOR_HOLDOUT: dict[str, tuple[str, ...]] = {
    "train": ("SD15", "BigGAN"),
    "calibration": ("GLIDE",),
    "test": ("Midjourney", "ADM", "VQDM", "Wukong"),
}

DEFAULT_COMPOSITION: dict[str, float] = {
    "legit_real_photo": 0.85,
    "fraud_synthetic_image": 0.06,
    "fraud_recycled_prior_claim": 0.04,
    "fraud_catalog_photo": 0.025,
    "fraud_screen_rephotograph": 0.015,
    "fraud_metadata_inconsistent_edit": 0.01,
}
FRAUD_CLASSES = frozenset(k for k in DEFAULT_COMPOSITION if k != "legit_real_photo")
# Classes that reuse another claim's image (a "ring" - benchmarks/splits.py
# ring-disjointness check operates on `image_group_id` for these).
RING_FORMING_CLASSES = frozenset({"fraud_recycled_prior_claim", "fraud_catalog_photo"})

CATEGORIES = ("electronics", "apparel", "home", "beauty", "sports")
PRICE_BANDS = ("low", "mid", "high")
PRICE_BAND_RANGES_INR = {"low": (200, 800), "mid": (800, 3000), "high": (3000, 15000)}

# A deliberately small, curated set of real Indian PIN codes spanning a
# few metros - enough geographic variety for PIN-bucketed address fuzzy
# matching (ingest/address.py) to have multiple candidates per bucket
# without needing a full national PIN database for a dev-scale corpus.
PIN_CODES = (
    "400001",  # Mumbai
    "560001",  # Bengaluru
    "110001",  # Delhi
    "600001",  # Chennai
    "700001",  # Kolkata
    "500001",  # Hyderabad
    "411001",  # Pune
    "380001",  # Ahmedabad
)

DEVICE_UAS = (
    "Mozilla/5.0 (Linux; Android 14; SM-G991B) AppleWebKit/537.36 "
    "Chrome/120.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 "
    "Mobile Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
)
DEVICE_SCREENS = ("1080x2400", "1170x2532", "1920x1080", "828x1792")
DEVICE_TIMEZONES = ("Asia/Kolkata",)
DEVICE_FONT_POOLS = (
    ["Roboto", "Noto Sans"],
    ["San Francisco", "Helvetica Neue"],
    ["Segoe UI", "Arial", "Calibri"],
)

SIMULATION_START = date(2025, 10, 1)
COHORT_WINDOW_DAYS: dict[str, tuple[int, int]] = {
    "train": (0, 125),
    "calibration": (126, 167),
    "test": (168, 209),
}


@dataclass(frozen=True)
class ClaimantIdentity:
    phone: str
    email: str
    address: str
    pin: str
    device_ua: str
    device_screen: str
    device_timezone: str
    device_fonts: list[str] = field(default_factory=list)


def _fake_india_phone(rng: random.Random) -> str:
    first = rng.choice("6789")
    rest = "".join(rng.choice("0123456789") for _ in range(9))
    return f"+91 {first}{rest[:4]} {rest[4:]}"


def generate_claimant_identity(rng: random.Random, faker: Faker) -> ClaimantIdentity:
    name_slug = faker.user_name()
    domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "gmail.com"])  # gmail-weighted
    pin = rng.choice(PIN_CODES)
    return ClaimantIdentity(
        phone=_fake_india_phone(rng),
        email=f"{name_slug}{rng.randint(1, 999)}@{domain}",
        address=f"{faker.building_number()} {faker.street_name()}, {faker.city()}",
        pin=pin,
        device_ua=rng.choice(DEVICE_UAS),
        device_screen=rng.choice(DEVICE_SCREENS),
        device_timezone=rng.choice(DEVICE_TIMEZONES),
        device_fonts=rng.choice(DEVICE_FONT_POOLS),
    )


def _draw_cohort(rng: random.Random) -> str:
    weights = [COHORT_FRACTIONS[c] for c in COHORTS]
    return rng.choices(COHORTS, weights=weights, k=1)[0]


def _timestamp_in_cohort(cohort: str, rng: random.Random) -> datetime:
    lo, hi = COHORT_WINDOW_DAYS[cohort]
    day_offset = rng.randint(lo, hi)
    seconds = rng.randint(0, 86399)
    return datetime.combine(SIMULATION_START, datetime.min.time()) + timedelta(
        days=day_offset, seconds=seconds
    )


def simulate_ledger(
    n_claims: int,
    merchants: list[str],
    seed: int,
    composition: dict[str, float] | None = None,
    repeat_claimant_rate: float = 0.10,
) -> pd.DataFrame:
    """Returns a DataFrame of n_claims simulated claim records (no image
    bytes attached - benchmarks/build_bench.py joins these against
    benchmarks/sources.py pools by `fraud_class` / `generator_family`).
    Deterministic for a given (n_claims, merchants, seed, composition).
    """
    composition = composition or DEFAULT_COMPOSITION
    rng = random.Random(seed)
    faker = Faker("en_IN")
    faker.seed_instance(seed)

    classes = list(composition.keys())
    weights = [composition[c] for c in classes]
    fraud_class_choices = rng.choices(classes, weights=weights, k=n_claims)

    # Identity pools, kept separate per cohort so identity reuse never
    # crosses a split boundary (see module docstring).
    identity_pool: dict[str, list[tuple[str, ClaimantIdentity]]] = {c: [] for c in COHORTS}
    # Rows already generated, indexed by cohort, so recycled/ring claims
    # can pick an earlier same-cohort claim to reuse.
    rows_by_cohort: dict[str, list[dict]] = {c: [] for c in COHORTS}

    rows: list[dict] = []
    for i, fraud_class in enumerate(fraud_class_choices):
        claim_id = f"claim_{i:06d}"
        cohort = _draw_cohort(rng)
        timestamp = _timestamp_in_cohort(cohort, rng)

        generator_family = None
        if fraud_class == "fraud_synthetic_image":
            generator_family = rng.choice(GENERATOR_HOLDOUT[cohort])

        image_group_id = claim_id
        if fraud_class in RING_FORMING_CLASSES:
            # Reuse an earlier claim's image from the SAME cohort (any
            # non-fraud_synthetic_image class, so the reuse narrative -
            # "someone's genuine or catalog photo, resubmitted" - holds).
            candidates = [
                r
                for r in rows_by_cohort[cohort]
                if r["fraud_class"] != "fraud_synthetic_image"
                and r["fraud_class"] not in RING_FORMING_CLASSES
            ]
            if candidates:
                original = rng.choice(candidates)
                image_group_id = original["image_group_id"]
                lo, hi = COHORT_WINDOW_DAYS[cohort]
                offset_days = rng.randint(1, 10)
                candidate_ts = original["claim_timestamp"] + timedelta(days=offset_days)
                window_end = datetime.combine(
                    SIMULATION_START, datetime.min.time()
                ) + timedelta(days=hi, hours=23, minutes=59)
                timestamp = min(candidate_ts, window_end)
            # else: no eligible original yet this cohort - falls back to
            # a fresh image_group_id (claim stands alone); rare at n=3000.

        if rng.random() < repeat_claimant_rate and identity_pool[cohort]:
            claimant_id, identity = rng.choice(identity_pool[cohort])
        else:
            identity = generate_claimant_identity(rng, faker)
            claimant_id = f"claimant_{cohort}_{len(identity_pool[cohort]):05d}"
            identity_pool[cohort].append((claimant_id, identity))

        price_band = rng.choice(PRICE_BANDS)
        lo_val, hi_val = PRICE_BAND_RANGES_INR[price_band]
        order_value = rng.randint(lo_val, hi_val)
        order_date = timestamp - timedelta(days=rng.randint(1, 14))

        row = {
            "claim_id": claim_id,
            "cohort": cohort,  # renamed to `split` at manifest-write time (build_bench.py)
            "merchant_id": rng.choice(merchants),
            "fraud_class": fraud_class,
            "label": 0 if fraud_class == "legit_real_photo" else 1,
            "generator_family": generator_family,
            "image_group_id": image_group_id,
            "claimant_id": claimant_id,
            "phone": identity.phone,
            "email": identity.email,
            "address": identity.address,
            "pin": identity.pin,
            "device_ua": identity.device_ua,
            "device_screen": identity.device_screen,
            "device_timezone": identity.device_timezone,
            "device_fonts": ",".join(identity.device_fonts),
            "category": rng.choice(CATEGORIES),
            "price_band": price_band,
            "order_value_inr": order_value,
            "order_date": order_date,
            "claim_timestamp": timestamp,
        }
        rows.append(row)
        rows_by_cohort[cohort].append(row)

    return pd.DataFrame(rows)
