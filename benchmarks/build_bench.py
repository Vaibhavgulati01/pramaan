"""Builds PRAMAAN-Bench-v1: joins the simulated claim ledger against real
sourced images, applies the per-class transforms, reconciles + verifies
the 4-way split, and writes a manifest.

Pipeline:
    simulate_ledger  ->  attach images  ->  reconcile splits  ->  verify
                                                                   |
                                        manifest.json + claims.csv <+

**What lands on disk** (under `data/{tier}/`, all gitignored - see
docs/DATA_CARD.md for why none of it is committed):
- `images/{claim_id}.jpg`   the claim's submitted bytes
- `claims.csv`              one row per claim, incl. split + label
- `manifest.json`           per-file provenance: source dataset, source
                            licence, source SHA256, output SHA256, the
                            transform applied, plus corpus-level build
                            metadata (config, seed, reconciliation report,
                            split verification result)

The manifest is what makes the corpus reproducible without redistributing
CC BY-NC images: it records exactly which upstream image produced each
claim, so a third party can rebuild byte-identically from the same
sources. That is the "ship the recipe, not the images" requirement in
PRAMAAN_v2_architecture.md Sec.5.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.simulate_ledger import (
    DEFAULT_COMPOSITION,
    GENERATOR_HOLDOUT,
    simulate_ledger,
)
from benchmarks.sources import (
    SourceImage,
    fetch_abo_images,
    fetch_genimage_pool,
    normalize_source_image_to_baseline,
)
from benchmarks.splits import ReconciliationReport, reconcile_splits, verify_splits
from benchmarks.transforms import (
    messaging_app_degradation,
    metadata_inconsistent_edit_transform,
    recycle_transform,
    screen_rephotograph_transform,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("data")

# Which transform each fraud class applies to its source image. Classes
# absent here submit their source bytes unmodified (fraud_catalog_photo:
# the "fraud" is contextual - a catalog photo passed off as damage
# evidence - not a pixel edit).
_TRANSFORMS: dict[str, Callable[[bytes, random.Random], bytes]] = {
    "legit_real_photo": messaging_app_degradation,
    "fraud_recycled_prior_claim": recycle_transform,
    "fraud_screen_rephotograph": screen_rephotograph_transform,
    "fraud_metadata_inconsistent_edit": metadata_inconsistent_edit_transform,
}


class BenchBuildError(RuntimeError):
    """Raised when the corpus cannot be built to spec - notably when split
    verification still fails after reconciliation. Never downgraded to a
    warning: a corpus with leaked splits invalidates every number
    downstream of it (docs/GUARANTEE.md)."""


@dataclass(frozen=True)
class ManifestEntry:
    claim_id: str
    split: str
    fraud_class: str
    label: int
    generator_family: str | None
    image_group_id: str
    merchant_id: str
    source_dataset: str
    source_license: str
    source_sha256: str
    transform: str
    output_sha256: str
    output_path: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assign_source_images(
    claims: pd.DataFrame,
    abo_pool: list[SourceImage],
    genimage_by_family: dict[str, list[SourceImage]],
    rng: random.Random,
) -> dict[str, SourceImage]:
    """Picks the source image backing each claim.

    Ring-forming claims (those whose `image_group_id` points at another
    claim) deliberately resolve to that claim's source image - that shared
    origin is precisely the signal P3's reuse graph must detect.
    """
    source_by_group: dict[str, SourceImage] = {}
    assigned: dict[str, SourceImage] = {}

    # Non-ring claims first, so a ring member always finds its origin's
    # image already assigned regardless of row order.
    originals = claims[claims["image_group_id"] == claims["claim_id"]]
    for _, row in originals.iterrows():
        if row["fraud_class"] == "fraud_synthetic_image":
            family = row["generator_family"]
            pool = genimage_by_family.get(family, [])
            if not pool:
                raise BenchBuildError(
                    f"no pooled images for generator family {family!r}; "
                    "increase min_per_generator or max_shards in fetch_genimage_pool"
                )
            source = pool[rng.randrange(len(pool))]
        else:
            source = abo_pool[rng.randrange(len(abo_pool))]
        source_by_group[str(row["image_group_id"])] = source
        assigned[str(row["claim_id"])] = source

    for _, row in claims.iterrows():
        claim_id = str(row["claim_id"])
        if claim_id in assigned:
            continue
        group_id = str(row["image_group_id"])
        ring_source = source_by_group.get(group_id)
        if ring_source is None:
            # Ring origin was dropped by reconciliation - fall back to a
            # fresh image so the claim is still well-formed.
            ring_source = abo_pool[rng.randrange(len(abo_pool))]
            source_by_group[group_id] = ring_source
        assigned[claim_id] = ring_source

    return assigned


def build_bench(
    n_claims: int,
    merchants: list[str],
    seed: int,
    tier: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    composition: dict[str, float] | None = None,
    abo_pool_size: int | None = None,
    min_per_generator: int | None = None,
    abo_pool: list[SourceImage] | None = None,
    genimage_pool: list[SourceImage] | None = None,
) -> dict[str, Any]:
    """Builds the corpus for one tier and returns the manifest dict.

    `abo_pool` / `genimage_pool` are fetched from the network when not
    supplied; passing them lets tests exercise assembly against synthetic
    images without network access (and lets a caller reuse pools across
    several builds).

    Raises BenchBuildError if split verification fails after reconciliation.
    """
    composition = composition or DEFAULT_COMPOSITION
    rng = random.Random(seed)

    logger.info(
        "simulating ledger: n_claims=%d merchants=%d seed=%d", n_claims, len(merchants), seed
    )
    ledger = simulate_ledger(
        n_claims=n_claims, merchants=merchants, seed=seed, composition=composition
    ).rename(columns={"cohort": "split"})

    logger.info("reconciling splits")
    claims, reconciliation = reconcile_splits(ledger)
    logger.info("%s", reconciliation.describe())

    report = verify_splits(claims)
    if not report.is_valid:
        raise BenchBuildError(report.describe())
    logger.info("split verification: %s", report.describe())

    n_needed_abo = abo_pool_size or max(200, int(len(claims) * 0.5))
    if min_per_generator is None:
        # Size the synthetic pool from what this corpus actually consumes
        # rather than a fixed constant: each GenImage row group is ~100MB
        # over the wire, so over-fetching dominates smoke-tier build time.
        # Headroom of 3x (min 10) keeps sampling varied without extra reads.
        synth = claims[claims["fraud_class"] == "fraud_synthetic_image"]
        per_family = synth["generator_family"].value_counts()
        peak = int(per_family.max()) if len(per_family) else 0
        min_per_generator = max(10, peak * 3)

    if abo_pool is None or genimage_pool is None:
        logger.info(
            "fetching source images (abo=%d, genimage>=%d/family)", n_needed_abo, min_per_generator
        )
    if abo_pool is None:
        abo_pool = fetch_abo_images(n_needed_abo)
    if genimage_pool is None:
        wanted = {f for families in GENERATOR_HOLDOUT.values() for f in families}
        genimage_pool = fetch_genimage_pool(min_per_generator, wanted_generators=wanted)

    if not abo_pool:
        raise BenchBuildError("ABO source pool is empty; cannot build a corpus.")

    genimage_by_family: dict[str, list[SourceImage]] = {}
    for img in genimage_pool:
        if img.generator and img.generator != "Real":
            genimage_by_family.setdefault(img.generator, []).append(img)

    sources = _assign_source_images(claims, abo_pool, genimage_by_family, rng)

    out_dir = output_root / tier
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("writing %d claim images to %s", len(claims), images_dir)
    entries: list[ManifestEntry] = []
    for _, row in claims.iterrows():
        claim_id = str(row["claim_id"])
        source = sources[claim_id]
        baseline = normalize_source_image_to_baseline(source.raw_bytes)

        transform = _TRANSFORMS.get(row["fraud_class"])
        transform_name = transform.__name__ if transform else "none"
        # Per-claim RNG so a claim's bytes depend only on (seed, claim_id),
        # not on iteration order or how many claims preceded it.
        claim_rng = random.Random(f"{seed}:{claim_id}")
        output_bytes = transform(baseline, claim_rng) if transform else baseline

        output_path = images_dir / f"{claim_id}.jpg"
        output_path.write_bytes(output_bytes)

        entries.append(
            ManifestEntry(
                claim_id=claim_id,
                split=str(row["split"]),
                fraud_class=str(row["fraud_class"]),
                label=int(row["label"]),
                generator_family=row["generator_family"]
                if pd.notna(row["generator_family"])
                else None,
                image_group_id=str(row["image_group_id"]),
                merchant_id=str(row["merchant_id"]),
                source_dataset=source.source_dataset,
                source_license=source.source_license,
                source_sha256=source.sha256,
                transform=transform_name,
                output_sha256=_sha256(output_bytes),
                output_path=str(output_path.relative_to(output_root)).replace("\\", "/"),
            )
        )

    claims_csv = out_dir / "claims.csv"
    claims.to_csv(claims_csv, index=False)

    manifest = {
        "name": "PRAMAAN-Bench-v1",
        "tier": tier,
        "config": {
            "n_claims_requested": n_claims,
            "merchants": merchants,
            "seed": seed,
            "composition": composition,
            "abo_pool_size": n_needed_abo,
            "min_per_generator": min_per_generator,
        },
        "reconciliation": asdict(reconciliation),
        "split_verification": "ok",
        "counts": {
            "claims": len(claims),
            "by_split": claims["split"].value_counts().to_dict(),
            "by_fraud_class": claims["fraud_class"].value_counts().to_dict(),
            "fraud_prevalence": float(claims["label"].mean()),
        },
        "sources": {
            "abo": {"dataset": "amaye15/amazon_berkeley_objects", "license": "CC-BY-NC-4.0"},
            "genimage": {"dataset": "TheKernel01/Tiny-GenImage", "license": "CC-BY-NC-SA-4.0"},
        },
        "entries": [asdict(e) for e in entries],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("wrote manifest: %s (%d entries)", manifest_path, len(entries))

    return manifest


def summarize(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    recon: ReconciliationReport | dict[str, Any] = manifest["reconciliation"]
    n_dropped = recon["n_dropped"] if isinstance(recon, dict) else recon.n_dropped
    lines = [
        f"PRAMAAN-Bench-v1 [{manifest['tier']}]",
        f"  claims:           {counts['claims']}",
        f"  fraud prevalence: {counts['fraud_prevalence']:.3f}",
        f"  by split:         {counts['by_split']}",
        f"  reconciliation:   {n_dropped} claim(s) dropped",
        "  split verification: OK (generator-holdout, temporal, entity, ring)",
    ]
    return "\n".join(lines)
