"""External data acquisition for PRAMAAN-Bench-v1, with local caching.

Two real, public sources (PRAMAAN_v2_architecture.md Sec.5):

- **ABO** (Amazon Berkeley Objects, CC BY-NC 4.0) via the
  `amaye15/amazon_berkeley_objects` HF mirror, streamed - real product
  photos, used for the legit-real-photo class.
- **Tiny-GenImage** (CC BY-NC-SA 4.0, derived from the GenImage
  benchmark) via direct parquet row-group reads over HF's `hf://`
  filesystem - labeled real/fake images with a `generator` field
  (ADM/BigGAN/GLIDE/Midjourney/SD15/VQDM/Wukong) matching the spec's
  generator-holdout families, used for the fraud-synthetic-image class.
  Note the mirror declares an `SD14` label in its schema but contains no
  SD14 rows; see the substitution note on
  `benchmarks.simulate_ledger.GENERATOR_HOLDOUT`.

Row-group reads (not `datasets streaming=True`, which was unreliably
slow/hanging for this particular repo in testing) let us pull a bounded
sample without downloading a ~470MB parquet shard in full.

**Never commit downloaded bytes.** Everything here writes into
`data/raw/` (gitignored) purely as a cache so repeated `pramaan data`
runs don't re-download. The public repo carries the manifest (paths,
SHA256s, licences) - never the images themselves. See
docs/DATA_CARD.md and scripts/check_image_licenses.py (Phase 6).
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/raw")

ABO_REPO = "amaye15/amazon_berkeley_objects"
ABO_LICENSE = "CC-BY-NC-4.0"

TINY_GENIMAGE_REPO = "TheKernel01/Tiny-GenImage"
TINY_GENIMAGE_LICENSE = "CC-BY-NC-SA-4.0"
TINY_GENIMAGE_GENERATOR_NAMES = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",  # declared by the mirror's schema; no rows actually carry it
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}
# Shards available as data/train-{i:05d}-of-00014.parquet
TINY_GENIMAGE_N_SHARDS = 14


@dataclass(frozen=True)
class SourceImage:
    raw_bytes: bytes
    sha256: str
    source_dataset: str
    source_license: str
    label: str  # "real" or "fake"
    generator: str | None  # None for ABO; a name from TINY_GENIMAGE_GENERATOR_NAMES for GenImage


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_abo_images(n: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> list[SourceImage]:
    """Streams n real product photos, caching each to disk so repeated
    builds don't re-download. Deterministic: always the first n images
    in stream order, so re-running with a larger n only downloads the
    delta.
    """
    cached_dir = cache_dir / "abo"
    cached_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(cached_dir.glob("*.jpg"))
    n_have = len(existing)

    if n_have < n:
        from datasets import load_dataset

        ds = load_dataset(ABO_REPO, split="train", streaming=True)
        it = iter(ds)
        for _ in range(n_have):
            next(it)  # skip already-cached images to resume deterministically
        for i in range(n_have, n):
            example = next(it)
            img = example["pixel_values"].convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            (cached_dir / f"{i:06d}.jpg").write_bytes(buf.getvalue())
        existing = sorted(cached_dir.glob("*.jpg"))

    out: list[SourceImage] = []
    for path in existing[:n]:
        raw = path.read_bytes()
        out.append(SourceImage(raw, _sha256(raw), ABO_REPO, ABO_LICENSE, "real", None))
    return out


def _shard_url(shard: int) -> str:
    return (
        f"hf://datasets/{TINY_GENIMAGE_REPO}/data/"
        f"train-{shard:05d}-of-{TINY_GENIMAGE_N_SHARDS:05d}.parquet"
    )


def _open_shard(shard: int, attempts: int = 3) -> pq.ParquetFile:
    """Opens a shard, retrying transient network failures - HF DNS/connection
    errors were observed intermittently during development, and a whole
    corpus build should not die on one flaky request."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            fs, path = fsspec.core.url_to_fs(_shard_url(shard))
            return pq.ParquetFile(fs.open(path, "rb"))
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable here
            last_error = exc
            logger.warning(
                "genimage: shard %d open failed (attempt %d/%d): %s",
                shard,
                attempt + 1,
                attempts,
                type(exc).__name__,
            )
            time.sleep(2**attempt)
    raise RuntimeError(f"could not open genimage shard {shard}") from last_error


def fetch_genimage_pool(
    min_per_generator: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_shards: int = 4,
    wanted_generators: Iterable[str] | None = None,
    min_real: int = 0,
) -> list[SourceImage]:
    """Pools at least `min_per_generator` images for each wanted generator
    family, caching to disk so later calls with the same or smaller
    requirement need no network at all.

    **Reads are planned, not blind.** Parquet is columnar, so the
    `generator` column for an entire shard costs ~20s to read while a
    single row group of image bytes costs ~60s and ~100MB. So we read the
    generator column first, work out which row groups actually contain a
    family we still need, and fetch image bytes only from those. This
    matters concretely: shard 0 contains no SD-family images at all, and
    the naive loop reads five 100MB row groups discovering that.

    `wanted_generators` defaults to every fake family the mirror declares.
    Pass the families the corpus actually uses to avoid fetching more.
    """
    cached_dir = cache_dir / "genimage"
    cached_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cached_dir / "_pool_manifest.tsv"

    if wanted_generators is None:
        wanted = {n for n in TINY_GENIMAGE_GENERATOR_NAMES.values() if n != "Real"}
    else:
        wanted = set(wanted_generators)

    pool: list[SourceImage] = []
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            sha, label, generator = line.split("\t")
            cached_file = cached_dir / f"{sha}.jpg"
            if not cached_file.exists():
                continue  # manifest/file drift (e.g. interrupted run): skip
            raw = cached_file.read_bytes()
            pool.append(
                SourceImage(
                    raw, sha, TINY_GENIMAGE_REPO, TINY_GENIMAGE_LICENSE, label, generator or None
                )
            )
        logger.info("genimage: %d image(s) already cached", len(pool))

    def _missing() -> set[str]:
        counts = Counter(img.generator for img in pool if img.generator)
        missing = {g for g in wanted if g != "Real" and counts[g] < min_per_generator}
        # "Real" has its own, usually much larger, quota: it backs the
        # non-synthetic claims that decorrelate source dataset from label
        # (see build_bench's real_photo_pool note).
        if "Real" in wanted and counts["Real"] < min_real:
            missing.add("Real")
        return missing

    seen_shas = {img.sha256 for img in pool}

    for shard in range(max_shards):
        missing = _missing()
        if not missing:
            break

        logger.info("genimage: planning shard %d (need %s)", shard, sorted(missing))
        parquet_file = _open_shard(shard)
        generators = parquet_file.read(columns=["generator"]).column("generator").to_pylist()

        # Map each row group to the families it holds, so we only pay for
        # row groups that can actually satisfy something we still need.
        offset = 0
        row_group_families: list[tuple[int, set[str]]] = []
        for rg_index in range(parquet_file.num_row_groups):
            n_rows = parquet_file.metadata.row_group(rg_index).num_rows
            families = {
                TINY_GENIMAGE_GENERATOR_NAMES[g] for g in generators[offset : offset + n_rows]
            }
            row_group_families.append((rg_index, families))
            offset += n_rows

        for rg_index, families in row_group_families:
            missing = _missing()
            if not missing:
                break
            if not (families & missing):
                logger.debug(
                    "genimage: skipping shard %d rg %d (no needed family)", shard, rg_index
                )
                continue

            logger.info(
                "genimage: reading shard %d row group %d for %s",
                shard,
                rg_index,
                sorted(families & missing),
            )
            table = parquet_file.read_row_group(rg_index, columns=["image", "label", "generator"])
            with manifest_path.open("a") as manifest_file:
                for row in table.to_pylist():
                    raw = row["image"]["bytes"]
                    if raw is None:
                        continue
                    generator = TINY_GENIMAGE_GENERATOR_NAMES[row["generator"]]
                    if generator not in wanted:
                        continue
                    sha = _sha256(raw)
                    if sha in seen_shas:
                        continue
                    seen_shas.add(sha)
                    (cached_dir / f"{sha}.jpg").write_bytes(raw)
                    label = "real" if row["label"] == 0 else "fake"
                    pool.append(
                        SourceImage(
                            raw, sha, TINY_GENIMAGE_REPO, TINY_GENIMAGE_LICENSE, label, generator
                        )
                    )
                    manifest_file.write(f"{sha}\t{label}\t{generator}\n")

    still_missing = _missing()
    if still_missing:
        counts = Counter(img.generator for img in pool if img.generator)
        raise RuntimeError(
            f"genimage: could not pool >={min_per_generator} images for "
            f"{sorted(still_missing)} within {max_shards} shard(s). "
            f"Have: { {g: counts[g] for g in sorted(wanted)} }. "
            "Raise max_shards, or check that these families exist in the "
            "upstream mirror at all (SD14 is declared but absent)."
        )
    return pool


def normalize_source_image_to_baseline(raw: bytes, target_size: int = 512) -> bytes:
    """Re-encodes a source pool image to a uniform baseline JPEG (fixed
    max dimension, quality 95) before it enters claim construction.

    This is a deliberate one-time re-encode at the *pool* stage, not a
    forensic-evidence violation of ingest/image.py's "never re-encode"
    rule: that rule is about a submitted CLAIM's bytes, which must stay
    untouched from the moment they're "submitted" in our simulation.
    Establishing a clean, uniform starting point here is what lets the
    later messaging-app degradation pipeline (benchmarks/transforms.py)
    be the sole source of the compression-history variation Pillar 2
    looks for - matching the spec's explicit intent that the legit class
    should not be separable by file cleanliness alone.
    """
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
