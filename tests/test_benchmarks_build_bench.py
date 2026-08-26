"""Corpus-assembly tests. These run against synthetic in-memory source
pools (no network): the goal is to prove build_bench wires the ledger,
transforms, reconciliation, verification, and manifest together
correctly. Whether the real ABO/GenImage fetches work is a separate
concern, covered by actually running `pramaan data --scale smoke`.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmarks.build_bench import BenchBuildError, build_bench, summarize
from benchmarks.simulate_ledger import GENERATOR_HOLDOUT
from benchmarks.sources import SourceImage


def _synthetic_image(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _pool(n: int, dataset: str, label: str, generator: str | None, offset: int = 0):
    out = []
    for i in range(n):
        raw = _synthetic_image(offset + i)
        out.append(
            SourceImage(
                raw_bytes=raw,
                sha256=hashlib.sha256(raw).hexdigest(),
                source_dataset=dataset,
                source_license="TEST-ONLY",
                label=label,
                generator=generator,
            )
        )
    return out


@pytest.fixture
def abo_pool():
    return _pool(30, "test/abo", "real", None)


@pytest.fixture
def genimage_pool():
    pool = []
    families = sorted({f for fams in GENERATOR_HOLDOUT.values() for f in fams})
    for idx, family in enumerate(families):
        pool.extend(_pool(6, "test/genimage", "fake", family, offset=1000 + idx * 100))
    return pool


@pytest.fixture
def manifest(tmp_path: Path, abo_pool, genimage_pool):
    return build_bench(
        n_claims=300,
        merchants=["m1", "m2", "m3"],
        seed=5,
        tier="smoke",
        output_root=tmp_path,
        abo_pool=abo_pool,
        genimage_pool=genimage_pool,
    )


def test_writes_expected_artifacts(tmp_path: Path, manifest) -> None:
    out = tmp_path / "smoke"
    assert (out / "manifest.json").exists()
    assert (out / "claims.csv").exists()
    assert (out / "images").is_dir()

    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["name"] == "PRAMAAN-Bench-v1"
    assert on_disk["tier"] == "smoke"


def test_one_image_per_claim_and_all_decodable(tmp_path: Path, manifest) -> None:
    images = list((tmp_path / "smoke" / "images").glob("*.jpg"))
    assert len(images) == manifest["counts"]["claims"]
    for path in images[:20]:
        with Image.open(path) as img:
            assert img.format == "JPEG"


def test_manifest_entry_per_claim_with_full_provenance(manifest) -> None:
    entries = manifest["entries"]
    assert len(entries) == manifest["counts"]["claims"]
    for e in entries:
        assert e["source_dataset"]
        assert e["source_license"]
        assert len(e["source_sha256"]) == 64
        assert len(e["output_sha256"]) == 64
        assert e["transform"]
        assert e["output_path"].startswith("smoke/images/")


def test_output_sha256_matches_bytes_on_disk(tmp_path: Path, manifest) -> None:
    for e in manifest["entries"][:20]:
        raw = (tmp_path / e["output_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["output_sha256"]


def test_split_verification_recorded_and_prevalence_plausible(manifest) -> None:
    assert manifest["split_verification"] == "ok"
    # ~15% by construction (configs/data.yaml); allow slack at n=300.
    assert 0.08 < manifest["counts"]["fraud_prevalence"] < 0.25


def test_transform_matches_fraud_class(manifest) -> None:
    expected = {
        "legit_real_photo": "none",
        "fraud_recycled_prior_claim": "recycle_transform",
        "fraud_screen_rephotograph": "screen_rephotograph_transform",
        "fraud_metadata_inconsistent_edit": "metadata_inconsistent_edit_transform",
        "fraud_catalog_photo": "none",
        "fraud_synthetic_image": "none",
    }
    for e in manifest["entries"]:
        assert e["transform"] == expected[e["fraud_class"]]


def test_image_dimensions_carry_no_label_signal(tmp_path: Path, manifest) -> None:
    """Regression guard for a real leak: an early build left synthetic-fraud
    claims at 512x512 while every other class sat near 256x256, so image
    size alone was a near-perfect fraud detector. Transport now draws the
    output resolution independently of class - this asserts it stayed that
    way, since the failure is silent and would inflate every pixel-model
    number in the repo.
    """
    sizes_by_label: dict[int, list[int]] = {0: [], 1: []}
    for e in manifest["entries"]:
        with Image.open(tmp_path / e["output_path"]) as img:
            sizes_by_label[e["label"]].append(max(img.size))

    legit, fraud = sizes_by_label[0], sizes_by_label[1]
    assert legit and fraud

    # Both classes must span a comparable range - not one fixed value each.
    assert len(set(legit)) > 1
    assert len(set(fraud)) > 1

    # And no threshold on size may separate them: the ranges must overlap
    # substantially rather than merely differ in mean.
    assert min(fraud) <= max(legit) and min(legit) <= max(fraud)
    overlap_lo, overlap_hi = max(min(legit), min(fraud)), min(max(legit), max(fraud))
    span = max(max(legit), max(fraud)) - min(min(legit), min(fraud))
    assert (overlap_hi - overlap_lo) > 0.5 * span


def test_every_claim_passes_through_transport(tmp_path: Path, manifest) -> None:
    """No claim may reach the corpus at its raw source resolution - that
    is what let source-pool resolution leak the label previously."""
    from benchmarks.transforms import _TRANSPORT_MAX_DIM, _TRANSPORT_MIN_DIM

    for e in manifest["entries"]:
        with Image.open(tmp_path / e["output_path"]) as img:
            assert _TRANSPORT_MIN_DIM <= max(img.size) <= _TRANSPORT_MAX_DIM


def test_synthetic_claims_use_genimage_source_of_the_right_family(manifest) -> None:
    for e in manifest["entries"]:
        if e["fraud_class"] == "fraud_synthetic_image":
            assert e["source_dataset"] == "test/genimage"
            assert e["generator_family"] in GENERATOR_HOLDOUT[e["split"]]
        else:
            assert e["source_dataset"] == "test/abo"


def test_ring_members_share_their_origin_source_image(manifest) -> None:
    by_group: dict[str, set[str]] = {}
    for e in manifest["entries"]:
        by_group.setdefault(e["image_group_id"], set()).add(e["source_sha256"])
    multi = [shas for shas in by_group.values() if len(shas) > 1]
    assert not multi, "claims in one image_group must all derive from one source image"

    # And the corpus must actually contain rings, or the reuse-graph
    # pillar would have nothing to detect at this scale.
    assert any(
        len([e for e in manifest["entries"] if e["image_group_id"] == gid]) > 1 for gid in by_group
    )


def test_deterministic_for_fixed_seed(tmp_path: Path, abo_pool, genimage_pool) -> None:
    kwargs = dict(
        n_claims=200,
        merchants=["m1", "m2"],
        seed=9,
        tier="smoke",
        abo_pool=abo_pool,
        genimage_pool=genimage_pool,
    )
    a = build_bench(output_root=tmp_path / "a", **kwargs)
    b = build_bench(output_root=tmp_path / "b", **kwargs)

    sha_a = [(e["claim_id"], e["output_sha256"]) for e in a["entries"]]
    sha_b = [(e["claim_id"], e["output_sha256"]) for e in b["entries"]]
    assert sha_a == sha_b


def test_empty_abo_pool_raises(tmp_path: Path, genimage_pool) -> None:
    with pytest.raises(BenchBuildError, match="ABO source pool is empty"):
        build_bench(
            n_claims=50,
            merchants=["m1"],
            seed=1,
            tier="smoke",
            output_root=tmp_path,
            abo_pool=[],
            genimage_pool=genimage_pool,
        )


def test_missing_generator_family_raises(tmp_path: Path, abo_pool) -> None:
    with pytest.raises(BenchBuildError, match="no pooled images for generator family"):
        build_bench(
            n_claims=400,
            merchants=["m1"],
            seed=1,
            tier="smoke",
            output_root=tmp_path,
            abo_pool=abo_pool,
            genimage_pool=[],  # no synthetic images at all
        )


def test_summarize_mentions_key_facts(manifest) -> None:
    text = summarize(manifest)
    assert "PRAMAAN-Bench-v1" in text
    assert "smoke" in text
    assert "fraud prevalence" in text
