import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmarks.build_bench import build_bench
from benchmarks.loaders import CorpusNotBuiltError, load_corpus
from benchmarks.simulate_ledger import GENERATOR_HOLDOUT
from benchmarks.sources import SourceImage


def _pool(n: int, dataset: str, label: str, generator: str | None, offset: int = 0):
    out = []
    for i in range(n):
        rng = np.random.default_rng(offset + i)
        arr = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()
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
def built(tmp_path: Path) -> Path:
    genimage = []
    families = sorted({f for fams in GENERATOR_HOLDOUT.values() for f in fams})
    for idx, family in enumerate(families):
        genimage.extend(_pool(6, "test/genimage", "fake", family, offset=1000 + idx * 100))
    build_bench(
        n_claims=250,
        merchants=["m1", "m2"],
        seed=3,
        tier="dev",
        output_root=tmp_path,
        abo_pool=_pool(25, "test/abo", "real", None),
        genimage_pool=genimage,
    )
    return tmp_path


def test_load_corpus_roundtrips_build_output(built: Path) -> None:
    corpus = load_corpus("dev", data_root=built)
    assert corpus.tier == "dev"
    assert len(corpus) > 0
    assert corpus.manifest["name"] == "PRAMAAN-Bench-v1"
    assert len(corpus.manifest["entries"]) == len(corpus)


def test_claims_are_sorted_by_timestamp(built: Path) -> None:
    corpus = load_corpus("dev", data_root=built)
    ts = corpus.claims["claim_timestamp"]
    assert ts.is_monotonic_increasing


def test_iter_claims_yields_in_timestamp_order(built: Path) -> None:
    corpus = load_corpus("dev", data_root=built)
    stamps = [c.claim_timestamp for c in corpus.iter_claims()]
    assert stamps == sorted(stamps)


def test_claim_image_bytes_match_manifest_sha(built: Path) -> None:
    corpus = load_corpus("dev", data_root=built)
    by_id = {e["claim_id"]: e for e in corpus.manifest["entries"]}
    for claim in list(corpus.iter_claims())[:20]:
        raw = claim.read_image_bytes()
        assert hashlib.sha256(raw).hexdigest() == by_id[claim.claim_id]["output_sha256"]


def test_split_returns_only_that_split(built: Path) -> None:
    corpus = load_corpus("dev", data_root=built)
    train = corpus.split("train")
    assert set(train["split"]) == {"train"}
    assert len(train) < len(corpus)


def test_split_rejects_unknown_name(built: Path) -> None:
    corpus = load_corpus("dev", data_root=built)
    with pytest.raises(KeyError, match="holdout"):
        corpus.split("holdout")


def test_iter_claims_can_filter_by_split(built: Path) -> None:
    corpus = load_corpus("dev", data_root=built)
    claims = list(corpus.iter_claims(split="test"))
    assert claims
    assert all(c.split == "test" for c in claims)


def test_generator_family_is_none_for_non_synthetic(built: Path) -> None:
    corpus = load_corpus("dev", data_root=built)
    for claim in corpus.iter_claims():
        if claim.fraud_class == "fraud_synthetic_image":
            assert claim.generator_family is not None
        else:
            assert claim.generator_family is None


def test_missing_corpus_names_the_build_command(tmp_path: Path) -> None:
    with pytest.raises(CorpusNotBuiltError, match="pramaan data --scale dev"):
        load_corpus("dev", data_root=tmp_path)


def test_missing_full_corpus_names_data_full(tmp_path: Path) -> None:
    with pytest.raises(CorpusNotBuiltError, match="pramaan data-full"):
        load_corpus("full", data_root=tmp_path)
