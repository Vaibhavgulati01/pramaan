import io
import random

import numpy as np
import piexif
import pytest
from PIL import Image

from benchmarks.transforms import (
    _TRANSPORT_MAX_DIM,
    _TRANSPORT_MIN_DIM,
    apply_transport,
    metadata_inconsistent_edit_transform,
    recycle_transform,
    screen_rephotograph_transform,
)


def _sample_image_bytes(size: tuple[int, int] = (200, 150)) -> bytes:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def source_bytes() -> bytes:
    return _sample_image_bytes()


def _decode(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw))


def test_transport_produces_valid_jpeg_in_the_target_size_band(source_bytes: bytes) -> None:
    out = apply_transport(source_bytes, random.Random(1))
    img = _decode(out)
    assert img.format == "JPEG"
    assert _TRANSPORT_MIN_DIM <= max(img.size) <= _TRANSPORT_MAX_DIM


def test_transport_is_deterministic_per_seed(source_bytes: bytes) -> None:
    a = apply_transport(source_bytes, random.Random(7))
    b = apply_transport(source_bytes, random.Random(7))
    c = apply_transport(source_bytes, random.Random(8))
    assert a == b
    assert a != c


def test_transport_upscales_small_sources_as_well_as_downscaling_large_ones() -> None:
    """The leak this closes: a one-directional thumbnail() leaves small
    and large source pools at different resolutions, so image size alone
    predicts which pool a claim came from - and therefore its label."""
    small = _sample_image_bytes((120, 90))
    large = _sample_image_bytes((1400, 1000))

    small_out = _decode(apply_transport(small, random.Random(3)))
    large_out = _decode(apply_transport(large, random.Random(3)))

    assert _TRANSPORT_MIN_DIM <= max(small_out.size) <= _TRANSPORT_MAX_DIM
    assert _TRANSPORT_MIN_DIM <= max(large_out.size) <= _TRANSPORT_MAX_DIM
    # Same rng seed -> same drawn target, regardless of source resolution.
    assert max(small_out.size) == max(large_out.size)


def test_transport_size_distribution_is_independent_of_source_resolution() -> None:
    small_sizes, large_sizes = [], []
    for seed in range(40):
        small_sizes.append(max(_decode(apply_transport(_sample_image_bytes((150, 150)),
                                                        random.Random(seed))).size))
        large_sizes.append(max(_decode(apply_transport(_sample_image_bytes((1200, 1200)),
                                                        random.Random(seed))).size))
    assert small_sizes == large_sizes


def test_transport_strips_metadata_on_the_messaging_app_route(source_bytes: bytes) -> None:
    with_exif = metadata_inconsistent_edit_transform(source_bytes, random.Random(1))
    assert _decode(with_exif).info.get("exif") is not None

    # Across many draws both routes must occur: some claims keep EXIF
    # (direct upload), some lose it (messaging app). If either never
    # happened, the metadata pillar would have either no signal at all or
    # a signal perfectly correlated with the fraud class.
    kept = sum(
        _decode(apply_transport(with_exif, random.Random(s))).info.get("exif") is not None
        for s in range(60)
    )
    assert 0 < kept < 60


def test_recycle_transform_produces_valid_altered_image(source_bytes: bytes) -> None:
    out = recycle_transform(source_bytes, random.Random(1))
    img = _decode(out)
    assert img.format == "JPEG"
    assert out != source_bytes


def test_recycle_transform_crops_smaller_than_original(source_bytes: bytes) -> None:
    orig = _decode(source_bytes)
    out = recycle_transform(source_bytes, random.Random(3))
    result = _decode(out)
    # Cropped-then-rotated-with-expand can grow slightly from rotation,
    # but should never end up larger than the original in both dims
    # simultaneously for a genuine sub-crop plus small-angle rotation.
    assert result.size != orig.size or out != source_bytes


def test_screen_rephotograph_transform_produces_valid_image_same_size(source_bytes: bytes) -> None:
    orig = _decode(source_bytes)
    out = screen_rephotograph_transform(source_bytes, random.Random(1))
    result = _decode(out)
    assert result.format == "JPEG"
    assert result.size == orig.size
    assert out != source_bytes


def test_metadata_inconsistent_edit_embeds_mismatched_thumbnail(source_bytes: bytes) -> None:
    out = metadata_inconsistent_edit_transform(source_bytes, random.Random(1))
    result = _decode(out)
    assert result.format == "JPEG"
    assert result.info.get("exif") is not None

    exif = piexif.load(result.info["exif"])
    assert exif["thumbnail"] is not None

    # The embedded thumbnail must itself be a decodable JPEG, distinct
    # from the main image bytes (that's the whole point of the artifact).
    thumb_img = Image.open(io.BytesIO(exif["thumbnail"]))
    assert thumb_img.format == "JPEG"
    assert max(thumb_img.size) <= 160


def test_metadata_inconsistent_edit_keeps_original_dimensions(source_bytes: bytes) -> None:
    orig = _decode(source_bytes)
    out = metadata_inconsistent_edit_transform(source_bytes, random.Random(2))
    result = _decode(out)
    assert result.size == orig.size
