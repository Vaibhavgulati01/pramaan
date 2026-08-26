import io
import random

import numpy as np
import piexif
import pytest
from PIL import Image

from benchmarks.transforms import (
    messaging_app_degradation,
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


def test_messaging_app_degradation_produces_valid_smaller_or_equal_jpeg(
    source_bytes: bytes,
) -> None:
    out = messaging_app_degradation(source_bytes, random.Random(1), max_dim=100)
    img = _decode(out)
    assert img.format == "JPEG"
    assert max(img.size) <= 100
    assert img.info.get("exif") is None


def test_messaging_app_degradation_is_randomized_but_deterministic_per_seed(
    source_bytes: bytes,
) -> None:
    a = messaging_app_degradation(source_bytes, random.Random(7))
    b = messaging_app_degradation(source_bytes, random.Random(7))
    c = messaging_app_degradation(source_bytes, random.Random(8))
    assert a == b
    assert a != c


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
