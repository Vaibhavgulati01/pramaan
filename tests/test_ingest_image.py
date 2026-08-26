import io

import numpy as np
import piexif
import pytest
from PIL import Image

from pramaan.ingest.image import normalize_image


def _make_jpeg_bytes(with_exif: bool = False) -> bytes:
    img = Image.new("RGB", (32, 16), color=(120, 60, 200))
    buf = io.BytesIO()
    if with_exif:
        exif_dict = {"0th": {piexif.ImageIFD.Make: b"PRAMAAN-test"}}
        exif_bytes = piexif.dump(exif_dict)
        img.save(buf, format="JPEG", exif=exif_bytes)
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color=(1, 2, 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_raw_bytes_are_preserved_exactly() -> None:
    raw = _make_jpeg_bytes(with_exif=True)
    result = normalize_image(raw)
    assert result.raw_bytes == raw


def test_decoded_array_has_expected_shape_and_dtype() -> None:
    raw = _make_jpeg_bytes()
    result = normalize_image(raw)
    assert result.decoded_array.shape == (16, 32, 3)
    assert result.decoded_array.dtype == np.uint8
    assert result.width == 32
    assert result.height == 16


def test_exif_blob_present_when_embedded() -> None:
    raw = _make_jpeg_bytes(with_exif=True)
    result = normalize_image(raw)
    assert result.exif_blob is not None
    assert isinstance(result.exif_blob, bytes)
    parsed = piexif.load(result.exif_blob)
    assert parsed["0th"][piexif.ImageIFD.Make] == b"PRAMAAN-test"


def test_exif_blob_none_when_absent() -> None:
    raw = _make_jpeg_bytes(with_exif=False)
    result = normalize_image(raw)
    assert result.exif_blob is None


def test_png_has_no_exif_blob() -> None:
    raw = _make_png_bytes()
    result = normalize_image(raw)
    assert result.exif_blob is None
    assert result.format == "PNG"


def test_corrupt_bytes_raise_rather_than_silently_returning_garbage() -> None:
    with pytest.raises(Exception):  # noqa: B017 - PIL raises UnidentifiedImageError/OSError
        normalize_image(b"this is not an image")
