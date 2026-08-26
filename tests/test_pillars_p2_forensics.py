"""P2 container/compression forensics.

The properties worth testing hardest are the graceful-degradation ones:
a PNG with no quantisation tables and a WhatsApp image with no thumbnail
are *normal*, and encoding either as evidence of fraud would be the same
mistake as treating absent C2PA as adverse.
"""

from __future__ import annotations

import io
import random

import numpy as np
import piexif
import pytest
from PIL import Image

from benchmarks.transforms import metadata_inconsistent_edit_transform
from pramaan.pillars.p2_forensics import extract_forensics


def _photo_bytes(size=(128, 96), quality=90, seed=0) -> bytes:
    """A smoothly-varying image rather than pure noise: noise has
    pathological DCT/FFT statistics and would make the frequency
    features meaningless."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float64)
    base = 128 + 60 * np.sin(xx / 11.0) + 40 * np.cos(yy / 7.0)
    arr = np.clip(base[..., None] + rng.normal(0, 6, (size[1], size[0], 3)), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_extracts_quantisation_tables_from_jpeg() -> None:
    f = extract_forensics(_photo_bytes())
    assert f.has_quant_tables
    assert f.n_quant_tables >= 1
    assert f.qt_luma_mean > 0


def test_png_has_no_quant_tables_and_that_is_not_adverse() -> None:
    f = extract_forensics(_png_bytes())
    assert not f.has_quant_tables
    assert f.n_quant_tables == 0
    assert f.qt_luma_mean == 0.0
    assert f.estimated_quality == 0.0


def test_estimated_quality_is_monotone_in_real_quality() -> None:
    low = extract_forensics(_photo_bytes(quality=40)).estimated_quality
    mid = extract_forensics(_photo_bytes(quality=75)).estimated_quality
    high = extract_forensics(_photo_bytes(quality=95)).estimated_quality
    assert low < mid < high


def test_absent_thumbnail_scores_zero_mismatch_not_high_mismatch() -> None:
    """"No thumbnail" must read as unknown, not as evidence. Most real
    Indian claim images arrive via WhatsApp with metadata stripped."""
    f = extract_forensics(_photo_bytes())
    assert not f.has_thumbnail
    assert not f.thumbnail_comparable
    assert f.thumbnail_mismatch == 0.0


def test_detects_desynced_thumbnail() -> None:
    edited = metadata_inconsistent_edit_transform(_photo_bytes(), random.Random(1))
    f = extract_forensics(edited)
    assert f.has_exif
    assert f.has_thumbnail
    assert f.thumbnail_comparable
    assert f.thumbnail_mismatch > 0.0


def test_consistent_thumbnail_mismatches_less_than_a_desynced_one() -> None:
    """The feature must actually discriminate, not merely be non-zero
    whenever a thumbnail exists."""
    photo = _photo_bytes()
    with Image.open(io.BytesIO(photo)) as img:
        matching = img.copy()
        matching.thumbnail((80, 80))
        thumb_buf = io.BytesIO()
        matching.save(thumb_buf, format="JPEG", quality=90)

    exif = piexif.dump(
        {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": thumb_buf.getvalue()}
    )
    out = io.BytesIO()
    piexif.insert(exif, photo, out)

    honest = extract_forensics(out.getvalue())
    desynced = extract_forensics(metadata_inconsistent_edit_transform(photo, random.Random(2)))

    assert honest.thumbnail_comparable and desynced.thumbnail_comparable
    assert honest.thumbnail_mismatch < desynced.thumbnail_mismatch


def test_malformed_exif_does_not_crash() -> None:
    """EXIF from the wild is frequently broken; a malformed blob is data,
    not a reason to fail the claim."""
    photo = _photo_bytes()
    with Image.open(io.BytesIO(photo)) as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=b"not-valid-exif-data")
    f = extract_forensics(buf.getvalue())
    assert not f.has_thumbnail  # degraded, not raised


def test_ela_falls_as_the_original_gets_more_compressed() -> None:
    """ELA here re-encodes at a fixed Q90 and measures the residual, so it
    reads as "how much detail does this image carry beyond what Q90
    preserves".

    An already-heavily-compressed image (Q35) is coarser than Q90, so
    re-encoding reproduces it almost exactly and the residual is small. A
    near-pristine image (Q97) loses real detail at Q90, so its residual is
    large. The direction is the opposite of the naive reading of "error
    level analysis", and getting it backwards would invert the feature's
    meaning for the fusion model.
    """
    heavily_compressed = extract_forensics(_photo_bytes(quality=35)).ela_mean
    near_pristine = extract_forensics(_photo_bytes(quality=97)).ela_mean
    assert heavily_compressed < near_pristine


def test_frequency_features_are_finite_and_populated() -> None:
    f = extract_forensics(_photo_bytes())
    for value in (f.dct_high_freq_ratio, f.fft_peak_ratio, f.blockiness):
        assert np.isfinite(value)
    assert 0.0 <= f.dct_high_freq_ratio <= 1.0
    assert f.fft_peak_ratio > 0
    assert f.blockiness > 0


def test_tiny_image_degrades_instead_of_crashing() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, format="JPEG")
    f = extract_forensics(buf.getvalue())
    assert f.blockiness == 0.0  # too small for block analysis
    assert np.isfinite(f.ela_mean)


def test_all_features_serialise_to_finite_floats() -> None:
    as_dict = extract_forensics(_photo_bytes()).as_dict()
    assert len(as_dict) == 14
    for name, value in as_dict.items():
        assert isinstance(value, float), name
        assert np.isfinite(value), name


def test_deterministic_for_identical_input() -> None:
    photo = _photo_bytes()
    assert extract_forensics(photo).as_dict() == extract_forensics(photo).as_dict()


def test_rejects_undecodable_bytes() -> None:
    with pytest.raises(Exception):  # noqa: B017 - PIL's specific type is not the point
        extract_forensics(b"not an image at all")
