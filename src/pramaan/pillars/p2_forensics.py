"""P2 - container and compression forensics
(PRAMAAN_v2_architecture.md Sec.4 L1, stage 2 of the cascade).

Reads the *container*, not the depicted scene: JPEG quantisation tables,
the embedded EXIF thumbnail, error-level analysis, DCT coefficient
statistics, and FFT periodicity. These answer "has this file been
re-encoded, edited, or photographed off a screen?" - questions about
provenance of the bytes rather than about what the image shows.

**Generator-agnostic by construction**, which is the property that lets
the L3 guarantee survive an attacker upgrading their image model: none
of these features look at whether an image is AI-generated. They look at
whether its compression history is consistent with the story the claim
tells. A perfect generator whose output is saved through an editor still
desyncs its thumbnail.

Every feature degrades gracefully. A PNG has no quantisation tables, most
WhatsApp images have no thumbnail, and neither is adverse on its own -
they are encoded as "absent", never as evidence of fraud, for the same
reason `p1_provenance` encodes missing C2PA as UNKNOWN.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import piexif
from PIL import Image

# JPEG standard Annex K luminance table, the baseline most encoders scale
# from. Distance from it is a rough proxy for "what quality was this saved
# at", which matters because a claim's stated history implies a range.
_STANDARD_LUMA_QT = np.array(
    [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ],
    dtype=np.float64,
)


@dataclass
class ForensicsFeatures:
    """P2's contribution to the fused feature vector."""

    # Quantisation tables
    has_quant_tables: bool = False
    n_quant_tables: int = 0
    qt_luma_mean: float = 0.0
    qt_distance_from_standard: float = 0.0
    estimated_quality: float = 0.0

    # Thumbnail consistency
    has_exif: bool = False
    has_thumbnail: bool = False
    thumbnail_mismatch: float = 0.0  # 0 when absent - "unknown", not "adverse"
    thumbnail_comparable: bool = False

    # Error-level analysis
    ela_mean: float = 0.0
    ela_p99: float = 0.0

    # DCT / frequency
    dct_high_freq_ratio: float = 0.0
    fft_peak_ratio: float = 0.0
    blockiness: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "forensics_has_quant_tables": float(self.has_quant_tables),
            "forensics_n_quant_tables": float(self.n_quant_tables),
            "forensics_qt_luma_mean": self.qt_luma_mean,
            "forensics_qt_distance_from_standard": self.qt_distance_from_standard,
            "forensics_estimated_quality": self.estimated_quality,
            "forensics_has_exif": float(self.has_exif),
            "forensics_has_thumbnail": float(self.has_thumbnail),
            "forensics_thumbnail_mismatch": self.thumbnail_mismatch,
            "forensics_thumbnail_comparable": float(self.thumbnail_comparable),
            "forensics_ela_mean": self.ela_mean,
            "forensics_ela_p99": self.ela_p99,
            "forensics_dct_high_freq_ratio": self.dct_high_freq_ratio,
            "forensics_fft_peak_ratio": self.fft_peak_ratio,
            "forensics_blockiness": self.blockiness,
        }


def _quant_features(img: Image.Image, features: ForensicsFeatures) -> None:
    tables = getattr(img, "quantization", None) or {}
    if not tables:
        return  # PNG or a decoder that did not expose them; not adverse

    features.has_quant_tables = True
    features.n_quant_tables = len(tables)

    luma = np.asarray(tables[min(tables)], dtype=np.float64)
    features.qt_luma_mean = float(luma.mean())

    if luma.size == _STANDARD_LUMA_QT.size:
        features.qt_distance_from_standard = float(
            np.abs(luma - _STANDARD_LUMA_QT).mean()
        )
        # Invert the standard IJG scaling to recover an approximate
        # quality setting. Useful as a feature, not as ground truth:
        # encoders differ, and this only has to be monotone in quality.
        ratio = float((luma / _STANDARD_LUMA_QT).mean())
        if ratio <= 0:
            features.estimated_quality = 100.0
        elif ratio < 1.0:
            features.estimated_quality = float(np.clip(100.0 - ratio * 50.0, 1.0, 100.0))
        else:
            features.estimated_quality = float(np.clip(50.0 / ratio, 1.0, 100.0))


def _thumbnail_features(
    raw: bytes, decoded: np.ndarray, img: Image.Image, features: ForensicsFeatures
) -> None:
    exif_blob = img.info.get("exif")
    features.has_exif = exif_blob is not None
    if not exif_blob:
        return

    try:
        thumb_bytes = piexif.load(exif_blob).get("thumbnail")
    except Exception:  # noqa: BLE001 - malformed EXIF is data, not a crash
        return
    if not thumb_bytes:
        return

    features.has_thumbnail = True
    try:
        with Image.open(io.BytesIO(thumb_bytes)) as thumb:
            thumb_arr = np.asarray(thumb.convert("RGB").resize((32, 32)), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return

    with Image.fromarray(decoded) as main:
        main_arr = np.asarray(main.resize((32, 32)), dtype=np.float64)

    # Mean absolute difference between the embedded thumbnail and a
    # downscale of the main image. An editor that regenerates the main
    # image but leaves a stale thumbnail behind shows up here.
    features.thumbnail_comparable = True
    features.thumbnail_mismatch = float(np.abs(thumb_arr - main_arr).mean() / 255.0)


def _ela_features(raw: bytes, decoded: np.ndarray, features: ForensicsFeatures) -> None:
    """Error-level analysis: re-encode at a fixed quality and measure the
    residual.

    Read the direction carefully - it is the opposite of what "error
    level" suggests. The residual measures how much detail the image
    carries *beyond* what Q90 preserves, so it is LARGE for a
    near-pristine image and SMALL for one already compressed below Q90
    (re-encoding something coarser than the target reproduces it almost
    exactly). Verified by
    `test_ela_falls_as_the_original_gets_more_compressed`.
    """
    with Image.fromarray(decoded) as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        with Image.open(buf) as recompressed:
            recompressed_arr = np.asarray(recompressed.convert("RGB"), dtype=np.float64)

    residual = np.abs(decoded.astype(np.float64) - recompressed_arr)
    features.ela_mean = float(residual.mean())
    features.ela_p99 = float(np.percentile(residual, 99))


def _dct_2d(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II via matrix multiply. Small enough here that avoiding a
    scipy.fft dependency keeps the import graph lean."""
    n = block.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    basis[:, 0] *= 1 / np.sqrt(2)
    basis *= np.sqrt(2 / n)
    return basis.T @ block @ basis


def _frequency_features(decoded: np.ndarray, features: ForensicsFeatures) -> None:
    grey = decoded.astype(np.float64).mean(axis=2)

    # --- DCT: energy above the low-frequency corner ---
    h, w = grey.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 >= 8 and w8 >= 8:
        blocks = (
            grey[:h8, :w8]
            .reshape(h8 // 8, 8, w8 // 8, 8)
            .transpose(0, 2, 1, 3)
            .reshape(-1, 8, 8)
        )
        # Sample rather than transform every block: this runs inside the
        # cascade's ~40ms stage-2 budget.
        sample = blocks[:: max(1, len(blocks) // 64)][:64]
        coeffs = np.stack([np.abs(_dct_2d(b)) for b in sample])
        total = coeffs.sum()
        if total > 0:
            low = coeffs[:, :3, :3].sum()
            features.dct_high_freq_ratio = float((total - low) / total)
        features.blockiness = _blockiness(grey)

    # --- FFT: periodic structure, as left by screen re-photography ---
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(grey - grey.mean())))
    if spectrum.size:
        centre_h, centre_w = spectrum.shape[0] // 2, spectrum.shape[1] // 2
        masked = spectrum.copy()
        # Blank the DC neighbourhood: it always dominates and says nothing
        # about periodic screen artifacts.
        masked[centre_h - 3 : centre_h + 4, centre_w - 3 : centre_w + 4] = 0
        median = float(np.median(masked))
        if median > 0:
            features.fft_peak_ratio = float(masked.max() / median)


def _blockiness(grey: np.ndarray) -> float:
    """Discontinuity across 8-pixel JPEG block boundaries relative to
    within-block gradients. Rises with repeated JPEG compression."""
    h, w = grey.shape
    if w < 18 or h < 18:
        return 0.0
    cols = np.arange(8, w - 1, 8)
    boundary = np.abs(grey[:, cols] - grey[:, cols - 1]).mean()
    inner_cols = cols[cols + 4 < w] + 4
    if inner_cols.size == 0:
        return 0.0
    inner = np.abs(grey[:, inner_cols] - grey[:, inner_cols - 1]).mean()
    return float(boundary / inner) if inner > 0 else 0.0


def extract_forensics(raw_bytes: bytes) -> ForensicsFeatures:
    """All P2 features for one image. Never raises on a decodable image:
    a feature that cannot be computed is left at its 'absent' default
    rather than aborting the claim."""
    features = ForensicsFeatures()

    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.load()
        decoded = np.asarray(img.convert("RGB"))
        _quant_features(img, features)
        _thumbnail_features(raw_bytes, decoded, img, features)

    _ela_features(raw_bytes, decoded, features)
    _frequency_features(decoded, features)
    return features
