"""Image normalisation that never destroys forensic evidence
(PRAMAAN_v2_architecture.md Sec.4 L0).

A shocking number of pipelines decode an image and then re-encode it as
their "canonical" copy - which silently strips or rewrites the JPEG
quantisation tables, EXIF, and thumbnail that Pillar 2 (container
forensics, Phase 2) depends on. This module keeps three artifacts
strictly separate and never derives one by round-tripping through
another:

- `raw_bytes`: the original file bytes, untouched, always.
- `decoded_array`: a decoded pixel array for CV/ML pillars (P3 reuse
  graph, the CLIP probe) - derived FROM raw_bytes, never written back to it.
- `exif_blob`: the raw EXIF APP1 segment bytes (not a re-serialised
  dict), or None if the image has no EXIF segment.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class NormalizedImage:
    raw_bytes: bytes
    decoded_array: np.ndarray
    exif_blob: bytes | None
    format: str | None
    width: int
    height: int


def normalize_image(raw_bytes: bytes) -> NormalizedImage:
    """Decodes `raw_bytes` for pixel access without ever re-encoding or
    otherwise mutating it. Raises PIL.UnidentifiedImageError (or OSError)
    on corrupt/unrecognised input - callers building a corpus from
    untrusted downloads should catch that around this call, rather than
    this function silently swallowing bad data.
    """
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.load()  # force full decode while the file handle is open
        exif_blob = img.info.get("exif")
        decoded_array = np.array(img.convert("RGB"))
        width, height = img.size
        image_format = img.format

    return NormalizedImage(
        raw_bytes=raw_bytes,
        decoded_array=decoded_array,
        exif_blob=exif_blob,
        format=image_format,
        width=width,
        height=height,
    )
