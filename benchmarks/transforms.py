"""Image transforms that construct PRAMAAN-Bench-v1's fraud classes from
a pool of source images (PRAMAAN_v2_architecture.md Sec.5 composition
table). Each transform operates on raw bytes in, raw bytes out - the
transformed bytes become the claim's own `raw_bytes` from that point on
(ingest/image.py's "never re-encode a claim's own bytes" rule applies
*after* this: once a transform below has produced a claim's submitted
image, nothing downstream re-encodes it again).

Two classes need no transform at all and aren't here:
- legit_real_photo: `messaging_app_degradation` below IS its transform.
- fraud_catalog_photo: the SKU's own (unmodified) catalog image is
  submitted as-is - the "fraud" is contextual (same bytes as a known
  catalog photo, submitted as claim evidence), not a pixel transform.
"""

from __future__ import annotations

import io
import random

import numpy as np
import piexif
from PIL import Image, ImageEnhance


def messaging_app_degradation(raw: bytes, rng: random.Random, max_dim: int = 1280) -> bytes:
    """resize, metadata strip, re-JPEG Q75-90 - applied to the LEGIT
    class specifically so it isn't separable from fraud classes by file
    cleanliness alone (spec Sec.5)."""
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(75, 90))  # no exif= -> strips metadata
        return buf.getvalue()


def recycle_transform(raw: bytes, rng: random.Random) -> bytes:
    """Crop + rotate + recolour - the fraud_recycled_prior_claim class:
    a prior claim's (or the catalog's) image, resubmitted by a different
    claimant with just enough alteration to not be a byte-identical
    match, but still a strong near-duplicate for P3's reuse graph."""
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        w, h = img.size
        crop_frac = rng.uniform(0.85, 0.97)
        cw, ch = max(1, int(w * crop_frac)), max(1, int(h * crop_frac))
        left = rng.randint(0, w - cw)
        top = rng.randint(0, h - ch)
        img = img.crop((left, top, left + cw, top + ch))
        img = img.rotate(rng.uniform(-8, 8), expand=True, fillcolor=(255, 255, 255))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(80, 95))
        return buf.getvalue()


def screen_rephotograph_transform(raw: bytes, rng: random.Random) -> bytes:
    """Moire-inducing resample - simulates photographing a phone/monitor
    screen rather than the physical item: aggressive downsample then
    nearest-neighbour upsample (aliasing) plus a faint fine-pitch grid
    multiplied over the image (subpixel/moire interference)."""
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        w, h = img.size
        small = img.resize((max(1, w // 3), max(1, h // 3)), Image.Resampling.BILINEAR)
        upsampled = small.resize((w, h), Image.Resampling.NEAREST)

        arr = np.array(upsampled).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        grid_period = rng.uniform(2.5, 4.0)
        pattern = 1.0 + 0.06 * np.sin(2 * np.pi * xx / grid_period) * np.sin(
            2 * np.pi * yy / grid_period
        )
        arr = np.clip(arr * pattern[..., None], 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=rng.randint(70, 90))
        return buf.getvalue()


def metadata_inconsistent_edit_transform(raw: bytes, rng: random.Random) -> bytes:
    """Re-saves through a PIL edit so the EXIF thumbnail and the main
    image desync - a real artifact of editing software that regenerates
    the main image but leaves a stale thumbnail behind. The embedded
    thumbnail is derived from the ORIGINAL (pre-edit) image at one JPEG
    quality; the main image is a cropped/re-encoded edit at a
    deliberately different quality, so QT-table and thumbnail-content
    checks (Pillar 2, Phase 2) both have a real signal to find."""
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")

        thumb = img.copy()
        thumb.thumbnail((160, 160), Image.Resampling.LANCZOS)
        thumb_buf = io.BytesIO()
        thumb.save(thumb_buf, format="JPEG", quality=90)

        w, h = img.size
        edited = img.crop((0, 0, max(1, int(w * 0.9)), max(1, int(h * 0.9)))).resize(
            (w, h), Image.Resampling.LANCZOS
        )
        main_buf = io.BytesIO()
        main_quality = rng.randint(55, 75)  # deliberately far from the thumbnail's quality=90
        edited.save(main_buf, format="JPEG", quality=main_quality)

        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": thumb_buf.getvalue()}
        exif_bytes = piexif.dump(exif_dict)
        out = io.BytesIO()
        piexif.insert(exif_bytes, main_buf.getvalue(), out)
        return out.getvalue()
