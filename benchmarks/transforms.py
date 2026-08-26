"""Image transforms that construct PRAMAAN-Bench-v1's claims from a pool
of source images (PRAMAAN_v2_architecture.md Sec.5 composition table).

Every claim is built in **two stages**, and keeping them separate is what
stops the benchmark from being trivially solvable:

1. **Fraud transform** - what the fraudster did (recycled someone's
   photo, re-shot a screen, edited so the EXIF thumbnail desyncs). Absent
   for `legit_real_photo`, and for `fraud_catalog_photo`, whose fraud is
   contextual - the SKU's own listing image submitted as damage evidence
   - rather than a pixel edit.

2. **Transport** (`apply_transport`) - how the image reached the
   merchant. Applied to **every** claim with parameters drawn
   independently of its class.

Stage 2 is not cosmetic. Sec.5 asks for a messaging-app degradation
pipeline so "the legit class isn't separable by file cleanliness alone",
and that requirement is symmetric: if only some classes are resized and
recompressed, then resolution and compression history separate the
classes for reasons that have nothing to do with fraud. An early build of
this corpus had exactly that leak - synthetic-fraud images came through
at 512x512 while every other class sat near 256x256, so image dimensions
alone were a near-perfect fraud detector. Drawing transport identically
for all classes removes that by construction.

Real claims arrive by more than one route, so transport is a mixture: an
image sent through a messaging app loses its metadata, while one uploaded
straight from a desktop keeps it. That mixture is the reason
`fraud_metadata_inconsistent_edit` is detectable at all in some claims and
invisible in others - a property of the domain, documented in
docs/LIMITATIONS.md rather than engineered away.
"""

from __future__ import annotations

import io
import random

import numpy as np
import piexif
from PIL import Image, ImageEnhance

# Transport resamples every claim into this range, independent of class.
# Both ABO (256px) and GenImage (512px) sources land in the same
# distribution, so source resolution cannot leak the label.
_TRANSPORT_MIN_DIM = 384
_TRANSPORT_MAX_DIM = 640

# Share of claims that arrive via a metadata-stripping messaging app; the
# rest are direct uploads that preserve EXIF.
_MESSAGING_APP_SHARE = 0.7


def apply_transport(raw: bytes, rng: random.Random) -> bytes:
    """Simulates delivery to the merchant: resize to a class-independent
    target, re-encode as JPEG, and (for the messaging-app route) strip
    metadata. Always the last step in building a claim's bytes.

    Resizing here is unconditional and two-directional - it upscales a
    small source as readily as it downscales a large one - because a
    one-directional `thumbnail()` preserves whatever spread the source
    pools happened to have, which is precisely the leak this exists to
    close.
    """
    target = rng.randint(_TRANSPORT_MIN_DIM, _TRANSPORT_MAX_DIM)
    via_messaging_app = rng.random() < _MESSAGING_APP_SHARE

    with Image.open(io.BytesIO(raw)) as img:
        exif = img.info.get("exif")
        img = img.convert("RGB")

        w, h = img.size
        scale = target / max(w, h)
        img = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.Resampling.LANCZOS,
        )

        buf = io.BytesIO()
        if via_messaging_app:
            # No exif= kwarg -> metadata stripped, as WhatsApp et al. do.
            img.save(buf, format="JPEG", quality=rng.randint(75, 90))
        elif exif:
            img.save(buf, format="JPEG", quality=rng.randint(85, 95), exif=exif)
        else:
            img.save(buf, format="JPEG", quality=rng.randint(85, 95))
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
