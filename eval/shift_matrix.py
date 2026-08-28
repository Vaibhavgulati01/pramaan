"""The shift & robustness matrix — the signature table
(PRAMAAN_v2_architecture.md Sec.6).

## Two experiments per condition, not one

The spec conflates two questions that have different answers and very
different operational meanings:

1. **Frozen calibration.** The certificate was computed in-distribution;
   the data has now shifted. *Does the guarantee still hold?* This is the
   question that matters in production, where shift arrives without
   announcing itself.
2. **Re-certified.** Learn-then-Test is re-run on shifted calibration
   data. *Could a guarantee be recovered if we knew about the shift?*
   This is the question that matters for whether the architecture is
   salvageable under a known attack.

Running both and reporting the gap is more informative than either alone.
A condition where (1) fails and (2) succeeds says "detectable and
fixable"; one where both fail says "this breaks the approach".

## On filling in real failures

A table of eight green ticks is not believable and a good reader will
assume it was fabricated. The perturbations below are applied to our own
held-out inputs to measure our own degradation, which is standard
defensive evaluation (`SAFETY.md`).
"""

from __future__ import annotations

import io
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# --- the perturbations ------------------------------------------------


def _recompress(quality: int) -> Callable[[bytes, random.Random], bytes]:
    def perturb(raw: bytes, rng: random.Random) -> bytes:
        with Image.open(io.BytesIO(raw)) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=quality)
            return buf.getvalue()

    return perturb


def strip_metadata(raw: bytes, rng: random.Random) -> bytes:
    with Image.open(io.BytesIO(raw)) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)  # no exif=
        return buf.getvalue()


def centre_crop_90(raw: bytes, rng: random.Random) -> bytes:
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        w, h = img.size
        cw, ch = int(w * 0.9), int(h * 0.9)
        left, top = (w - cw) // 2, (h - ch) // 2
        buf = io.BytesIO()
        img.crop((left, top, left + cw, top + ch)).save(buf, format="JPEG", quality=92)
        return buf.getvalue()


def screenshot_round_trip(raw: bytes, rng: random.Random) -> bytes:
    """PNG round-trip then re-JPEG, as a screenshot-and-resend would do."""
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        png = io.BytesIO()
        img.save(png, format="PNG")
        png.seek(0)
        with Image.open(png) as reopened:
            buf = io.BytesIO()
            reopened.convert("RGB").save(buf, format="JPEG", quality=88)
            return buf.getvalue()


def colour_jitter_rotate(raw: bytes, rng: random.Random) -> bytes:
    from PIL import ImageEnhance

    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB").rotate(2.0, expand=False, fillcolor=(255, 255, 255))
        img = ImageEnhance.Color(img).enhance(1.15)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def identity(raw: bytes, rng: random.Random) -> bytes:
    return raw


# Ordered as Sec.6's table presents them. `unseen_generator` is not a
# pixel perturbation - it is handled by the generator-holdout split
# itself, and is listed here so the matrix is complete.
SHIFT_CONDITIONS: dict[str, Callable[[bytes, random.Random], bytes] | None] = {
    "in_distribution": identity,
    "unseen_generator_families": None,  # supplied by the split, not a transform
    "jpeg_q60": _recompress(60),
    "jpeg_q40": _recompress(40),
    "metadata_stripped": strip_metadata,
    "centre_crop_90": centre_crop_90,
    "screenshot_round_trip": screenshot_round_trip,
    "colour_jitter_rotate": colour_jitter_rotate,
}


# --- the matrix -------------------------------------------------------


@dataclass
class ShiftCell:
    """One condition, both experiments."""

    condition: str
    n: int
    pr_auc: float
    recall_at_operating_point: float

    # Experiment 1: certificate frozen in-distribution, applied here.
    frozen_realised_fdr: float
    frozen_certificate_holds: bool

    # Experiment 2: Learn-then-Test re-run on shifted calibration data.
    recertified: bool
    recertified_alpha: float | None = None
    recertified_threshold: float | None = None
    note: str = ""

    @property
    def recoverable(self) -> bool:
        """Frozen certificate broke, but re-certification rescued it."""
        return (not self.frozen_certificate_holds) and self.recertified

    def as_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "n": self.n,
            "pr_auc": self.pr_auc,
            "recall": self.recall_at_operating_point,
            "frozen_realised_fdr": self.frozen_realised_fdr,
            "frozen_certificate_holds": self.frozen_certificate_holds,
            "recertified": self.recertified,
            "recertified_alpha": self.recertified_alpha,
            "recertified_threshold": self.recertified_threshold,
            "recoverable": self.recoverable,
            "note": self.note,
        }


@dataclass
class ShiftMatrix:
    alpha: float
    delta: float
    cells: list[ShiftCell] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "cells": [c.as_dict() for c in self.cells],
            "n_frozen_holds": sum(1 for c in self.cells if c.frozen_certificate_holds),
            "n_recoverable": sum(1 for c in self.cells if c.recoverable),
            "n_conditions": len(self.cells),
        }

    def to_markdown(self) -> str:
        lines = [
            "| Condition | n | PR-AUC | Realised FDR | Certificate holds? | Re-certifiable? |",
            "|---|---|---|---|---|---|",
        ]
        for cell in self.cells:
            holds = "✅" if cell.frozen_certificate_holds else "❌"
            recert = (
                "—"
                if cell.frozen_certificate_holds
                else ("✅ recovered" if cell.recertified else "❌")
            )
            lines.append(
                f"| {cell.condition} | {cell.n} | {cell.pr_auc:.3f} | "
                f"{cell.frozen_realised_fdr:.4f} | {holds} | {recert} |"
            )
        return "\n".join(lines)


def evaluate_frozen_certificate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    alpha: float,
) -> tuple[float, bool, int]:
    """Applies an in-distribution threshold to shifted data.

    Returns (realised FDR, whether it stayed within alpha, n_denied).
    Deliberately checks the *realised* rate rather than re-running the
    statistical test: the question is whether the guarantee held in
    practice, not whether a new one could be issued.
    """
    denied = np.asarray(probabilities) >= threshold
    n_denied = int(denied.sum())
    if n_denied == 0:
        # Denying nothing cannot violate a false-denial bound, but it also
        # demonstrates nothing. Flagged by n_denied so a reader can see it.
        return 0.0, True, 0

    realised = float((np.asarray(labels)[denied] == 0).mean())
    return realised, realised <= alpha, n_denied
