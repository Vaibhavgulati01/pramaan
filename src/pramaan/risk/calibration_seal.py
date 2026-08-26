"""Records and verifies that the calibration split was used once and
never changed (PRAMAAN_v2_architecture.md Sec.4 L3, honesty note #3).

The claim in docs/GUARANTEE.md is that the calibration split is consumed
exactly once, by Learn-then-Test, and never for model selection. That is
easy to assert and easy to violate by accident - retune a threshold, peek
at a metric, rebuild the corpus with a different seed, and the
certificate silently becomes a statement about data that has already been
optimised against.

So the split's content is hashed the first time LTT consumes it, and the
hash is committed. CI recomputes it and fails on any mismatch. This
converts "we didn't tune on the calibration set" from an assertion into
a checkable property - the same move the frozen test-set SHA makes for
the test split.

The seal covers the claim ids, their labels, and the image bytes each
claim resolved to. It deliberately does NOT cover the model's
predictions: those change legitimately whenever the model is retrained,
and sealing them would fail for the wrong reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SEAL_FILENAME = "calibration_seal.json"


@dataclass(frozen=True)
class CalibrationSeal:
    """A fingerprint of the calibration split at the moment LTT used it."""

    tier: str
    n_claims: int
    n_fraud: int
    content_sha256: str
    sealed_at: str
    note: str = (
        "Hash of the calibration split's claim ids, labels and image SHAs. "
        "If this changes, the split was rebuilt or reused and any certificate "
        "computed against the earlier version no longer applies."
    )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_calibration_hash(
    claim_ids: list[str],
    labels: np.ndarray,
    image_shas: list[str],
) -> str:
    """Order-independent content hash of the calibration split.

    Sorted by claim id so that a reordered - but otherwise identical -
    split does not read as a different one. A genuine change (a claim
    added, a label corrected, an image rebuilt) does change the hash.
    """
    labels = np.asarray(labels).astype(int)
    if not (len(claim_ids) == len(labels) == len(image_shas)):
        raise ValueError("claim_ids, labels and image_shas must be the same length")

    rows = sorted(
        f"{cid}\t{int(label)}\t{sha}"
        for cid, label, sha in zip(claim_ids, labels, image_shas, strict=True)
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def seal_path(tier: str, reports_root: Path = Path("reports")) -> Path:
    return reports_root / tier / SEAL_FILENAME


def write_seal(
    tier: str,
    claim_ids: list[str],
    labels: np.ndarray,
    image_shas: list[str],
    reports_root: Path = Path("reports"),
    overwrite: bool = False,
) -> CalibrationSeal:
    """Records the seal. Refuses to overwrite an existing one unless asked
    explicitly - silently resealing would defeat the entire mechanism."""
    path = seal_path(tier, reports_root)
    content_hash = compute_calibration_hash(claim_ids, labels, image_shas)

    if path.exists() and not overwrite:
        existing = read_seal(tier, reports_root)
        if existing is not None and existing.content_sha256 != content_hash:
            raise CalibrationSealBroken(
                f"calibration split for tier {tier!r} has changed since it was sealed.\n"
                f"  sealed:  {existing.content_sha256}\n"
                f"  current: {content_hash}\n"
                "Any certificate computed against the sealed version no longer "
                "applies. See docs/GUARANTEE.md, caveat 3."
            )
        return existing if existing is not None else _write(path, tier, labels, content_hash)

    return _write(path, tier, labels, content_hash)


def _write(path: Path, tier: str, labels: np.ndarray, content_hash: str) -> CalibrationSeal:
    labels = np.asarray(labels).astype(int)
    seal = CalibrationSeal(
        tier=tier,
        n_claims=int(labels.size),
        n_fraud=int((labels == 1).sum()),
        content_sha256=content_hash,
        sealed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal.as_dict(), indent=2) + "\n")
    return seal


def read_seal(tier: str, reports_root: Path = Path("reports")) -> CalibrationSeal | None:
    path = seal_path(tier, reports_root)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return CalibrationSeal(
        tier=data["tier"],
        n_claims=data["n_claims"],
        n_fraud=data["n_fraud"],
        content_sha256=data["content_sha256"],
        sealed_at=data["sealed_at"],
        note=data.get("note", ""),
    )


class CalibrationSealBroken(RuntimeError):
    """Raised when the calibration split no longer matches its seal.

    Never downgraded to a warning: a changed calibration split means the
    certificate describes data that no longer exists.
    """


def verify_seal(
    tier: str,
    claim_ids: list[str],
    labels: np.ndarray,
    image_shas: list[str],
    reports_root: Path = Path("reports"),
) -> CalibrationSeal:
    """Recomputes the hash and compares it to the recorded seal."""
    existing = read_seal(tier, reports_root)
    if existing is None:
        raise CalibrationSealBroken(
            f"no calibration seal recorded for tier {tier!r}. Run "
            "`pramaan certify` to seal the split before relying on a certificate."
        )

    current = compute_calibration_hash(claim_ids, labels, image_shas)
    if current != existing.content_sha256:
        raise CalibrationSealBroken(
            f"calibration split for tier {tier!r} does not match its seal.\n"
            f"  sealed:  {existing.content_sha256} ({existing.sealed_at})\n"
            f"  current: {current}\n"
            "See docs/GUARANTEE.md, caveat 3."
        )
    return existing
