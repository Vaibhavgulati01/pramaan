"""The calibration-split seal.

docs/GUARANTEE.md caveat 3 claims the calibration split is used once and
never changes. This is what converts that from an assertion into a
checkable property, so the tests here are about the ways the mechanism
could quietly stop working: a reordered split reading as changed, a
genuinely changed split reading as unchanged, or a reseal silently
overwriting the record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pramaan.risk.calibration_seal import (
    CalibrationSealBroken,
    compute_calibration_hash,
    read_seal,
    verify_seal,
    write_seal,
)

IDS = ["c3", "c1", "c2"]
LABELS = np.array([1, 0, 0])
SHAS = ["sha_c3", "sha_c1", "sha_c2"]


def test_hash_is_order_independent() -> None:
    """A reordered split is the same split. Order sensitivity would make
    the seal fire on rebuilds that changed nothing."""
    a = compute_calibration_hash(IDS, LABELS, SHAS)
    order = [2, 0, 1]
    b = compute_calibration_hash(
        [IDS[i] for i in order], LABELS[order], [SHAS[i] for i in order]
    )
    assert a == b


def test_hash_changes_when_a_label_changes() -> None:
    flipped = LABELS.copy()
    flipped[0] = 0
    assert compute_calibration_hash(IDS, LABELS, SHAS) != compute_calibration_hash(
        IDS, flipped, SHAS
    )


def test_hash_changes_when_an_image_changes() -> None:
    """The image bytes are part of the split's identity: a rebuilt corpus
    with different transport randomness is a different calibration set."""
    other = [*SHAS[:-1], "sha_rebuilt"]
    assert compute_calibration_hash(IDS, LABELS, SHAS) != compute_calibration_hash(
        IDS, LABELS, other
    )


def test_hash_changes_when_a_claim_is_added() -> None:
    assert compute_calibration_hash(IDS, LABELS, SHAS) != compute_calibration_hash(
        [*IDS, "c4"], np.append(LABELS, 1), [*SHAS, "sha_c4"]
    )


def test_hash_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_calibration_hash(IDS, np.array([1, 0]), SHAS)


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    seal = write_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)
    loaded = read_seal("dev", reports_root=tmp_path)
    assert loaded is not None
    assert loaded.content_sha256 == seal.content_sha256
    assert loaded.n_claims == 3
    assert loaded.n_fraud == 1


def test_verify_passes_on_unchanged_split(tmp_path: Path) -> None:
    write_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)
    assert verify_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)


def test_verify_fails_loudly_on_a_changed_split(tmp_path: Path) -> None:
    """The whole point: a certificate computed against the sealed version
    no longer applies, so this must never be a warning."""
    write_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)
    changed = LABELS.copy()
    changed[1] = 1
    with pytest.raises(CalibrationSealBroken, match="does not match its seal"):
        verify_seal("dev", IDS, changed, SHAS, reports_root=tmp_path)


def test_verify_fails_when_no_seal_exists(tmp_path: Path) -> None:
    with pytest.raises(CalibrationSealBroken, match="no calibration seal"):
        verify_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)


def test_resealing_a_changed_split_raises(tmp_path: Path) -> None:
    """Re-running certify against a rebuilt calibration split must not
    quietly replace the record - that would defeat the mechanism."""
    write_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)
    changed = LABELS.copy()
    changed[0] = 0
    with pytest.raises(CalibrationSealBroken, match="has changed since it was sealed"):
        write_seal("dev", IDS, changed, SHAS, reports_root=tmp_path)


def test_resealing_an_identical_split_is_idempotent(tmp_path: Path) -> None:
    first = write_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)
    second = write_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)
    assert first.content_sha256 == second.content_sha256
    assert first.sealed_at == second.sealed_at  # original timestamp preserved


def test_explicit_overwrite_is_allowed(tmp_path: Path) -> None:
    """A deliberate reseal must be possible - but only deliberately."""
    write_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)
    changed = LABELS.copy()
    changed[0] = 0
    reseal = write_seal("dev", IDS, changed, SHAS, reports_root=tmp_path, overwrite=True)
    assert reseal.content_sha256 == compute_calibration_hash(IDS, changed, SHAS)


def test_tiers_are_sealed_independently(tmp_path: Path) -> None:
    write_seal("dev", IDS, LABELS, SHAS, reports_root=tmp_path)
    write_seal("full", IDS, LABELS, SHAS, reports_root=tmp_path)
    assert read_seal("dev", reports_root=tmp_path) is not None
    assert read_seal("full", reports_root=tmp_path) is not None
    assert (tmp_path / "dev" / "calibration_seal.json").exists()
    assert (tmp_path / "full" / "calibration_seal.json").exists()
