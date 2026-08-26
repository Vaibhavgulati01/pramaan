"""P1 provenance.

The property under test above all others: **absent provenance is
UNKNOWN, never adverse**. Practically no consumer photo in India carries
C2PA (WhatsApp strips it), so a pillar that treated silence as evidence
would deny essentially every honest claim.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from pramaan.pillars.p1_provenance import (
    ProvenanceFeatures,
    ProvenanceStatus,
    _classify,
    c2pa_available,
    extract_provenance,
)


def _plain_jpeg() -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_image_without_c2pa_is_unknown_not_adverse() -> None:
    f = extract_provenance(_plain_jpeg())
    assert f.status is ProvenanceStatus.UNKNOWN
    assert not f.has_manifest
    assert not f.declares_ai_generation
    assert f.validation_error is None


def test_unknown_contributes_nothing_in_either_direction() -> None:
    """Every feature is a *positive* assertion, so a missing manifest
    yields an all-zero vector rather than anything a model could read as
    risk."""
    as_dict = extract_provenance(_plain_jpeg()).as_dict()
    assert set(as_dict.values()) == {0.0}


def test_garbage_bytes_do_not_raise() -> None:
    """A claim must never fail because its provenance could not be read."""
    f = extract_provenance(b"not an image")
    assert f.status in (ProvenanceStatus.UNKNOWN, ProvenanceStatus.INVALID)


def test_reports_when_the_library_is_unavailable(monkeypatch) -> None:
    """Running without `pramaan[provenance]` is a legitimate deployment
    state, not an error - but it must be visible rather than silently
    identical to 'image had no manifest'."""
    monkeypatch.setattr("pramaan.pillars.p1_provenance.c2pa_available", lambda: False)
    f = extract_provenance(_plain_jpeg())
    assert f.status is ProvenanceStatus.UNKNOWN
    assert not f.library_available


def test_library_is_available_in_this_environment() -> None:
    # Guards against the optional dependency silently disappearing and
    # every provenance result quietly degrading to UNKNOWN.
    assert c2pa_available()


# --- manifest classification -----------------------------------------


def _manifest(*assertion_labels: str, actions: list[str] | None = None) -> str:
    import json

    assertions = [{"label": label} for label in assertion_labels]
    if actions:
        assertions.append(
            {"label": "c2pa.actions", "data": {"actions": [{"action": a} for a in actions]}}
        )
    return json.dumps(
        {
            "active_manifest": "m1",
            "manifests": {"m1": {"assertions": assertions}},
        }
    )


def test_classifies_ai_generated_manifest() -> None:
    f = _classify(_manifest("trainedAlgorithmicMedia", actions=["c2pa.created"]))
    assert f.declares_ai_generation
    assert f.status is ProvenanceStatus.VALID_AI_GENERATED


def test_classifies_camera_capture() -> None:
    f = _classify(_manifest(actions=["c2pa.captured"]))
    assert f.declares_camera_capture
    assert not f.declares_ai_generation
    assert f.status is ProvenanceStatus.VALID_CAMERA


def test_classifies_edited() -> None:
    f = _classify(_manifest(actions=["c2pa.edited"]))
    assert f.declares_editing
    assert f.status is ProvenanceStatus.VALID_EDITED


def test_ai_generation_outranks_editing() -> None:
    """An AI-generated image that was then edited is still AI-generated;
    the stronger claim must win."""
    f = _classify(_manifest("trainedAlgorithmicMedia", actions=["c2pa.edited"]))
    assert f.status is ProvenanceStatus.VALID_AI_GENERATED


def test_valid_manifest_asserting_nothing_recognised_is_unknown() -> None:
    """Signed but uninformative. Valid is not the same as exculpatory."""
    f = _classify(_manifest("com.example.something"))
    assert f.manifest_valid
    assert f.status is ProvenanceStatus.UNKNOWN
    assert not f.declares_ai_generation


def test_unparseable_manifest_is_invalid_not_unknown() -> None:
    """A manifest that is present but broken IS informative, unlike an
    absent one - it must not be collapsed into UNKNOWN."""
    f = _classify("{not valid json")
    assert f.status is ProvenanceStatus.INVALID
    assert f.has_manifest
    assert not f.manifest_valid
    assert f.validation_error


def test_assertion_count_is_reported() -> None:
    f = _classify(_manifest("a", "b", "c"))
    assert f.n_assertions == 3


def test_feature_dict_shape_is_stable() -> None:
    as_dict = ProvenanceFeatures().as_dict()
    assert len(as_dict) == 6
    assert all(isinstance(v, float) for v in as_dict.values())


@pytest.mark.parametrize(
    "status",
    [
        ProvenanceStatus.UNKNOWN,
        ProvenanceStatus.VALID_CAMERA,
        ProvenanceStatus.VALID_AI_GENERATED,
        ProvenanceStatus.INVALID,
    ],
)
def test_status_values_are_stable_strings(status: ProvenanceStatus) -> None:
    # These land in the audit record (Sec.4 L5), so they are part of a
    # contract with anyone reading decision logs.
    assert isinstance(status.value, str)
    assert status.value.isupper()
