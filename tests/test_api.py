"""The API surface.

The load-bearing test here is `test_response_never_leaks_a_score`. The
response boundary is a security control, not a design preference: an
attacker who can read a continuous score can hill-climb against it, and
`SAFETY.md` commits publicly to not providing one. A refactor that
helpfully adds `p_hat` to the response would break that promise silently,
so it is asserted rather than trusted.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pramaan.api.app import _policy_from_certificate, app, state

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    # Context-manager form: TestClient only fires lifespan startup inside
    # `with`, and without it the model never loads - which silently
    # skipped the security tests that matter most here.
    with TestClient(app) as test_client:
        yield test_client


def _image_bytes() -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _post(client: TestClient) -> object:
    return client.post(
        "/adjudicate",
        data={
            "claim_id": "claim_test_1",
            "claimant_id": "claimant_1",
            "merchant_id": "merchant_1",
            "category": "electronics",
            "price_band": "mid",
            "order_value_inr": "2500",
        },
        files={"image": ("claim.jpg", _image_bytes(), "image/jpeg")},
    )


# --- health -----------------------------------------------------------


def test_healthz_always_answers(client: TestClient) -> None:
    """A service that will not start is worse than one that reports
    itself unhealthy - /healthz must say what is missing."""
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body and "model_loaded" in body


def test_healthz_reports_the_schema_version_when_loaded(client: TestClient) -> None:
    body = client.get("/healthz").json()
    if body["model_loaded"]:
        assert body["feature_schema_version"]


# --- the response boundary -------------------------------------------


def test_response_never_leaks_a_score(client: TestClient) -> None:
    """THE security test. A continuous score is a hill-climbing target."""
    response = _post(client)
    if response.status_code == 503:
        pytest.skip("no model loaded in this environment")

    body = response.json()
    forbidden = {
        "p_hat", "probability", "score", "raw_score", "logit",
        "shap_values", "gradient", "hamming", "clip_similarity",
        "matched_claim_id", "distance", "similarity",
    }
    assert not (forbidden & set(body)), f"response leaked: {forbidden & set(body)}"


def test_reason_codes_carry_no_full_precision_attribution(client: TestClient) -> None:
    """Reason codes are returned, but rounded - full-precision SHAP edges
    toward the gradient oracle SAFETY.md rules out."""
    response = _post(client)
    if response.status_code == 503:
        pytest.skip("no model loaded")

    for code in response.json()["reason_codes"]:
        assert set(code) <= {"code", "text", "shap"}
        assert len(str(code["shap"]).split(".")[-1]) <= 3


def test_decision_is_one_of_three_tiers(client: TestClient) -> None:
    response = _post(client)
    if response.status_code == 503:
        pytest.skip("no model loaded")
    assert response.json()["decision"] in {"APPROVE", "REVIEW", "DENY"}


def test_response_is_signed(client: TestClient) -> None:
    response = _post(client)
    if response.status_code == 503:
        pytest.skip("no model loaded")
    assert len(response.json()["signature"]) == 64


def test_response_states_whether_it_is_certified(client: TestClient) -> None:
    response = _post(client)
    if response.status_code == 503:
        pytest.skip("no model loaded")
    body = response.json()
    assert isinstance(body["certified"], bool)
    if not body["certified"]:
        # An uncertified policy must never auto-deny.
        assert body["decision"] != "DENY"


def test_missing_model_returns_503_not_500(client: TestClient) -> None:
    """A missing model is an operational state, not a crash."""
    original = state.model
    state.model = None
    try:
        assert _post(client).status_code == 503
    finally:
        state.model = original


# --- /explain ---------------------------------------------------------


def test_explain_documents_the_boundary(client: TestClient) -> None:
    """Served as an endpoint rather than left to docs, because a caller
    integrating against this API should meet the boundary immediately."""
    body = client.get("/explain").json()
    assert "deliberately_not_returned" in body
    assert any("probability" in item for item in body["deliberately_not_returned"])
    assert "why" in body


def test_explain_states_provenance_handling(client: TestClient) -> None:
    body = client.get("/explain").json()
    assert "never as adverse" in body["absent_provenance"]


def test_explain_notes_reason_codes_are_templated(client: TestClient) -> None:
    assert "never model-generated" in client.get("/explain").json()["reason_codes_are"]


# --- policy loading ---------------------------------------------------


def test_missing_certificate_disables_auto_deny(tmp_path: Path) -> None:
    """Serving a denial without a guarantee is the unbounded liability
    Sec.1 opens by describing."""
    policy = _policy_from_certificate(tmp_path / "nonexistent.json")
    assert not policy.certified
    assert policy.t_deny > 1.0


def test_uncertified_certificate_disables_auto_deny(tmp_path: Path) -> None:
    import json

    path = tmp_path / "certificate.json"
    path.write_text(json.dumps({"certified": False, "chosen": None, "attempts": []}))
    policy = _policy_from_certificate(path)
    assert not policy.certified
    assert policy.t_deny > 1.0
    assert "auto-deny disabled" in policy.selection_note


def test_certified_certificate_sets_a_reachable_threshold(tmp_path: Path) -> None:
    import json

    path = tmp_path / "certificate.json"
    path.write_text(
        json.dumps(
            {
                "certified": True,
                "chosen": {
                    "alpha": 0.03,
                    "delta": 0.10,
                    "least_conservative": 0.87,
                    "results": [],
                },
            }
        )
    )
    policy = _policy_from_certificate(path)
    assert policy.certified
    assert policy.t_deny == pytest.approx(0.87)
    assert policy.alpha == 0.03


def test_dev_certificate_currently_disables_auto_deny() -> None:
    """The committed dev certificate certified nothing, so the served
    policy must refuse to auto-deny. This is the end-to-end version of
    the guarantee's honesty."""
    path = REPO_ROOT / "reports" / "dev" / "certificate.json"
    if not path.exists():
        pytest.skip("no dev certificate committed")
    policy = _policy_from_certificate(path)
    assert not policy.certified
    assert policy.t_deny > 1.0
