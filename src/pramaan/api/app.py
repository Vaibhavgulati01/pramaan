"""FastAPI service: /adjudicate, /explain, /healthz (Sec.4, Sec.10).

One service, one endpoint set. Sec.10 rules out microservices, and this
is the whole product surface.

## The response boundary is a security control, not a design preference

`/adjudicate` returns a tiered decision and templated reason codes.
It does **not** return `p_hat`, per-feature SHAP values, gradients,
distances, similarities, match counts, or the identity of a matched
prior claim. An attacker who can read a continuous score can hill-climb
against it: submit, observe, adjust, repeat. A tiered decision with
three levels gives far less to climb.

`SAFETY.md` commits to this publicly, and
`tests/test_api.py::test_response_never_leaks_a_score` enforces it.

## Statefulness

The reuse index and behavioural aggregates accumulate across claims and
are strictly backward-looking. A real deployment would back them with a
shared store; this service holds them in memory, which is honest for a
single-process demo and is stated rather than implied.
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from pramaan import __version__
from pramaan.audit.evidence_pack import SignedEvidencePack, build_record
from pramaan.audit.reason_codes import compute_shap_values, render_reason_codes
from pramaan.cascade.cascade import FEATURE_KEYS, Cascade, CascadeConfig
from pramaan.fusion.model import FusionModel
from pramaan.policy.selective import SelectivePolicy

logger = logging.getLogger(__name__)

# Signing key for evidence records. A deployment supplies its own; the
# fallback exists so the demo runs, and says so rather than looking like
# a secret.
SIGNING_KEY = os.environ.get(
    "PRAMAAN_SIGNING_KEY", "development-key-not-for-production"
).encode()


class DecisionResponse(BaseModel):
    """What a caller receives.

    Deliberately narrow. Every field here is something an adjudicator
    needs; nothing here helps an attacker calibrate.
    """

    # `model_sha` collides with pydantic's protected `model_` namespace.
    # The field name is part of the audit-record contract in Sec.4 L5, so
    # the namespace is relaxed rather than the field renamed.
    model_config = ConfigDict(protected_namespaces=())

    claim_id: str
    decision: str = Field(description="APPROVE, REVIEW or DENY")
    reason_codes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Templated, deterministic. Never free-text generated.",
    )
    certified: bool = Field(
        description="Whether the deny threshold carries a Learn-then-Test certificate"
    )
    certified_alpha: float | None = None
    certified_delta: float | None = None
    provenance: str = Field(
        default="UNKNOWN",
        description="Absent C2PA is UNKNOWN, never adverse",
    )
    cascade_stage_exited: int = 0
    compute_ms: float = 0.0
    model_sha: str = ""
    feature_schema_version: str = ""
    signature: str = Field(description="HMAC over the audit record")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    version: str
    model_loaded: bool
    feature_schema_version: str | None = None
    policy: str | None = None
    indexed_claims: int = 0


class AppState:
    """Loaded model, policy and accumulating indices."""

    def __init__(self) -> None:
        self.model: FusionModel | None = None
        self.policy: SelectivePolicy | None = None
        self.cascade: Cascade | None = None
        self.model_sha: str = ""

    def load(self, tier: str, repo_root: Path) -> None:
        model_dir = repo_root / "data" / tier / "model"
        self.model = FusionModel.load(model_dir)
        self.model_sha = hashlib.sha256(
            (model_dir / "lightgbm.txt").read_bytes()
        ).hexdigest()[:16]

        certificate_path = repo_root / "reports" / tier / "certificate.json"
        self.policy = _policy_from_certificate(certificate_path)
        self.cascade = Cascade(CascadeConfig())
        logger.info("loaded %s model %s; policy: %s", tier, self.model_sha,
                    self.policy.describe())


def _policy_from_certificate(path: Path) -> SelectivePolicy:
    """Builds the serving policy from the committed certificate.

    **If nothing certified, auto-deny is disabled** (`t_deny` set beyond
    reach) rather than falling back to an uncertified threshold. Serving
    a denial without a guarantee is precisely the unbounded liability
    Sec.1 opens by describing.
    """
    import json

    if not path.exists():
        return SelectivePolicy(
            t_approve=0.02, t_deny=1.01, certified=False,
            selection_note="no certificate found; auto-deny disabled",
        )

    certificate = json.loads(path.read_text())
    chosen = certificate.get("chosen")
    if not certificate.get("certified") or chosen is None:
        return SelectivePolicy(
            t_approve=0.02, t_deny=1.01, certified=False,
            selection_note="no rung certified; auto-deny disabled, claims go to review",
        )

    return SelectivePolicy(
        t_approve=0.02,
        t_deny=float(chosen["least_conservative"]),
        certified=True,
        alpha=chosen["alpha"],
        delta=chosen["delta"],
        selection_note="cost-optimal member of the certified set",
    )


state = AppState()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Loads the model on startup.

    A missing model is NOT fatal: the service starts and /healthz reports
    exactly what is absent, which is more useful operationally than a
    process that refuses to boot.
    """
    tier = os.environ.get("PRAMAAN_SCALE", "dev")
    repo_root = Path(__file__).resolve().parents[3]
    try:
        state.load(tier, repo_root)
    except FileNotFoundError:
        logger.warning("no %s model found; /healthz will report unhealthy", tier)
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="PRAMAAN",
    version=__version__,
    description=(
        "Risk-controlled selective adjudication of claim evidence. "
        "Returns a tiered decision and templated reason codes - never raw "
        "scores or gradients (see SAFETY.md)."
    ),
)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.model is not None else "no model loaded",
        version=__version__,
        model_loaded=state.model is not None,
        feature_schema_version=state.model.schema_version if state.model else None,
        policy=state.policy.describe() if state.policy else None,
        indexed_claims=len(state.cascade.reuse_index) if state.cascade else 0,
    )


@app.post("/adjudicate", response_model=DecisionResponse)
async def adjudicate(
    claim_id: str = Form(...),
    claimant_id: str = Form(...),
    merchant_id: str = Form(...),
    category: str = Form("electronics"),
    price_band: str = Form("mid"),
    order_value_inr: float = Form(...),
    image: UploadFile = File(...),
) -> DecisionResponse:
    if state.model is None or state.policy is None or state.cascade is None:
        raise HTTPException(status_code=503, detail="model not loaded; see /healthz")

    raw = await image.read()
    now = datetime.now(UTC).replace(tzinfo=None)

    result = state.cascade.process(
        claim_id=claim_id,
        claimant_id=claimant_id,
        merchant_id=merchant_id,
        category=category,
        timestamp=now,
        order_date=now,
        order_value=order_value_inr,
        raw_bytes=raw,
    )

    frame = pd.DataFrame([result.features], columns=list(FEATURE_KEYS))
    group = np.array([f"{category}|{price_band}"])
    probability = float(state.model.predict(frame, group)[0])
    decision = str(state.policy.decide(np.array([probability]))[0])

    shap_values = compute_shap_values(
        state.model.booster, frame.to_numpy()[0], FEATURE_KEYS
    )
    reason_codes = render_reason_codes(shap_values, result.features)

    record = build_record(
        claim_id=claim_id,
        decision=decision,
        p_hat=probability,
        reason_codes=reason_codes,
        certified=state.policy.certified,
        certified_alpha=state.policy.alpha,
        certified_delta=state.policy.delta,
        cascade_stage_exited=int(result.exited_at_stage),
        compute_ms=result.compute_ms,
        provenance_status=(
            result.provenance.status.value if result.provenance else "UNKNOWN"
        ),
        model_sha=state.model_sha,
        policy_sha=hashlib.sha256(
            state.policy.describe().encode()
        ).hexdigest()[:16],
        feature_schema_version=state.model.schema_version,
        image_sha256=hashlib.sha256(raw).hexdigest(),
    )
    pack = SignedEvidencePack.create(record, SIGNING_KEY)

    # NOTE: `probability` is computed above and deliberately NOT returned.
    return DecisionResponse(
        claim_id=claim_id,
        decision=decision,
        reason_codes=[r.as_dict() for r in reason_codes],
        certified=state.policy.certified,
        certified_alpha=state.policy.alpha,
        certified_delta=state.policy.delta,
        provenance=record.provenance_status,
        cascade_stage_exited=record.cascade_stage_exited,
        compute_ms=round(result.compute_ms, 1),
        model_sha=state.model_sha,
        feature_schema_version=state.model.schema_version,
        signature=pack.signature,
    )


@app.get("/explain")
def explain() -> dict[str, Any]:
    """What the system does and what it deliberately will not tell you.

    Served as an endpoint rather than left to documentation because the
    response boundary is a security property, and a caller integrating
    against this API should meet it immediately.
    """
    return {
        "decisions": ["APPROVE", "REVIEW", "DENY"],
        "guarantee": (
            "Distribution-free finite-sample bound on the false-denial rate via "
            "Learn-then-Test. See docs/GUARANTEE.md for the three caveats that "
            "qualify it."
        ),
        "certified": state.policy.certified if state.policy else False,
        "policy": state.policy.describe() if state.policy else None,
        "reason_codes_are": "templated and deterministic, never model-generated text",
        "deliberately_not_returned": [
            "raw fraud probability",
            "per-feature SHAP values at full precision",
            "gradients",
            "perceptual distances or CLIP similarities",
            "the identity of a matched prior claim",
        ],
        "why": (
            "A continuous score is a hill-climbing target. Returning tiered "
            "decisions and templated reasons keeps the service from acting as a "
            "black-box evasion oracle (SAFETY.md)."
        ),
        "absent_provenance": "encoded as UNKNOWN, never as adverse",
    }
