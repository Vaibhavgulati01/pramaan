"""Signed, reproducible decision records (Sec.4 L5).

Every decision emits a record that can be re-derived and verified later.
The signature covers the decision *content* — claim, decision, reason
codes, model and policy identity — so a record cannot be edited after the
fact without detection.

## What is deliberately absent

- **Raw scores and gradients.** The record carries `p_hat` rounded and
  reason codes, never per-feature SHAP at full precision (`SAFETY.md`).
- **Image bytes.** The record references the claim's image SHA, never
  embeds it — both for size and because the corpus images are
  NC-licensed (`docs/DATA_CARD.md`).
- **Free-text explanation.** Reason codes come from a fixed template
  table; nothing here calls a text generator.

## On signing

HMAC-SHA256 with a deployment key. Symmetric signing is the right choice
for an internal audit log: the verifier is the same organisation that
produced it, and it avoids a key-distribution problem that buys nothing
here. It is NOT a non-repudiation scheme — a party holding the key could
forge a record, so it protects against tampering and accident, not
against a malicious insider. Saying which threat it addresses matters
more than the algorithm.

`docs/THREAT_MODEL.md` covers the append-only log this feeds.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pramaan.audit.reason_codes import ReasonCode

RECORD_VERSION = "1.0.0"


@dataclass
class EvidenceRecord:
    """One adjudication, in the shape Sec.4 L5 specifies."""

    claim_id: str
    decision: str
    p_hat: float
    reason_codes: list[ReasonCode] = field(default_factory=list)

    certified_alpha: float | None = None
    certified_delta: float | None = None
    certified: bool = False

    cascade_stage_exited: int = 0
    compute_ms: float = 0.0

    provenance_status: str = "UNKNOWN"
    model_sha: str = ""
    policy_sha: str = ""
    feature_schema_version: str = ""
    image_sha256: str = ""

    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    record_version: str = RECORD_VERSION

    def to_payload(self) -> dict[str, Any]:
        """The signed content. Ordering is fixed so the signature is
        reproducible - a dict whose key order varied would produce a
        different signature for identical content."""
        return {
            "record_version": self.record_version,
            "claim_id": self.claim_id,
            "decision": self.decision,
            # Rounded on purpose: an audit record documents a decision,
            # and full-precision scores edge toward a scoring oracle.
            "p_hat": round(self.p_hat, 4),
            "certified": self.certified,
            "certified_alpha": self.certified_alpha,
            "certified_delta": self.certified_delta,
            "cascade_stage_exited": self.cascade_stage_exited,
            "compute_ms": round(self.compute_ms, 2),
            "reason_codes": [r.as_dict() for r in self.reason_codes],
            "provenance": {
                "c2pa": (
                    "absent (encoded as UNKNOWN, not adverse)"
                    if self.provenance_status == "UNKNOWN"
                    else self.provenance_status
                )
            },
            "model_sha": self.model_sha,
            "policy_sha": self.policy_sha,
            "feature_schema_version": self.feature_schema_version,
            "image_sha256": self.image_sha256,
            "timestamp": self.timestamp,
        }

    def canonical_json(self) -> str:
        """Deterministic serialisation - sorted keys, no incidental
        whitespace - so the same record always hashes identically."""
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))


def sign_record(record: EvidenceRecord, key: bytes) -> str:
    return hmac.new(key, record.canonical_json().encode("utf-8"), hashlib.sha256).hexdigest()


def verify_record(record: EvidenceRecord, signature: str, key: bytes) -> bool:
    """Constant-time comparison, so a verifier cannot be used as a timing
    oracle to forge a signature byte by byte."""
    return hmac.compare_digest(sign_record(record, key), signature)


@dataclass
class SignedEvidencePack:
    record: EvidenceRecord
    signature: str

    def as_dict(self) -> dict[str, Any]:
        return {"record": self.record.to_payload(), "signature": self.signature}

    @classmethod
    def create(cls, record: EvidenceRecord, key: bytes) -> SignedEvidencePack:
        return cls(record=record, signature=sign_record(record, key))

    def verify(self, key: bytes) -> bool:
        return verify_record(self.record, self.signature, key)


def build_record(
    claim_id: str,
    decision: str,
    p_hat: float,
    reason_codes: list[ReasonCode],
    *,
    certified: bool,
    certified_alpha: float | None,
    certified_delta: float | None,
    cascade_stage_exited: int,
    compute_ms: float,
    provenance_status: str,
    model_sha: str,
    policy_sha: str,
    feature_schema_version: str,
    image_sha256: str,
) -> EvidenceRecord:
    return EvidenceRecord(
        claim_id=claim_id,
        decision=decision,
        p_hat=p_hat,
        reason_codes=reason_codes,
        certified=certified,
        certified_alpha=certified_alpha,
        certified_delta=certified_delta,
        cascade_stage_exited=cascade_stage_exited,
        compute_ms=compute_ms,
        provenance_status=provenance_status,
        model_sha=model_sha,
        policy_sha=policy_sha,
        feature_schema_version=feature_schema_version,
        image_sha256=image_sha256,
    )
