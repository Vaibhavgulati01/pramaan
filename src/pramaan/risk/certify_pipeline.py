"""Runs Learn-then-Test on the calibration split and records the result.

This is the one place the calibration split is consumed. It:

1. loads the trained fusion model (fitted on train only),
2. scores the calibration split,
3. seals that split's content hash (docs/GUARANTEE.md caveat 3),
4. walks the pre-committed alpha/delta ladder from
   docs/PREREGISTRATION.md, recording every rung including failures,
5. writes `reports/{tier}/certificate.json`.

**Every rung is recorded, not just the one that succeeded.** Reporting
only the α that certified - without the stricter ones that did not -
would misrepresent how hard the guarantee was to obtain, and the ladder
is pre-committed precisely so that walking it is not a search for a
flattering number.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from benchmarks.loaders import load_corpus
from pramaan.fusion.model import FusionModel
from pramaan.fusion.pipeline import extract_features
from pramaan.risk.calibration_seal import write_seal
from pramaan.risk.certified_set import CertifiedSet, walk_alpha_delta_ladder

logger = logging.getLogger(__name__)


def _ladder_from_config(config_path: Path) -> list[tuple[float, float]]:
    config = yaml.safe_load(config_path.read_text())
    return [(rung["alpha"], rung["delta"]) for rung in config["alpha_delta_ladder"]]


def _serialise(outcome: CertifiedSet) -> dict[str, Any]:
    """Full record of one ladder rung, including every threshold tested."""
    return {
        "alpha": outcome.alpha,
        "delta": outcome.delta,
        "min_denied": outcome.min_denied,
        "effective_min_denied": outcome.effective_min_denied,
        "certified": not outcome.is_empty,
        "certified_thresholds": outcome.certified_thresholds,
        "least_conservative": outcome.least_conservative,
        "stopped_at": outcome.stopped_at,
        "stop_reason": outcome.stop_reason,
        "summary": outcome.describe(),
        "results": [asdict(r) for r in outcome.results],
    }


def certify(
    tier: str,
    repo_root: Path,
    use_clip: bool = True,
) -> dict[str, Any]:
    """Certifies on the calibration split and writes the certificate."""
    data_root = repo_root / "data"
    reports_root = repo_root / "reports"

    model = FusionModel.load(data_root / tier / "model")
    extracted = extract_features(tier, data_root=data_root, use_clip=use_clip)

    features, labels, groups = extracted.for_split("calibration")
    if len(features) == 0:
        raise ValueError(f"no calibration rows in tier {tier!r}")

    mask = extracted.splits == "calibration"
    claim_ids = [cid for cid, m in zip(extracted.claim_ids, mask, strict=True) if m]

    # Seal the split before certifying against it, so the recorded hash
    # describes exactly the data the certificate was computed on.
    corpus = load_corpus(tier, data_root=data_root)
    sha_by_claim = {e["claim_id"]: e["output_sha256"] for e in corpus.manifest["entries"]}
    seal = write_seal(
        tier,
        claim_ids,
        labels,
        [sha_by_claim[c] for c in claim_ids],
        reports_root=reports_root,
    )
    logger.info("calibration seal %s (n=%d)", seal.content_sha256[:16], seal.n_claims)

    probabilities = model.predict(features, groups)

    ladder = _ladder_from_config(repo_root / "configs" / "risk.yaml")
    chosen, attempts = walk_alpha_delta_ladder(probabilities, labels, ladder)

    certificate: dict[str, Any] = {
        "tier": tier,
        "scale_warning": (
            "DEV-SCALE: mechanism validation only, not a held-out result. "
            "The reportable certificate comes from the `full` tier "
            "(docs/EVALUATION_PROTOCOL.md)."
        )
        if tier != "full"
        else None,
        "n_calibration": int(labels.size),
        "n_fraud": int((labels == 1).sum()),
        "calibration_seal": seal.as_dict(),
        "ladder": [list(rung) for rung in ladder],
        "certified": chosen is not None,
        "chosen": _serialise(chosen) if chosen is not None else None,
        # Every rung, including the ones that failed.
        "attempts": [_serialise(a) for a in attempts],
        "model_schema_version": model.schema_version,
    }

    out_dir = reports_root / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "certificate.json"
    path.write_text(json.dumps(certificate, indent=2, default=str) + "\n")
    logger.info("wrote %s", path)

    return certificate


def summarise(certificate: dict[str, Any]) -> str:
    lines: list[str] = []
    tier = certificate["tier"]
    lines.append(f"Learn-then-Test [{tier}]")
    if certificate.get("scale_warning"):
        lines.append(f"  {certificate['scale_warning']}")
    lines.append(
        f"  calibration split: n={certificate['n_calibration']} "
        f"(fraud={certificate['n_fraud']})"
    )
    lines.append(f"  seal: {certificate['calibration_seal']['content_sha256'][:16]}...")
    lines.append("")
    lines.append("  ladder (every rung attempted, in pre-committed order):")
    for attempt in certificate["attempts"]:
        mark = "CERTIFIED" if attempt["certified"] else "failed   "
        lines.append(
            f"    alpha={attempt['alpha']:<5} delta={attempt['delta']:<5} {mark}  "
            f"{attempt['summary']}"
        )

    chosen = certificate.get("chosen")
    lines.append("")
    if chosen is None:
        lines.append("  RESULT: no rung certified. That is the published result -")
        lines.append("  see docs/PREREGISTRATION.md for why the ladder is walked in a")
        lines.append("  pre-committed order rather than searched.")
    else:
        best = min(
            (r for r in chosen["results"] if r["threshold"] == chosen["least_conservative"]),
            key=lambda r: r["threshold"],
        )
        lines.append(
            f"  RESULT: at most {chosen['alpha']:.1%} of auto-denied claims are "
            f"legitimate, with {1 - chosen['delta']:.0%} confidence"
        )
        lines.append(
            f"    (t={best['threshold']:.2f}, n_denied={best['n_denied']}, "
            f"empirical FDR={best['conditional_rate']:.4f}, "
            f"HB p={best['conditional_pvalue']:.4f})"
        )
        lines.append(
            f"    unconditional variant: {best['unconditional_rate']:.4f} "
            f"(HB p={best['unconditional_pvalue']:.4f})"
        )
    return "\n".join(lines)
