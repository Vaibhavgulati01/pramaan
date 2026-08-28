"""Regenerates every number and figure, one command
(PRAMAAN_v2_architecture.md Sec.7 non-negotiables).

`metrics.json` is the single source of truth. README figures are injected
from it by `scripts/inject_metrics.py`, and CI checks the committed
README matches. Nothing is hand-typed.

## Tier discipline

Which split is evaluated depends on the tier, and this is the one place
it could go wrong:

- `smoke` / `dev` evaluate the **train** split's out-of-fold predictions.
  They are mechanism validation and are labelled `DEV — not a held-out
  result` everywhere they surface.
- `full` evaluates the **test** split, once. That is the only tier whose
  output may be quoted as a result.

`run_manifest.json` records the git SHA, config hash, seed and dataset
SHAs alongside, so any number can be traced to the exact state that
produced it.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from benchmarks.baselines.models import build_baselines
from benchmarks.loaders import load_corpus
from eval.ablations import run_ablations, run_configuration_ablations
from eval.bootstrap import bootstrap_metric, bootstrap_scalar
from eval.negative_controls import label_shuffle_control, random_feature_control
from pramaan.fusion.calibration import evaluate_calibration
from pramaan.fusion.model import FusionModel
from pramaan.fusion.pipeline import extract_features
from pramaan.policy.costs import CostModel
from pramaan.policy.exploration import apply_epsilon_exploration, exploration_cost_inr
from pramaan.policy.label_maturity import assess_maturity
from pramaan.policy.selective import APPROVE, DENY, REVIEW, select_policy
from pramaan.risk.certified_set import certify_thresholds

logger = logging.getLogger(__name__)

# Which split each tier reports on. See the module docstring.
EVAL_SPLIT = {"smoke": "train", "dev": "train", "full": "test"}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git is not a reason to fail an eval
        return "unknown"


def _precision_at_recall(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    eligible = precision[recall >= target]
    return float(eligible.max()) if eligible.size else 0.0


def _recall_at_precision(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    eligible = recall[precision >= target]
    return float(eligible.max()) if eligible.size else 0.0


def run_eval(
    tier: str,
    repo_root: Path,
    use_clip: bool = True,
    n_resamples: int = 2000,
    skip_slow: bool = False,
) -> dict[str, Any]:
    """Runs the full evaluation for one tier and writes reports/{tier}/."""
    started = time.time()
    data_root = repo_root / "data"
    out_dir = repo_root / "reports" / tier
    out_dir.mkdir(parents=True, exist_ok=True)

    split_name = EVAL_SPLIT[tier]
    model = FusionModel.load(data_root / tier / "model")
    extracted = extract_features(tier, data_root=data_root, use_clip=use_clip)
    corpus = load_corpus(tier, data_root=data_root)

    features, labels, groups = extracted.for_split(split_name)
    mask = extracted.splits == split_name
    claim_ids = [c for c, m in zip(extracted.claim_ids, mask, strict=True) if m]

    by_id = corpus.claims.set_index("claim_id")
    order_value = np.array([float(by_id.loc[c, "order_value_inr"]) for c in claim_ids])
    entries = {e["claim_id"]: e for e in corpus.manifest["entries"]}
    abo_mask = np.array(["GenImage" not in entries[c]["source_dataset"] for c in claim_ids])

    # For dev/smoke the honest scores are out-of-fold; for full they are
    # genuine held-out predictions on a split the model never saw.
    if split_name == "train":
        if model.oof_predictions is None or model.oof_calibrated is None:
            raise RuntimeError(
                f"the {tier} model has no saved out-of-fold predictions. They are "
                "the only honest scores available for the train split - in-sample "
                "predictions would overstate every metric here. Retrain with: "
                f"pramaan train --scale {tier}"
            )
        if len(model.oof_predictions) != len(labels):
            raise RuntimeError(
                f"saved out-of-fold predictions cover {len(model.oof_predictions)} "
                f"claims but the train split has {len(labels)}. The corpus was "
                f"rebuilt after training; retrain with: pramaan train --scale {tier}"
            )
        scores = model.oof_predictions
        calibrated = model.oof_calibrated
    else:
        scores = model.predict_uncalibrated(features)
        calibrated = model.predict(features, groups)

    metrics: dict[str, Any] = {
        "tier": tier,
        "split_evaluated": split_name,
        "is_reportable_result": tier == "full",
        "scale_warning": None
        if tier == "full"
        else (
            f"{tier.upper()} — not a held-out result. Evaluated on the train "
            "split's out-of-fold predictions for mechanism validation only. "
            "See docs/EVALUATION_PROTOCOL.md."
        ),
        "n_claims": int(len(labels)),
        "prevalence": float(labels.mean()),
    }

    # --- claim-level ranking metrics ---------------------------------
    pr_auc = bootstrap_metric(average_precision_score, labels, scores, n_resamples)
    metrics["pr_auc"] = pr_auc.as_dict()
    metrics["p_at_90r"] = _precision_at_recall(labels, scores, 0.90)
    metrics["r_at_95p"] = _recall_at_precision(labels, scores, 0.95)

    # --- calibration --------------------------------------------------
    calibration = evaluate_calibration(calibrated, labels, groups)
    metrics["calibration"] = {
        "overall": calibration.overall.as_dict(),
        "per_group": {k: v.as_dict() for k, v in calibration.per_group.items()},
        "worst_groups": [
            {"group": name, **m.as_dict()} for name, m in calibration.worst_groups(5)
        ],
    }

    # --- baselines ----------------------------------------------------
    baseline_rows = []
    for baseline in build_baselines():
        baseline.fit(features, labels)
        baseline_scores = baseline.predict_proba(features)
        # A constant scorer has no ranking, so PR-AUC degenerates to
        # prevalence; reported anyway because the FLOOR is the point.
        interval = bootstrap_metric(
            average_precision_score, labels, baseline_scores, min(n_resamples, 500)
        )
        baseline_rows.append(
            {
                "name": baseline.name,
                "description": baseline.description,
                **interval.as_dict(),
            }
        )
    baseline_rows.append({"name": "pramaan_full", "description": "", **pr_auc.as_dict()})
    metrics["baselines"] = baseline_rows

    # --- policy + cost ------------------------------------------------
    costs = CostModel.from_yaml(repo_root / "configs" / "costs.yaml")
    certified = certify_thresholds(calibrated, labels, alpha=0.10, delta=0.10)
    policy_report = select_policy(calibrated, labels, order_value, costs, certified)
    actions = policy_report.policy.decide(calibrated)

    per_claim_cost = costs.realised_cost(actions, labels, order_value)
    cost_ci = bootstrap_scalar(per_claim_cost, min(n_resamples, 1000))
    metrics["policy"] = policy_report.as_dict()
    metrics["cost"] = {
        "per_1000_claims": policy_report.cost_per_1000,
        "per_claim_ci": cost_ci.as_dict(),
        "review_load_pct": policy_report.review_rate * 100.0,
        "baselines_per_1000": {
            name: costs.cost_per_1000(
                np.full(len(labels), action, dtype=object), labels, order_value
            )
            for name, action in
            [("approve_all", APPROVE), ("deny_all", DENY), ("review_all", REVIEW)]
        },
    }

    # --- exploration + maturity ---------------------------------------
    log = apply_epsilon_exploration(actions, costs.epsilon, np.random.default_rng(1337))
    metrics["exploration"] = {
        **log.as_dict(),
        **exploration_cost_inr(log, labels, order_value, costs),
    }
    timestamps = pd.to_datetime([by_id.loc[c, "claim_timestamp"] for c in claim_ids])
    metrics["label_maturity"] = assess_maturity(pd.Series(timestamps)).as_dict()

    # --- cascade cost -------------------------------------------------
    exit_stages = extracted.exit_stages[mask]
    metrics["cascade"] = {
        "mean_compute_ms": float(extracted.compute_ms[mask].mean()),
        "stage_exit_distribution": {
            f"stage_{stage}": float((exit_stages == stage).mean())
            for stage in (1, 2, 3)
        },
    }

    # --- ablations + negative controls --------------------------------
    if not skip_slow:
        logger.info("running ablations (both full-corpus and source-controlled)")
        suite = run_ablations(features, labels, groups, abo_mask=abo_mask)
        metrics["ablations"] = suite.as_dict()
        metrics["configuration_ablations"] = [
            r.as_dict() for r in run_configuration_ablations(features, labels, groups)
        ]

        logger.info("running negative controls")
        base = float(average_precision_score(labels, scores))
        metrics["negative_controls"] = [
            label_shuffle_control(features, labels, groups, base).as_dict(),
            random_feature_control(features, labels, groups, base).as_dict(),
        ]

    metrics["elapsed_seconds"] = round(time.time() - started, 1)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")

    manifest = {
        "git_sha": _git_sha(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tier": tier,
        "split_evaluated": split_name,
        "seed": model.config.seed,
        "feature_schema_version": model.schema_version,
        "model_config": asdict(model.config),
        "corpus_manifest_counts": corpus.manifest["counts"],
        "n_resamples": n_resamples,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("wrote %s", out_dir / "metrics.json")
    return metrics


def summarise(metrics: dict[str, Any]) -> str:
    lines = [f"Evaluation [{metrics['tier']}] on the {metrics['split_evaluated']} split"]
    if metrics.get("scale_warning"):
        lines.append(f"  {metrics['scale_warning']}")
    lines.append(f"  claims: {metrics['n_claims']}  prevalence: {metrics['prevalence']:.3f}")

    pr = metrics["pr_auc"]
    lines.append(
        f"  PR-AUC {pr['point']:.4f} [{pr['ci_lower']:.4f}, {pr['ci_upper']:.4f}]"
    )
    cal = metrics["calibration"]["overall"]
    lines.append(f"  Brier {cal['brier']:.4f} | ECE {cal['ece']:.4f} | MCE {cal['mce']:.4f}")

    cost = metrics["cost"]
    lines.append("")
    lines.append(f"  Rs {cost['per_1000_claims']:,.0f} per 1000 claims "
                 f"at {cost['review_load_pct']:.1f}% review load")
    for name, value in cost["baselines_per_1000"].items():
        lines.append(f"    vs {name:12s} Rs {value:>12,.0f}")

    if "ablations" in metrics:
        lines.append("")
        lines.append("  Ablations (delta PR-AUC; abo_only holds source constant):")
        for result in metrics["ablations"]["results"]:
            if result["name"] == "full":
                continue
            lines.append(
                f"    {result['name']:22s} [{result['subset']:8s}] "
                f"{result['delta_pr_auc']:+.4f}"
            )

    if "negative_controls" in metrics:
        lines.append("")
        lines.append("  Negative controls:")
        for control in metrics["negative_controls"]:
            mark = "PASS" if control["passed"] else "FAIL"
            lines.append(
                f"    {control['name']:18s} {mark}  PR-AUC {control['pr_auc']:.4f} "
                f"(prevalence {control['prevalence']:.4f})"
            )

    return "\n".join(lines)
