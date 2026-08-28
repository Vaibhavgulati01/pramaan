"""Leave-one-out ablations (PRAMAAN_v2_architecture.md Sec.6).

**Each ablation is run twice: on the full corpus and on the ABO-only
subset.** That is not redundancy — it is the difference between a
confounded number and a controlled one.

Every synthetic-fraud claim is sourced from GenImage, and GenImage claims
carry 2.58× the fraud rate of ABO ones, so a forensics feature that
detects "this came from GenImage" earns gain without detecting fraud.
Measured, that inflates the forensics ablation roughly threefold
(−0.1024 vs −0.0335 PR-AUC). The full-corpus figure is an **upper bound**
on the pixel pillars; the ABO-only figure holds source constant.

Reporting only the first would overstate forensics by ~3× and would make
§6's headline claim look settled when it is not. See
`docs/LIMITATIONS.md`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pramaan.fusion.calibration import calibration_metrics
from pramaan.fusion.model import FusionConfig, FusionModel

logger = logging.getLogger(__name__)

# Which feature prefixes each ablation removes. `None` means the
# configuration changes something other than the feature set.
PILLAR_ABLATIONS: dict[str, str | None] = {
    "full": None,
    "no_provenance": "provenance_",
    "no_forensics": "forensics_",
    "no_reuse": "reuse_",
    "no_behaviour": "behaviour_",
    "no_rings": "ring_",
}


@dataclass
class AblationResult:
    name: str
    subset: str  # "all" or "abo_only"
    pr_auc: float
    brier: float
    ece: float
    delta_pr_auc: float = 0.0
    n: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "subset": self.subset,
            "pr_auc": self.pr_auc,
            "brier": self.brier,
            "ece": self.ece,
            "delta_pr_auc": self.delta_pr_auc,
            "n": self.n,
            "note": self.note,
        }


@dataclass
class AblationSuite:
    results: list[AblationResult] = field(default_factory=list)

    def for_subset(self, subset: str) -> list[AblationResult]:
        return [r for r in self.results if r.subset == subset]

    def as_dict(self) -> dict[str, object]:
        return {
            "results": [r.as_dict() for r in self.results],
            "interpretation_warning": (
                "Full-corpus ablations are confounded by source dataset: every "
                "synthetic-fraud claim comes from GenImage, which carries 2.58x the "
                "fraud rate of ABO. Read the 'abo_only' subset for the "
                "source-controlled figure. See docs/LIMITATIONS.md."
            ),
        }


def _run_one(
    name: str,
    subset: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    drop_prefix: str | None,
    config: FusionConfig,
) -> AblationResult:
    frame = features.copy()
    if drop_prefix is not None:
        columns = [c for c in frame.columns if c.startswith(drop_prefix)]
        if not columns:
            raise ValueError(f"ablation {name!r} matched no columns for {drop_prefix!r}")
        # NaN rather than dropped: the model keeps a fixed schema, and
        # LightGBM reads NaN as genuinely missing - the same way a
        # skipped cascade stage presents at serving time.
        frame[columns] = np.nan

    model = FusionModel(config).fit(frame, labels, groups)
    assert model.oof_predictions is not None
    assert model.oof_calibrated is not None

    metrics = calibration_metrics(model.oof_calibrated, labels.astype(float))
    return AblationResult(
        name=name,
        subset=subset,
        pr_auc=float(average_precision_score(labels, model.oof_predictions)),
        brier=metrics.brier,
        ece=metrics.ece,
        n=len(frame),
    )


def run_ablations(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    abo_mask: np.ndarray | None = None,
    config: FusionConfig | None = None,
) -> AblationSuite:
    """Runs every pillar ablation on the full corpus and, if `abo_mask` is
    given, again on the source-controlled subset."""
    config = config or FusionConfig(n_estimators=200)
    suite = AblationSuite()

    subsets: list[tuple[str, np.ndarray | None]] = [("all", None)]
    if abo_mask is not None:
        subsets.append(("abo_only", np.asarray(abo_mask, dtype=bool)))

    for subset_name, mask in subsets:
        subset_features = features if mask is None else features[mask]
        subset_labels = labels if mask is None else labels[mask]
        subset_groups = groups if mask is None else groups[mask]

        if len(np.unique(subset_labels)) < 2:
            logger.warning("subset %s has one class; skipping", subset_name)
            continue

        baseline: float | None = None
        for name, prefix in PILLAR_ABLATIONS.items():
            logger.info("ablation %s [%s]", name, subset_name)
            result = _run_one(
                name, subset_name, subset_features, subset_labels,
                subset_groups, prefix, config,
            )
            if name == "full":
                baseline = result.pr_auc
            else:
                assert baseline is not None
                result.delta_pr_auc = result.pr_auc - baseline
            suite.results.append(result)

    return suite


def run_configuration_ablations(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: FusionConfig | None = None,
) -> list[AblationResult]:
    """Ablations that change the model configuration rather than the
    features: calibration-off and monotone-constraints-off.

    Calibration-off is expected to leave PR-AUC untouched (it is a
    ranking metric, and isotonic regression is monotone) while degrading
    ECE substantially. If PR-AUC moves, something is wrong with the
    calibrator.
    """
    config = config or FusionConfig(n_estimators=200)
    results: list[AblationResult] = []

    full = FusionModel(config).fit(features, labels, groups)
    assert full.oof_predictions is not None and full.oof_calibrated is not None
    base_pr = float(average_precision_score(labels, full.oof_predictions))
    base_metrics = calibration_metrics(full.oof_calibrated, labels.astype(float))
    results.append(
        AblationResult("full", "all", base_pr, base_metrics.brier, base_metrics.ece, 0.0,
                       len(features))
    )

    # Calibration off: score with the raw model output.
    raw_metrics = calibration_metrics(full.oof_predictions, labels.astype(float))
    results.append(
        AblationResult(
            "no_calibration", "all", base_pr, raw_metrics.brier, raw_metrics.ece,
            0.0, len(features),
            note="PR-AUC is unchanged by construction: isotonic calibration is "
                 "monotone, so it cannot alter ranking. ECE is the metric that moves.",
        )
    )

    # Monotone constraints off.
    unconstrained = FusionModel(
        FusionConfig(
            n_estimators=config.n_estimators,
            n_folds=config.n_folds,
            apply_monotone_constraints=False,
        )
    ).fit(features, labels, groups)
    assert unconstrained.oof_predictions is not None
    assert unconstrained.oof_calibrated is not None
    unconstrained_metrics = calibration_metrics(
        unconstrained.oof_calibrated, labels.astype(float)
    )
    unconstrained_pr = float(average_precision_score(labels, unconstrained.oof_predictions))
    results.append(
        AblationResult(
            "no_monotone_constraints", "all", unconstrained_pr,
            unconstrained_metrics.brier, unconstrained_metrics.ece,
            unconstrained_pr - base_pr, len(features),
            note="Constraints buy robustness under shift and coherent reason codes, "
                 "not in-distribution accuracy. A small gain here is expected and is "
                 "not an argument for removing them.",
        )
    )

    return results
