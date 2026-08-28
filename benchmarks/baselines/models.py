"""Outward-facing baselines (PRAMAAN_v2_architecture.md Sec.6).

The point of these is to compare *outward*, not just to ablate PRAMAAN
against itself. A system that only beats its own ablations has shown
nothing about whether the architecture was worth building.

Each baseline answers a specific objection a reviewer would raise:

| Baseline | The objection it answers |
|---|---|
| approve-all / deny-all | "what does doing nothing cost?" |
| rules engine | "merchants already run rules; is ML worth it?" |
| CLIP linear probe | "just use a foundation model" |
| ResNet-50 + linear head | "just use a synthetic-image detector" |
| behaviour-only GBM | "just use tabular fraud ML" |

## On naming

The fourth is called **"ResNet-50 features + linear head, in the spirit
of CNNSpot"** everywhere, and never "CNNSpot". It is not a faithful
reimplementation - CNNSpot has a specific training recipe, augmentation
scheme and pretrained checkpoint. Calling it CNNSpot would be a small
overclaim, and small overclaims are what make a reader start doubting the
large ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pramaan.fusion.model import FusionConfig


@dataclass
class BaselineResult:
    name: str
    description: str
    scores: np.ndarray


class Baseline:
    """Common interface: fit on train, score anything."""

    name = "baseline"
    description = ""

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> Baseline:
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class ConstantBaseline(Baseline):
    """Approve-all (score 0) or deny-all (score 1). Trivial floors.

    deny-all is not merely trivial: under the FP > FN cost asymmetry it is
    catastrophically expensive, which is itself the argument for the
    abstention band.
    """

    def __init__(self, score: float, name: str) -> None:
        self.score = score
        self.name = name
        self.description = f"constant score {score}"

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.score, dtype=float)


class RulesEngineBaseline(Baseline):
    """What merchants actually run today: count-and-value thresholds.

    Deliberately hand-specified rather than fitted, because that is what
    makes it a fair representation of the incumbent. Fitting the
    thresholds would produce a small logistic model wearing a rules
    costume, and beating that would prove nothing about beating rules.
    """

    name = "rules_engine"
    description = "claim-count + order-value thresholds, hand-specified"

    def __init__(
        self,
        max_prior_claims: int = 2,
        max_prior_claims_30d: int = 1,
        escalation_ratio: float = 3.0,
    ) -> None:
        self.max_prior_claims = max_prior_claims
        self.max_prior_claims_30d = max_prior_claims_30d
        self.escalation_ratio = escalation_ratio

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        def column(name: str) -> np.ndarray:
            return np.nan_to_num(features[name].to_numpy(dtype=float), nan=0.0)

        triggers = (
            (column("behaviour_n_prior_claims") > self.max_prior_claims).astype(float)
            + (column("behaviour_n_prior_claims_30d") > self.max_prior_claims_30d).astype(float)
            + (column("behaviour_order_value_vs_prior_mean") > self.escalation_ratio).astype(float)
            + (column("reuse_matched_prior_claim") > 0).astype(float)
        )
        # Scaled to [0, 1] so it is comparable on a PR curve. A rules
        # engine has no calibrated probability, and pretending otherwise
        # would flatter it in the calibration metrics.
        return np.clip(triggers / 4.0, 0.0, 1.0)


class FeatureSubsetBaseline(Baseline):
    """Logistic regression on one pillar's features only.

    Used for 'just use a foundation model' (reuse features, which carry
    the CLIP similarity), 'just use a synthetic-image detector'
    (forensics features, which carry the pixel statistics), and 'just use
    tabular fraud ML' (behaviour features).

    Logistic regression rather than a second GBM on purpose: the question
    is whether the *signal* suffices, not whether a big model can squeeze
    it, and a linear probe is the standard way to ask that.
    """

    def __init__(self, prefix: str, name: str, description: str) -> None:
        self.prefix = prefix
        self.name = name
        self.description = description
        self._model: LogisticRegression | None = None
        self._columns: list[str] = []

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> FeatureSubsetBaseline:
        self._columns = [c for c in features.columns if c.startswith(self.prefix)]
        if not self._columns:
            raise ValueError(f"no features with prefix {self.prefix!r}")

        matrix = np.nan_to_num(features[self._columns].to_numpy(dtype=float), nan=0.0)
        self._model = LogisticRegression(max_iter=2000, random_state=FusionConfig().seed)
        self._model.fit(matrix, np.asarray(labels).astype(int))
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name} is not fitted")
        matrix = np.nan_to_num(features[self._columns].to_numpy(dtype=float), nan=0.0)
        return self._model.predict_proba(matrix)[:, 1]


def build_baselines() -> list[Baseline]:
    """The standard set, in the order Sec.6's table presents them."""
    return [
        ConstantBaseline(0.0, "approve_all"),
        ConstantBaseline(1.0, "deny_all"),
        RulesEngineBaseline(),
        FeatureSubsetBaseline(
            "reuse_",
            "clip_probe",
            "linear probe on reuse/CLIP features - 'just use a foundation model'",
        ),
        FeatureSubsetBaseline(
            "forensics_",
            "resnet_style_pixel_probe",
            "linear probe on pixel/container statistics - ResNet-50 features + "
            "linear head, in the spirit of CNNSpot (NOT a faithful reimplementation)",
        ),
        FeatureSubsetBaseline(
            "behaviour_",
            "behaviour_only_gbm",
            "linear probe on behavioural features - 'just use tabular fraud ML'",
        ),
    ]
