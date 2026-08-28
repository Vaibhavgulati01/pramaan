"""LightGBM fusion + cross-fitted Mondrian calibration
(PRAMAAN_v2_architecture.md Sec.4 L2).

## Which split does what, and why it matters

This is the part that quietly invalidates the guarantee if done
casually. Sec.4 L3's honesty note #3 requires the calibration split be
used **once, for Learn-then-Test, and never for model selection**. But
isotonic calibration also needs held-out predictions - fitted on a
model's own training scores it learns that the model is far more
accurate than it is, and calibrates the wrong way.

Spending the calibration split on isotonic would therefore burn the very
data LTT needs. So:

    train split        -> fits the LightGBM model
                       -> ALSO fits the calibrator, via K-fold
                          cross-fitting so the calibrator only ever sees
                          out-of-fold (honest) predictions
    calibration split  -> untouched here. Reserved entirely for LTT
                          (Phase 4).
    test split         -> untouched here. Unsealed once, in Phase 6.

Cross-fitting gives honest predictions for every training claim without
spending a separate split: for each fold, fit on the other K-1 and
predict the held-out one. The final model is then refit on all of train.

`FusionModel.fit` takes only training data and will raise if handed
anything labelled otherwise, so the discipline is enforced by the code
rather than by remembering.

## Determinism

Single-threaded, fixed seed, `deterministic=true`, `force_row_wise=true`
(configs/model.yaml). Phase 6 asserts two runs produce byte-identical
metrics, and LightGBM is only reproducible under exactly these settings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pramaan.fusion.calibration import MondrianIsotonicCalibrator
from pramaan.fusion.schema import (
    FEATURE_SCHEMA_VERSION,
    feature_names,
    monotone_constraint_vector,
    validate_schema,
)

logger = logging.getLogger(__name__)

DEFAULT_N_FOLDS = 5


@dataclass
class FusionConfig:
    """LightGBM parameters. Defaults mirror configs/model.yaml."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 20
    seed: int = 1337
    n_folds: int = DEFAULT_N_FOLDS
    apply_monotone_constraints: bool = True

    def to_lgb_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "objective": "binary",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "verbose": -1,
            # Determinism block - all four are required together.
            "num_threads": 1,
            "deterministic": True,
            "force_row_wise": True,
            "seed": self.seed,
            "bagging_seed": self.seed,
            "feature_fraction_seed": self.seed,
        }
        if self.apply_monotone_constraints:
            params["monotone_constraints"] = monotone_constraint_vector()
            # "advanced" respects constraints without the aggressive
            # tree-shape penalty the basic method applies, which
            # otherwise costs real accuracy for no correctness gain.
            params["monotone_constraints_method"] = "advanced"
        return params


@dataclass
class FusionTrainingReport:
    n_train: int
    n_features: int
    schema_version: str
    n_folds: int
    monotone_constraints_applied: int
    calibrator_cells: list[str] = field(default_factory=list)
    oof_brier_uncalibrated: float = 0.0
    oof_brier_calibrated: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "n_train": self.n_train,
            "n_features": self.n_features,
            "schema_version": self.schema_version,
            "n_folds": self.n_folds,
            "monotone_constraints_applied": self.monotone_constraints_applied,
            "calibrator_cells": self.calibrator_cells,
            "oof_brier_uncalibrated": self.oof_brier_uncalibrated,
            "oof_brier_calibrated": self.oof_brier_calibrated,
        }


class ModelNotTrainedError(FileNotFoundError):
    """Raised when a model is loaded before it has been trained.

    A subclass of FileNotFoundError so existing handlers still catch it,
    but carrying the command that fixes it rather than a raw path.
    """


class SplitDisciplineError(RuntimeError):
    """Raised when training is handed data from a split it must not see.

    The calibration split is reserved for LTT and the test split is
    sealed until Phase 6. Enforcing that here means the rule cannot be
    broken by forgetting it.
    """


class FusionModel:
    """LightGBM + Mondrian isotonic, fitted together on the train split."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        validate_schema()
        self.config = config or FusionConfig()
        self.feature_names = feature_names()
        self.schema_version = FEATURE_SCHEMA_VERSION
        # Native Booster, not LGBMClassifier: one code path for both a
        # freshly-fitted model and one reloaded from disk.
        self.booster: lgb.Booster | None = None
        self.calibrator = MondrianIsotonicCalibrator()
        self.report: FusionTrainingReport | None = None

    # --- fitting ------------------------------------------------------

    # Out-of-fold predictions from the last fit, kept so calibration can
    # be reported on them. Reporting calibration on IN-SAMPLE predictions
    # instead is meaningless here: the calibrator was fitted on
    # out-of-fold scores, so pushing in-sample (much sharper) scores
    # through it measures the mismatch between two distributions rather
    # than calibration quality. It produced a misleadingly bad ECE of
    # 0.093 next to an excellent Brier of 0.023, which is the giveaway.
    oof_predictions: np.ndarray | None = None
    oof_calibrated: np.ndarray | None = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        splits: np.ndarray | None = None,
    ) -> FusionModel:
        """Fits the model and its calibrator on TRAIN data only.

        `splits` is optional but checked when given: passing anything
        other than `train` is a programming error serious enough to stop
        the run.
        """
        if splits is not None:
            other = sorted(set(np.asarray(splits).tolist()) - {"train"})
            if other:
                raise SplitDisciplineError(
                    f"FusionModel.fit received rows from split(s) {other}. "
                    "The calibration split is reserved for Learn-then-Test "
                    "(docs/GUARANTEE.md) and test is sealed until Phase 6."
                )

        features = self._align(features)
        labels = np.asarray(labels).astype(int)
        groups = np.asarray(groups)

        if len(np.unique(labels)) < 2:
            raise ValueError("training data contains only one class")

        oof = self._cross_fitted_predictions(features, labels)

        self.calibrator.fit(oof, labels.astype(float), groups)
        self.oof_predictions = oof
        self.oof_calibrated = self.calibrator.predict(oof, groups)

        classifier = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators, **self.config.to_lgb_params()
        )
        classifier.fit(features, labels)
        self.booster = classifier.booster_

        from pramaan.fusion.calibration import brier_score

        self.report = FusionTrainingReport(
            n_train=len(features),
            n_features=len(self.feature_names),
            schema_version=self.schema_version,
            n_folds=self.config.n_folds,
            monotone_constraints_applied=sum(
                1 for c in monotone_constraint_vector() if c != 0
            )
            if self.config.apply_monotone_constraints
            else 0,
            calibrator_cells=self.calibrator.cell_names,
            oof_brier_uncalibrated=brier_score(oof, labels.astype(float)),
            oof_brier_calibrated=brier_score(
                self.calibrator.predict(oof, groups), labels.astype(float)
            ),
        )
        return self

    def _cross_fitted_predictions(
        self, features: pd.DataFrame, labels: np.ndarray
    ) -> np.ndarray:
        """Out-of-fold probabilities for every training row.

        These are what the calibrator is fitted on. Using in-sample
        predictions instead would teach it the model is far sharper than
        it is - the classic way to produce a confidently miscalibrated
        system while every training metric looks excellent.
        """
        oof = np.zeros(len(features), dtype=float)
        folds = StratifiedKFold(
            n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed
        )
        for fold_index, (fit_idx, held_idx) in enumerate(folds.split(features, labels)):
            model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators, **self.config.to_lgb_params()
            )
            model.fit(features.iloc[fit_idx], labels[fit_idx])
            oof[held_idx] = model.predict_proba(features.iloc[held_idx])[:, 1]
            logger.debug("cross-fit fold %d/%d done", fold_index + 1, self.config.n_folds)
        return oof

    # --- scoring ------------------------------------------------------

    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reorders columns to the canonical schema and fails on a
        mismatch. LightGBM's monotone constraints are POSITIONAL, so a
        reordered frame silently applies constraints to the wrong
        features - which would be invisible and completely wrong."""
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"feature frame is missing columns: {missing[:8]}")
        return features.loc[:, list(self.feature_names)]

    def predict_uncalibrated(self, features: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        aligned = self._align(features)
        # The native Booster is used rather than LGBMClassifier so that a
        # freshly-fitted model and one reloaded from disk take exactly the
        # same code path. For a binary objective Booster.predict returns
        # P(y=1) directly.
        return np.asarray(self.booster.predict(aligned), dtype=float)

    def predict(self, features: pd.DataFrame, groups: np.ndarray) -> np.ndarray:
        """Calibrated probability of fraud."""
        raw = self.predict_uncalibrated(features)
        return self.calibrator.predict(raw, np.asarray(groups))

    def feature_importance(self) -> dict[str, float]:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        importances = self.booster.feature_importance(importance_type="gain")
        return dict(zip(self.feature_names, importances.astype(float), strict=True))

    # --- persistence --------------------------------------------------

    def save(self, directory: Path) -> None:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "lightgbm.txt"))

        import pickle

        (directory / "calibrator.pkl").write_bytes(pickle.dumps(self.calibrator))

        # Out-of-fold predictions are a genuine training artifact, not a
        # cache: evaluation at dev/smoke scale reports on them because
        # they are the only honest (non-in-sample) scores available for
        # the train split. Recomputing them would mean refitting K models.
        if self.oof_predictions is not None and self.oof_calibrated is not None:
            np.savez(
                directory / "oof.npz",
                raw=self.oof_predictions,
                calibrated=self.oof_calibrated,
            )
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "feature_names": list(self.feature_names),
                    "config": self.config.__dict__,
                    "report": self.report.as_dict() if self.report else None,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, directory: Path) -> FusionModel:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            # A bare FileNotFoundError here tells the user a path they did
            # not ask about; this tells them the command to run.
            tier = directory.parent.name or "dev"
            raise ModelNotTrainedError(
                f"no trained model at {directory}. "
                f"Train it with: pramaan train --scale {tier}"
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata["schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"model was trained on feature schema {metadata['schema_version']}, "
                f"but this build uses {FEATURE_SCHEMA_VERSION}. Retrain rather than "
                "scoring a vector the model was never fitted for."
            )

        import pickle

        model = cls(FusionConfig(**metadata["config"]))
        model.booster = lgb.Booster(model_file=str(directory / "lightgbm.txt"))
        model.calibrator = pickle.loads((directory / "calibrator.pkl").read_bytes())

        oof_path = directory / "oof.npz"
        if oof_path.exists():
            arrays = np.load(oof_path)
            model.oof_predictions = arrays["raw"]
            model.oof_calibrated = arrays["calibrated"]
        return model
