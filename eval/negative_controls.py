"""Negative controls (PRAMAAN_v2_architecture.md Sec.6).

Leakage audits prove the splits are clean *by construction*. These prove
it *empirically*, by checking that the pipeline fails when it should.

A test suite that only ever confirms things work cannot distinguish a
working system from one that is leaking. Three controls, each targeting a
different way a fraud pipeline silently cheats:

1. **Label shuffle** — retrain on permuted labels. PR-AUC must collapse
   to base rate. If it does not, something in the feature pipeline is
   carrying label information it should not have.
2. **Random features** — replace the vector with noise. Same expectation.
   Distinguishes "the features carry signal" from "the evaluation
   harness is broken".
3. **Temporal constraint disabled** — re-run the reuse graph without the
   strictly-earlier rule and report the inflated number it *would* have
   produced. This is the most persuasive of the three: it demonstrates
   the common bug is understood because it was measured, not merely
   avoided.

Unlike the ablations, these are **not pre-registered hypotheses** — they
validate the methodology rather than test a claim about the
architecture, so they run and are reported at `dev` scale freely
(`docs/EVALUATION_PROTOCOL.md`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pramaan.fusion.model import FusionConfig, FusionModel

logger = logging.getLogger(__name__)


@dataclass
class ControlResult:
    name: str
    description: str
    pr_auc: float
    baseline_pr_auc: float
    prevalence: float
    passed: bool
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "pr_auc": self.pr_auc,
            "baseline_pr_auc": self.baseline_pr_auc,
            "prevalence": self.prevalence,
            "passed": self.passed,
            "note": self.note,
        }


def _fit_and_score(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: FusionConfig,
) -> float:
    model = FusionModel(config).fit(features, labels, groups)
    assert model.oof_predictions is not None
    return float(average_precision_score(labels, model.oof_predictions))


def label_shuffle_control(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    baseline_pr_auc: float,
    seed: int = 1337,
    tolerance: float = 0.05,
) -> ControlResult:
    """Retrain on permuted labels; PR-AUC must fall to prevalence."""
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(labels).copy()
    rng.shuffle(shuffled)

    prevalence = float(np.mean(labels))
    pr_auc = _fit_and_score(features, shuffled, groups, FusionConfig(n_estimators=150))
    passed = pr_auc <= prevalence + tolerance

    return ControlResult(
        name="label_shuffle",
        description="retrained on permuted labels; PR-AUC must collapse to prevalence",
        pr_auc=pr_auc,
        baseline_pr_auc=baseline_pr_auc,
        prevalence=prevalence,
        passed=passed,
        note=(
            ""
            if passed
            else "FAILED: the pipeline finds signal in shuffled labels, which means "
            "something is carrying label information it should not have"
        ),
    )


def random_feature_control(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    baseline_pr_auc: float,
    seed: int = 1337,
    tolerance: float = 0.05,
) -> ControlResult:
    """Replace every feature with noise; PR-AUC must fall to prevalence.

    Distinguishes "our features carry signal" from "our evaluation
    harness reports a high number regardless".
    """
    rng = np.random.default_rng(seed)
    noise = pd.DataFrame(
        rng.standard_normal(features.shape),
        columns=features.columns,
        index=features.index,
    )

    prevalence = float(np.mean(labels))
    pr_auc = _fit_and_score(noise, labels, groups, FusionConfig(n_estimators=150))
    passed = pr_auc <= prevalence + tolerance

    return ControlResult(
        name="random_features",
        description="every feature replaced with noise; PR-AUC must collapse to prevalence",
        pr_auc=pr_auc,
        baseline_pr_auc=baseline_pr_auc,
        prevalence=prevalence,
        passed=passed,
        note="" if passed else "FAILED: the harness reports signal where none exists",
    )


@dataclass
class TemporalLeakMeasurement:
    """What disabling the temporal constraint would have bought.

    Reported as a positive number: the inflation in match rate a leaky
    reuse index produces. A *small* value here would be a mildly
    embarrassing but honest result - it would mean the temporal guard
    buys less than the architecture claims.
    """

    honest_match_rate: float
    leaked_match_rate: float
    honest_legit_fp_rate: float
    leaked_legit_fp_rate: float
    n_claims: int

    @property
    def inflation(self) -> float:
        return self.leaked_match_rate - self.honest_match_rate

    def as_dict(self) -> dict[str, float]:
        return {
            "honest_match_rate": self.honest_match_rate,
            "leaked_match_rate": self.leaked_match_rate,
            "match_rate_inflation": self.inflation,
            "honest_legit_fp_rate": self.honest_legit_fp_rate,
            "leaked_legit_fp_rate": self.leaked_legit_fp_rate,
            "n_claims": float(self.n_claims),
        }


def temporal_leak_control(
    phashes: dict[str, int],
    claim_order: list[str],
    labels_by_claim: dict[str, int],
    claimants: dict[str, str],
    merchants: dict[str, str],
    timestamps: dict[str, datetime],
    hamming_threshold: int = 2,
) -> TemporalLeakMeasurement:
    """Measures the inflation a future-inclusive reuse index produces.

    The honest index only matches strictly-earlier claims. The leaked one
    matches against the whole corpus, including claims that had not
    happened yet - the exact bug `tests/test_pillars_p3_temporal.py`
    guards against. Running both and reporting the gap demonstrates the
    bug is understood because it was measured.
    """
    from pramaan.pillars.p3_reuse import TemporalReuseIndex, hamming_distance

    honest_index = TemporalReuseIndex(hamming_threshold=hamming_threshold)
    honest_hits = 0
    honest_legit_hits = 0
    n_legit = 0

    for claim_id in claim_order:
        features = honest_index.query_then_add(
            claim_id,
            claimants[claim_id],
            merchants[claim_id],
            timestamps[claim_id],
            phashes[claim_id],
        )
        if features.matched_prior_claim:
            honest_hits += 1
            if labels_by_claim[claim_id] == 0:
                honest_legit_hits += 1
        if labels_by_claim[claim_id] == 0:
            n_legit += 1

    # The leaky version: every claim compared against every other,
    # regardless of order. This is what a naive implementation does.
    leaked_hits = 0
    leaked_legit_hits = 0
    for claim_id in claim_order:
        matched = any(
            other != claim_id
            and hamming_distance(phashes[claim_id], phashes[other]) <= hamming_threshold
            for other in claim_order
        )
        if matched:
            leaked_hits += 1
            if labels_by_claim[claim_id] == 0:
                leaked_legit_hits += 1

    n = len(claim_order)
    return TemporalLeakMeasurement(
        honest_match_rate=honest_hits / n if n else 0.0,
        leaked_match_rate=leaked_hits / n if n else 0.0,
        honest_legit_fp_rate=honest_legit_hits / n_legit if n_legit else 0.0,
        leaked_legit_fp_rate=leaked_legit_hits / n_legit if n_legit else 0.0,
        n_claims=n,
    )
