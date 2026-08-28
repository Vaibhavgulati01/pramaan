"""ε-exploration, doubly-robust OPE, and label maturity.

The unifying theme is the selective-labeling problem: you only see
outcomes for claims you approved, so naive evaluation is optimistic in a
way more held-out data cannot fix. These tests check the three mechanisms
that address it actually behave as claimed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pramaan.policy.costs import CostModel
from pramaan.policy.dr_ope import (
    build_reward_model,
    dr_estimate,
    policy_action_matrix,
    support_diagnostics,
)
from pramaan.policy.exploration import (
    apply_epsilon_exploration,
    censoring_report,
    exploration_cost_inr,
)
from pramaan.policy.label_maturity import (
    assess_maturity,
    filter_to_matured,
    maturity_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def costs() -> CostModel:
    return CostModel.from_yaml(REPO_ROOT / "configs" / "costs.yaml")


# --- epsilon exploration ---------------------------------------------


def test_zero_epsilon_changes_nothing() -> None:
    actions = np.array(["DENY"] * 100, dtype=object)
    log = apply_epsilon_exploration(actions, 0.0, np.random.default_rng(0))
    assert log.n_explored == 0
    assert (log.actions == "DENY").all()
    assert (log.propensities == 1.0).all()


def test_exploration_flips_roughly_epsilon_of_denials() -> None:
    actions = np.array(["DENY"] * 10_000, dtype=object)
    log = apply_epsilon_exploration(actions, 0.01, np.random.default_rng(0))
    assert 0.005 < log.exploration_rate < 0.02


def test_only_denials_are_explored() -> None:
    """Exploring APPROVE would mean denying claims believed honest -
    paying the expensive error (C_FP) to learn something already
    observed, since approved claims yield labels anyway."""
    actions = np.array(["APPROVE"] * 500 + ["REVIEW"] * 500, dtype=object)
    log = apply_epsilon_exploration(actions, 0.5, np.random.default_rng(0))
    assert log.n_explored == 0


def test_propensities_reflect_the_logging_policy() -> None:
    actions = np.array(["DENY"] * 1000, dtype=object)
    epsilon = 0.1
    log = apply_epsilon_exploration(actions, epsilon, np.random.default_rng(1))

    assert np.allclose(log.propensities[log.explored], epsilon)
    assert np.allclose(log.propensities[~log.explored], 1 - epsilon)


def test_propensities_are_never_zero() -> None:
    """A zero propensity makes a claim uninformative about
    counterfactuals, which is exactly what DR-OPE needs to avoid."""
    actions = np.array(["DENY", "APPROVE", "REVIEW"] * 100, dtype=object)
    log = apply_epsilon_exploration(actions, 0.01, np.random.default_rng(2))
    assert (log.propensities > 0).all()


def test_invalid_epsilon_rejected() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        apply_epsilon_exploration(np.array(["DENY"]), 1.0, np.random.default_rng(0))


def test_exploration_cost_is_reported_in_rupees(costs: CostModel) -> None:
    """Exploration deliberately approves claims believed fraudulent. That
    costs money, and the number must be visible rather than buried."""
    actions = np.array(["DENY"] * 2000, dtype=object)
    log = apply_epsilon_exploration(actions, 0.05, np.random.default_rng(3))

    labels = np.ones(2000, dtype=int)  # all genuinely fraudulent
    value = np.full(2000, 2000.0)
    report = exploration_cost_inr(log, labels, value, costs)

    assert report["n_explored"] > 0
    assert report["cost_inr"] > 0
    assert report["cost_per_1000_claims"] > 0
    assert report["labels_gained_in_denial_region"] == report["n_explored"]


def test_exploration_savings_are_reported_not_netted_off(costs: CostModel) -> None:
    """Exploring sometimes rescues an honest claimant the policy would
    have wrongly denied. That is reported separately - the gross cost is
    the number a merchant has to approve."""
    actions = np.array(["DENY"] * 2000, dtype=object)
    log = apply_epsilon_exploration(actions, 0.05, np.random.default_rng(4))

    labels = np.zeros(2000, dtype=int)  # all actually legitimate
    value = np.full(2000, 2000.0)
    report = exploration_cost_inr(log, labels, value, costs)

    assert report["cost_inr"] == 0.0
    assert report["savings_inr"] > 0
    assert report["net_inr"] < 0  # exploration paid for itself here


def test_censoring_report_measures_the_blind_spot() -> None:
    actions = np.array(["DENY"] * 200 + ["APPROVE"] * 700 + ["REVIEW"] * 100, dtype=object)
    report = censoring_report(actions)
    assert report["censored_rate"] == pytest.approx(0.2)
    assert report["observed_rate"] == pytest.approx(0.8)


def test_censoring_report_handles_empty() -> None:
    assert censoring_report(np.array([], dtype=object))["censored_rate"] == 0.0


# --- doubly-robust OPE ------------------------------------------------


def _logged_setup(costs: CostModel, n: int = 2000, epsilon: float = 0.05):
    rng = np.random.default_rng(7)
    labels = (rng.uniform(0, 1, n) < 0.15).astype(int)
    value = rng.uniform(500, 5000, n)

    # Logging policy: deny suspected fraud, approve the rest.
    suspected = labels == 1
    base = np.where(suspected, "DENY", "APPROVE").astype(object)
    log = apply_epsilon_exploration(base, epsilon, rng)

    q_hat = build_reward_model(labels, value, costs)
    rewards = -costs.realised_cost(log.actions, labels, value)
    return log, labels, value, q_hat, rewards


def test_dr_recovers_the_value_of_the_logging_policy(costs: CostModel) -> None:
    """Evaluating the logging policy against its own logs must reproduce
    its realised cost - the sanity check that the estimator is wired up
    correctly at all."""
    log, labels, value, q_hat, rewards = _logged_setup(costs)
    target = policy_action_matrix(log.actions)

    estimate = dr_estimate(rewards, log.actions, log.propensities, q_hat, target)
    realised = costs.cost_per_1000(log.actions, labels, value)
    assert estimate.cost_per_1000 == pytest.approx(realised, rel=0.02)


def test_dr_prefers_a_genuinely_better_policy(costs: CostModel) -> None:
    """A policy that reviews everything should evaluate as cheaper than
    one that denies everything, given the FP > FN asymmetry."""
    log, labels, value, q_hat, rewards = _logged_setup(costs)

    review_all = policy_action_matrix(np.full(len(labels), "REVIEW", dtype=object))
    deny_all = policy_action_matrix(np.full(len(labels), "DENY", dtype=object))

    v_review = dr_estimate(rewards, log.actions, log.propensities, q_hat, review_all)
    v_deny = dr_estimate(rewards, log.actions, log.propensities, q_hat, deny_all)
    assert v_review.cost_per_1000 < v_deny.cost_per_1000


def test_dr_decomposes_into_direct_and_correction(costs: CostModel) -> None:
    log, _, _, q_hat, rewards = _logged_setup(costs)
    target = policy_action_matrix(log.actions)
    estimate = dr_estimate(rewards, log.actions, log.propensities, q_hat, target)
    assert estimate.value == pytest.approx(estimate.direct_term + estimate.correction_term)


def test_dr_reports_clipping_and_effective_sample_size(costs: CostModel) -> None:
    """Clipping trades bias for variance, so the rate must be visible;
    ESS shows whether the estimate rests on a few heavy weights."""
    log, _, _, q_hat, rewards = _logged_setup(costs, epsilon=0.01)
    target = policy_action_matrix(np.full(len(rewards), "APPROVE", dtype=object))
    estimate = dr_estimate(rewards, log.actions, log.propensities, q_hat, target, clip=5.0)

    assert 0.0 <= estimate.clipped_fraction <= 1.0
    assert 0 < estimate.effective_sample_size <= estimate.n
    assert estimate.standard_error > 0


def test_dr_rejects_zero_propensity(costs: CostModel) -> None:
    log, _, _, q_hat, rewards = _logged_setup(costs)
    bad = log.propensities.copy()
    bad[0] = 0.0
    with pytest.raises(ValueError, match="propensity of 0"):
        dr_estimate(rewards, log.actions, bad, q_hat, policy_action_matrix(log.actions))


def test_dr_rejects_non_normalised_target(costs: CostModel) -> None:
    log, _, _, q_hat, rewards = _logged_setup(costs)
    bad_target = np.full((len(rewards), 3), 0.5)
    with pytest.raises(ValueError, match="sum to 1"):
        dr_estimate(rewards, log.actions, log.propensities, q_hat, bad_target)


def test_support_diagnostics_flag_off_support_mass(costs: CostModel) -> None:
    """An action the logging policy never took cannot be evaluated. The
    estimate silently extrapolates, so the gap must be reported."""
    log, _, _, _, _ = _logged_setup(costs)
    # Target does the opposite of the logging policy wherever it denied.
    opposite = np.where(log.actions == "DENY", "APPROVE", "DENY").astype(object)
    diagnostics = support_diagnostics(
        log.actions, log.propensities, policy_action_matrix(opposite)
    )
    assert diagnostics["fraction_target_mass_off_support"] > 0.5


# --- label maturity ---------------------------------------------------


def _timestamps(n: int = 100, span_days: int = 300) -> pd.Series:
    base = datetime(2026, 1, 1)
    return pd.Series([base + timedelta(days=i * span_days / n) for i in range(n)])


def test_recent_claims_are_censored() -> None:
    """A claim filed last week has no settled outcome. Treating it as
    legitimate is the easiest way to manufacture a flattering metric."""
    timestamps = _timestamps()
    report = assess_maturity(timestamps, maturity_days=120)
    assert report.n_censored > 0
    assert report.n_matured + report.n_censored == report.n_total


def test_maturity_mask_matches_the_window() -> None:
    timestamps = _timestamps()
    as_of = timestamps.max().to_pydatetime()
    mask = maturity_mask(timestamps, as_of, maturity_days=120)
    cutoff = as_of - timedelta(days=120)
    assert (pd.to_datetime(timestamps[mask]) <= cutoff).all()
    assert (pd.to_datetime(timestamps[~mask]) > cutoff).all()


def test_longer_window_censors_more() -> None:
    timestamps = _timestamps()
    short = assess_maturity(timestamps, maturity_days=30)
    long = assess_maturity(timestamps, maturity_days=200)
    assert long.n_censored > short.n_censored


def test_as_of_defaults_to_the_latest_claim_not_wall_clock() -> None:
    """On a fixed benchmark, wall-clock time is not the relevant clock -
    using it would make the report change every day the tests run."""
    timestamps = _timestamps()
    report = assess_maturity(timestamps)
    assert report.as_of == timestamps.max().to_pydatetime()


def test_filter_returns_both_the_data_and_the_censoring_figure() -> None:
    """Returning them together makes it awkward to filter without also
    having the number that should accompany any downstream metric."""
    frame = pd.DataFrame({"claim_timestamp": _timestamps(), "label": 0})
    matured, report = filter_to_matured(frame, maturity_days=120)
    assert len(matured) == report.n_matured
    assert report.censored_rate > 0
    assert "censored" in report.describe()


def test_report_serialises() -> None:
    report = assess_maturity(_timestamps())
    as_dict = report.as_dict()
    for key in ("maturity_days", "n_matured", "n_censored", "censored_rate"):
        assert key in as_dict
