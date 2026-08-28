"""The three-tier selective policy.

The property that matters most: **the certificate constrains, cost
decides within it**. A cheaper-but-uncertified threshold must never be
selected, and when nothing certifies the policy must refuse to auto-deny
rather than quietly falling back to the best uncertified option — which
would discard the entire point of the risk-control layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pramaan.policy.costs import CostModel
from pramaan.policy.selective import (
    APPROVE,
    DENY,
    REVIEW,
    SelectivePolicy,
    select_policy,
)
from pramaan.risk.certified_set import certify_thresholds

REPO_ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(5)


@pytest.fixture
def costs() -> CostModel:
    return CostModel.from_yaml(REPO_ROOT / "configs" / "costs.yaml")


def _population(n: int = 4000, fraud_rate: float = 0.15, legit_leak: float = 0.004):
    labels = (RNG.uniform(0, 1, n) < fraud_rate).astype(int)
    p = np.where(labels == 1, RNG.uniform(0.85, 1.0, n), RNG.uniform(0.0, 0.3, n))
    leak = (labels == 0) & (RNG.uniform(0, 1, n) < legit_leak)
    p[leak] = RNG.uniform(0.85, 1.0, int(leak.sum()))
    value = RNG.uniform(500, 6000, n)
    return np.clip(p, 0, 1), labels, value


# --- the decision rule ------------------------------------------------


def test_three_tiers_are_produced() -> None:
    policy = SelectivePolicy(t_approve=0.2, t_deny=0.8, certified=True)
    actions = policy.decide(np.array([0.05, 0.5, 0.95]))
    assert actions.tolist() == [APPROVE, REVIEW, DENY]


def test_boundaries_are_inclusive_on_both_sides() -> None:
    policy = SelectivePolicy(t_approve=0.2, t_deny=0.8, certified=True)
    actions = policy.decide(np.array([0.2, 0.8]))
    assert actions.tolist() == [APPROVE, DENY]


def test_band_width_is_the_abstention_region() -> None:
    assert SelectivePolicy(0.2, 0.8, True).band_width() == pytest.approx(0.6)


def test_unreachable_deny_threshold_means_nothing_is_denied() -> None:
    policy = SelectivePolicy(t_approve=0.2, t_deny=1.01, certified=False)
    actions = policy.decide(np.array([0.05, 0.5, 0.99, 1.0]))
    assert DENY not in actions.tolist()


# --- selection under a certificate ------------------------------------


def test_selects_only_from_the_certified_set(costs: CostModel) -> None:
    """The certificate constrains; cost decides within it. A threshold
    outside the certified set is not eligible however cheap it looks."""
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    assert not certified.is_empty, "fixture must certify for this test to mean anything"

    report = select_policy(p, labels, value, costs, certified)
    assert report.policy.certified
    assert report.policy.t_deny in certified.certified_thresholds


def test_records_the_certificate_parameters(costs: CostModel) -> None:
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    report = select_policy(p, labels, value, costs, certified)
    assert report.policy.alpha == 0.10
    assert report.policy.delta == 0.10


def test_chosen_point_is_the_cheapest_eligible_one(costs: CostModel) -> None:
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    report = select_policy(p, labels, value, costs, certified)
    assert report.candidates
    assert report.cost_per_1000 == pytest.approx(report.candidates[0]["cost_per_1000"])


def test_approve_threshold_stays_below_deny_threshold(costs: CostModel) -> None:
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    report = select_policy(p, labels, value, costs, certified)
    assert report.policy.t_approve < report.policy.t_deny


def test_rates_sum_to_one(costs: CostModel) -> None:
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    report = select_policy(p, labels, value, costs, certified)
    total = report.approve_rate + report.review_rate + report.deny_rate
    assert total == pytest.approx(1.0)


# --- refusing to deny without a certificate ---------------------------


def test_no_certificate_means_no_auto_deny(costs: CostModel) -> None:
    """Auto-denying without a certificate is the unbounded liability Sec.1
    opens by describing. The policy must refuse rather than fall back."""
    p, labels, value = _population()
    empty = certify_thresholds(p, labels, alpha=0.0001, delta=0.01)
    assert empty.is_empty

    report = select_policy(p, labels, value, costs, empty)
    assert not report.policy.certified
    assert report.deny_rate == 0.0
    assert "auto-deny disabled" in report.policy.selection_note


def test_none_certificate_is_handled_like_an_empty_one(costs: CostModel) -> None:
    p, labels, value = _population()
    report = select_policy(p, labels, value, costs, None)
    assert not report.policy.certified
    assert report.deny_rate == 0.0


def test_uncertified_counterfactual_is_reported(costs: CostModel) -> None:
    """The price of insisting on a certificate must be visible, not
    implied - so the best uncertified policy is reported alongside."""
    p, labels, value = _population()
    empty = certify_thresholds(p, labels, alpha=0.0001, delta=0.01)
    report = select_policy(p, labels, value, costs, empty)

    assert report.uncertified_best is not None
    # The uncertified search space is a strict superset of the certified
    # one (it may also decline to deny), so it can never be worse. If it
    # ever were, the counterfactual would be measuring the wrong thing.
    assert report.uncertified_best["cost_per_1000"] <= report.cost_per_1000


def test_certificate_can_be_free_under_the_cost_asymmetry(costs: CostModel) -> None:
    """A consequence of C_FP > C_FN worth stating: because denying an
    honest claimant is the expensive mistake, the cost-optimal
    uncertified policy often declines to auto-deny anyway. When that
    happens, insisting on a certificate costs nothing at all.

    This is not guaranteed in general - it depends on the cost model and
    the score distribution - so the test asserts the weaker, always-true
    property and records the observation.
    """
    p, labels, value = _population()
    empty = certify_thresholds(p, labels, alpha=0.0001, delta=0.01)
    report = select_policy(p, labels, value, costs, empty)

    assert report.uncertified_best is not None
    gap = report.cost_per_1000 - report.uncertified_best["cost_per_1000"]
    assert gap >= 0.0  # the certificate is never a bargain, only sometimes free


def test_counterfactual_is_not_reported_when_certified(costs: CostModel) -> None:
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    report = select_policy(p, labels, value, costs, certified)
    assert report.uncertified_best is None


# --- cost behaviour ---------------------------------------------------


def test_selected_policy_beats_denying_everything(costs: CostModel) -> None:
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    report = select_policy(p, labels, value, costs, certified)

    deny_all = costs.cost_per_1000(np.full(len(p), DENY, dtype=object), labels, value)
    assert report.cost_per_1000 < deny_all


def test_selected_policy_beats_approving_everything(costs: CostModel) -> None:
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    report = select_policy(p, labels, value, costs, certified)

    approve_all = costs.cost_per_1000(
        np.full(len(p), APPROVE, dtype=object), labels, value
    )
    assert report.cost_per_1000 < approve_all


def test_report_serialises(costs: CostModel) -> None:
    p, labels, value = _population()
    certified = certify_thresholds(p, labels, alpha=0.10, delta=0.10)
    as_dict = select_policy(p, labels, value, costs, certified).as_dict()
    for key in ("t_approve", "t_deny", "certified", "cost_per_1000", "review_rate"):
        assert key in as_dict


def test_describe_states_whether_it_is_certified() -> None:
    assert "UNCERTIFIED" in SelectivePolicy(0.2, 1.01, certified=False).describe()
    assert "certified alpha" in SelectivePolicy(
        0.2, 0.9, certified=True, alpha=0.03, delta=0.10
    ).describe()
