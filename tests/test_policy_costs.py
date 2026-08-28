"""The rupee cost model.

The claim under test is the one the whole abstention band rests on: at
Indian order values a false positive costs MORE than a false negative
once churn is priced. If that ever stopped holding, the architecture's
central argument would go with it — so it is asserted, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pramaan.policy.costs import CostModel

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def costs() -> CostModel:
    return CostModel.from_yaml(REPO_ROOT / "configs" / "costs.yaml")


def test_loads_the_committed_config(costs: CostModel) -> None:
    assert costs.fn_fixed == 180.0
    assert costs.fp_fixed == 250.0
    assert costs.p_churn == 0.35
    assert costs.ltv == 3000.0
    assert costs.review_cost == 40.0


def test_false_positives_cost_more_at_every_plausible_order_value(costs: CostModel) -> None:
    """THE load-bearing claim (Sec.4 L4). Both costs carry order_value
    identically, so it cancels and the asymmetry is carried entirely by
    the fixed terms plus churn."""
    values = np.array([100, 500, 2_000, 8_000, 50_000], dtype=float)
    assert (costs.cost_false_positive(values) > costs.cost_false_negative(values)).all()


def test_crossover_reports_none_when_fp_is_always_dearer(costs: CostModel) -> None:
    assert costs.crossover_order_value() is None


def test_crossover_exists_when_churn_is_priced_away() -> None:
    """Sanity on the mechanism: with no churn cost and a large FN addon,
    false negatives become the dearer error and the helper must say so
    rather than silently reporting None."""
    cheap_churn = CostModel(fn_fixed=5_000.0, fp_fixed=100.0, p_churn=0.0, ltv=0.0)
    assert cheap_churn.crossover_order_value() is not None


def test_churn_dominates_the_fixed_false_negative_cost(costs: CostModel) -> None:
    """The churn term alone (Rs 1,050) is ~6x the entire fixed cost of a
    false negative. This is why maximising recall destroys value."""
    assert costs.churn_component > 5 * costs.fn_fixed


def test_review_cost_is_flat(costs: CostModel) -> None:
    """An analyst's time does not scale with the order value."""
    small = costs.cost_review(np.array([100.0]))
    large = costs.cost_review(np.array([100_000.0]))
    assert small[0] == large[0] == costs.review_cost


# --- expected costs and the optimal action ---------------------------


def test_expected_costs_have_the_right_shape(costs: CostModel) -> None:
    p = np.array([0.1, 0.5, 0.9])
    value = np.array([1000.0, 1000.0, 1000.0])
    expected = costs.expected_costs(p, value)
    assert set(expected) == {"APPROVE", "DENY", "REVIEW"}
    for arr in expected.values():
        assert arr.shape == (3,)


def test_confident_fraud_is_denied(costs: CostModel) -> None:
    actions = costs.optimal_action(np.array([0.999]), np.array([2000.0]))
    assert actions[0] == "DENY"


def test_confident_legitimate_is_approved(costs: CostModel) -> None:
    actions = costs.optimal_action(np.array([0.0001]), np.array([2000.0]))
    assert actions[0] == "APPROVE"


def test_uncertain_claims_go_to_review(costs: CostModel) -> None:
    """The abstention band is *derived*: a claim belongs in REVIEW exactly
    when paying a human beats acting on the model in expectation."""
    actions = costs.optimal_action(np.array([0.5]), np.array([5000.0]))
    assert actions[0] == "REVIEW"


def test_the_deny_bar_is_higher_than_the_approve_bar(costs: CostModel) -> None:
    """Because a false positive is dearer, the model must be more certain
    to deny than to approve. This asymmetry IS the architecture's
    argument, expressed as behaviour."""
    p = np.round(np.arange(0.001, 1.0, 0.001), 4)
    value = np.full_like(p, 2000.0)
    actions = costs.optimal_action(p, value)

    approve_p = p[actions == "APPROVE"]
    deny_p = p[actions == "DENY"]
    assert approve_p.size and deny_p.size

    distance_to_approve = approve_p.max()          # how close to 0 we insist on
    distance_to_deny = 1.0 - deny_p.min()          # how close to 1 we insist on
    assert distance_to_deny < distance_to_approve


# --- realised cost ---------------------------------------------------


def test_correct_decisions_cost_nothing_extra(costs: CostModel) -> None:
    actions = np.array(["APPROVE", "DENY"])
    labels = np.array([0, 1])  # both correct
    realised = costs.realised_cost(actions, labels, np.array([1000.0, 1000.0]))
    assert realised.tolist() == [0.0, 0.0]


def test_wrong_decisions_incur_their_costs(costs: CostModel) -> None:
    actions = np.array(["APPROVE", "DENY"])
    labels = np.array([1, 0])  # both wrong
    value = np.array([1000.0, 1000.0])
    realised = costs.realised_cost(actions, labels, value)
    assert realised[0] == pytest.approx(costs.cost_false_negative(1000.0))
    assert realised[1] == pytest.approx(costs.cost_false_positive(1000.0))


def test_review_always_costs_the_review_fee(costs: CostModel) -> None:
    """Review is never 'wrong' - it buys a correct answer for a flat fee."""
    for label in (0, 1):
        realised = costs.realised_cost(
            np.array(["REVIEW"]), np.array([label]), np.array([9999.0])
        )
        assert realised[0] == costs.review_cost


def test_cost_per_1000_scales_correctly(costs: CostModel) -> None:
    actions = np.array(["REVIEW"] * 100)
    labels = np.zeros(100, dtype=int)
    value = np.full(100, 1000.0)
    assert costs.cost_per_1000(actions, labels, value) == pytest.approx(
        costs.review_cost * 1000
    )


def test_cost_per_1000_of_empty_input_is_zero(costs: CostModel) -> None:
    assert costs.cost_per_1000(np.array([]), np.array([]), np.array([])) == 0.0


def test_recall_maximising_policy_costs_more_than_abstaining(costs: CostModel) -> None:
    """The concrete consequence of the asymmetry: a deny-everything-
    suspicious policy destroys more value than sending the same claims to
    review, on a realistic 15%-prevalence mix."""
    rng = np.random.default_rng(0)
    n = 2000
    labels = (rng.uniform(0, 1, n) < 0.15).astype(int)
    value = rng.uniform(500, 5000, n)

    deny_all = np.full(n, "DENY", dtype=object)
    review_all = np.full(n, "REVIEW", dtype=object)

    assert costs.cost_per_1000(deny_all, labels, value) > costs.cost_per_1000(
        review_all, labels, value
    )
