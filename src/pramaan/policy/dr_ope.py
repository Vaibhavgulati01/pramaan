"""Doubly-robust off-policy evaluation
(PRAMAAN_v2_architecture.md Sec.4 L4).

Answers: *what would a candidate policy have cost, had we deployed it,
given only logs from the incumbent?* Without this, evaluating a new
policy means shipping it and finding out — on real claimants.

## Why "doubly robust"

Two simpler estimators each fail in a different way:

- **Direct method**: fit a reward model `q̂(claim, action)` and average
  its predictions under the target policy. Cheap and low-variance, but
  inherits every bias in `q̂` — and `q̂` is fitted on logged data that is
  censored exactly where the incumbent denied.
- **IPS (inverse propensity scoring)**: reweight observed rewards by
  `π_target(a|x) / π_logging(a|x)`. Unbiased, but the variance explodes
  when propensities are small — one claim logged with propensity 0.01
  carries a weight of 100.

The doubly-robust estimator combines them:

    V_DR = mean[ Σ_a π_target(a|x)·q̂(x,a)  +  w·(r - q̂(x, a_logged)) ]

It is consistent if **either** `q̂` is correct **or** the propensities
are correct — hence "doubly robust". The direct term carries the estimate
and the IPS term corrects its bias, so `q̂` errors cancel rather than
propagate.

## The honest caveats

1. **Support.** An action the logging policy never took at a given claim
   (propensity 0) cannot be evaluated there. `support_diagnostics()`
   reports how much of the target policy's mass falls outside the logged
   support — the estimate is only as trustworthy as that number is small.
2. **Clipping trades bias for variance.** Weights are clipped, which
   biases the estimate toward the direct method. The clip rate is
   reported rather than tuned silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pramaan.policy.costs import CostModel

ACTIONS = ("APPROVE", "REVIEW", "DENY")
_ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}


@dataclass
class DREstimate:
    """A doubly-robust value estimate, in rupees per claim.

    Rewards here are **negative costs**, so a higher value is better and
    `value * -1000` is the familiar rupee-per-1,000-claims figure.
    """

    value: float
    direct_term: float
    correction_term: float
    n: int
    clipped_fraction: float
    effective_sample_size: float
    standard_error: float

    @property
    def cost_per_1000(self) -> float:
        return -self.value * 1000.0

    def as_dict(self) -> dict[str, float]:
        return {
            "dr_value": self.value,
            "dr_cost_per_1000": self.cost_per_1000,
            "direct_term": self.direct_term,
            "correction_term": self.correction_term,
            "n": float(self.n),
            "clipped_fraction": self.clipped_fraction,
            "effective_sample_size": self.effective_sample_size,
            "standard_error": self.standard_error,
        }


def actions_to_index(actions: np.ndarray) -> np.ndarray:
    return np.array([_ACTION_INDEX[a] for a in np.asarray(actions, dtype=object)])


def policy_action_matrix(actions: np.ndarray) -> np.ndarray:
    """One-hot (n, 3) matrix for a deterministic policy."""
    index = actions_to_index(actions)
    matrix = np.zeros((len(index), len(ACTIONS)), dtype=float)
    matrix[np.arange(len(index)), index] = 1.0
    return matrix


def build_reward_model(
    labels: np.ndarray,
    order_value: np.ndarray,
    costs: CostModel,
) -> np.ndarray:
    """`q̂(x, a)`: expected reward (negative cost) for every action.

    Built from the cost model and the realised label rather than fitted,
    because on this benchmark the labels are known. On real logs this
    would be a fitted regressor, and the DR estimator's whole point is
    that it stays consistent when that fit is imperfect.
    """
    labels = np.asarray(labels).astype(int)
    order_value = np.asarray(order_value, dtype=float)
    n = len(labels)

    q = np.zeros((n, len(ACTIONS)), dtype=float)
    # APPROVE: costs only if the claim was fraudulent.
    q[:, _ACTION_INDEX["APPROVE"]] = -np.where(
        labels == 1, costs.cost_false_negative(order_value), 0.0
    )
    # DENY: costs only if the claim was legitimate.
    q[:, _ACTION_INDEX["DENY"]] = -np.where(
        labels == 0, costs.cost_false_positive(order_value), 0.0
    )
    q[:, _ACTION_INDEX["REVIEW"]] = -costs.cost_review(order_value)
    return q


def dr_estimate(
    rewards: np.ndarray,
    logged_actions: np.ndarray,
    propensities: np.ndarray,
    q_hat: np.ndarray,
    target_policy_probs: np.ndarray,
    clip: float = 10.0,
) -> DREstimate:
    """Doubly-robust value of `target_policy_probs` from logged data.

    `rewards` are negative costs; `propensities` are P(logged action)
    under the logging policy; `q_hat` is (n, 3); `target_policy_probs` is
    (n, 3) and each row must sum to 1.
    """
    rewards = np.asarray(rewards, dtype=float)
    propensities = np.asarray(propensities, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)
    target = np.asarray(target_policy_probs, dtype=float)
    action_index = actions_to_index(logged_actions)
    n = len(rewards)

    if not (len(propensities) == len(action_index) == len(q_hat) == len(target) == n):
        raise ValueError("all inputs must have the same leading dimension")
    if np.any(propensities <= 0):
        raise ValueError(
            "propensity of 0 makes a logged action uninformative about "
            "counterfactuals; the logging policy must be stochastic where "
            "off-policy evaluation is needed (see exploration.py)"
        )
    if not np.allclose(target.sum(axis=1), 1.0):
        raise ValueError("target_policy_probs rows must sum to 1")

    rows = np.arange(n)
    direct = (target * q_hat).sum(axis=1)

    target_prob_of_logged = target[rows, action_index]
    weights = target_prob_of_logged / propensities
    clipped = np.minimum(weights, clip)
    clipped_fraction = float((weights > clip).mean())

    residual = rewards - q_hat[rows, action_index]
    correction = clipped * residual

    per_claim = direct + correction
    value = float(per_claim.mean())

    # Kish effective sample size: how many independent observations the
    # reweighting is really worth. A large gap from n means the estimate
    # rests on a handful of heavily-weighted claims.
    ess = float(clipped.sum() ** 2 / np.sum(clipped**2)) if np.any(clipped) else 0.0

    return DREstimate(
        value=value,
        direct_term=float(direct.mean()),
        correction_term=float(correction.mean()),
        n=n,
        clipped_fraction=clipped_fraction,
        effective_sample_size=ess,
        standard_error=float(per_claim.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
    )


def support_diagnostics(
    logged_actions: np.ndarray,
    propensities: np.ndarray,
    target_policy_probs: np.ndarray,
) -> dict[str, float]:
    """How much of the target policy sits outside the logged support.

    An action the logging policy never took at a claim cannot be
    evaluated there, and the DR estimate silently extrapolates via `q̂`.
    Reporting this is what stops the estimate being read as more
    trustworthy than it is.
    """
    target = np.asarray(target_policy_probs, dtype=float)
    propensities = np.asarray(propensities, dtype=float)
    action_index = actions_to_index(logged_actions)
    rows = np.arange(len(action_index))

    agreement = target[rows, action_index]
    weights = agreement / np.clip(propensities, 1e-12, None)

    return {
        "mean_target_prob_on_logged_action": float(agreement.mean()),
        "fraction_target_mass_off_support": float((agreement == 0).mean()),
        "max_importance_weight": float(weights.max()) if weights.size else 0.0,
        "mean_importance_weight": float(weights.mean()) if weights.size else 0.0,
    }
