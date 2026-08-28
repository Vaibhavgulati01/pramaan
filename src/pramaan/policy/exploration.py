"""ε-exploration: buying unbiased labels in the denial region
(PRAMAAN_v2_architecture.md Sec.4 L4).

## The selective-labeling problem

You only observe ground truth for claims you **approved**. A denied claim
never reveals whether it was actually fraudulent — the refund was
refused, the customer went away, and nothing further was learned.

So tomorrow's model trains on a censored, policy-biased sample: every
claim the current policy denies is invisible to it. The model then looks
excellent on exactly the claims it was already good at, and the evaluation
is optimistic in a way no amount of held-out data fixes, because the
held-out data is censored the same way.

Almost no production fraud system handles this. The fix is small and
slightly uncomfortable: **approve a random ε of the claims the policy
would have denied**, and log the propensity. Those claims come back with
real labels, and because they were selected at random rather than by the
model, they are an unbiased window into the denial region.

## The cost is real and is reported in rupees

ε-exploration deliberately approves claims believed fraudulent. That
costs money — `exploration_cost_inr()` computes exactly how much, and it
is reported separately rather than buried in the headline figure. Paying
it knowingly is the point; hiding it would not be.

## Propensity

Every logged decision records the probability that the *logging* policy
would take the action it took. Doubly-robust off-policy evaluation
(`dr_ope.py`) needs this to reweight, and a propensity of 0 makes a claim
uninformative about counterfactuals — which is why exploration must be
stochastic rather than a fixed rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pramaan.policy.costs import CostModel
from pramaan.policy.selective import APPROVE, DENY, REVIEW


@dataclass
class ExplorationLog:
    """Actions actually taken, with the propensity of each.

    `propensity[i]` is P(logging policy chose `action[i]` | claim i). For
    a deterministic policy that is 1.0; under ε-exploration a claim in
    the denial region has ε for APPROVE and 1-ε for DENY.
    """

    actions: np.ndarray
    propensities: np.ndarray
    explored: np.ndarray  # True where the policy's action was overridden

    @property
    def n_explored(self) -> int:
        return int(self.explored.sum())

    @property
    def exploration_rate(self) -> float:
        return float(self.explored.mean()) if self.explored.size else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "n_explored": float(self.n_explored),
            "exploration_rate": self.exploration_rate,
            "min_propensity": float(self.propensities.min()) if self.propensities.size else 0.0,
        }


def apply_epsilon_exploration(
    actions: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> ExplorationLog:
    """Approves a random ε of would-be denials, logging propensities.

    Only DENY is explored. Exploring APPROVE would mean denying claims the
    policy believed honest — paying the expensive error (C_FP) to learn
    something we already observe, since approved claims yield labels
    anyway. The asymmetry in the cost model makes this the only sensible
    direction.
    """
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")

    actions = np.asarray(actions, dtype=object)
    final = actions.copy()
    propensities = np.ones(len(actions), dtype=float)
    explored = np.zeros(len(actions), dtype=bool)

    deny_mask = actions == DENY
    if epsilon > 0.0 and deny_mask.any():
        draws = rng.uniform(0.0, 1.0, size=len(actions))
        flip = deny_mask & (draws < epsilon)

        final[flip] = APPROVE
        explored[flip] = True
        # Propensity of the action actually taken.
        propensities[flip] = epsilon
        propensities[deny_mask & ~flip] = 1.0 - epsilon

    return ExplorationLog(actions=final, propensities=propensities, explored=explored)


def exploration_cost_inr(
    log: ExplorationLog,
    labels: np.ndarray,
    order_value: np.ndarray,
    costs: CostModel,
) -> dict[str, float]:
    """What exploration cost, in rupees, and what it bought.

    The cost is only incurred on explored claims that really were
    fraudulent: approving a claim the policy wanted to deny is free if the
    claim was honest — indeed it avoided a false positive, which is
    reported as `savings_inr` rather than quietly netted off.
    """
    labels = np.asarray(labels).astype(int)
    order_value = np.asarray(order_value, dtype=float)

    explored_fraud = log.explored & (labels == 1)
    explored_legit = log.explored & (labels == 0)

    cost = float(costs.cost_false_negative(order_value[explored_fraud]).sum())
    savings = float(costs.cost_false_positive(order_value[explored_legit]).sum())
    n = len(labels)

    return {
        "n_explored": float(log.n_explored),
        "n_explored_fraud": float(explored_fraud.sum()),
        "n_explored_legit": float(explored_legit.sum()),
        "cost_inr": cost,
        # Exploration sometimes rescues an honest claimant the policy
        # would have wrongly denied. Reported, never netted off - the
        # gross cost is the number a merchant has to approve.
        "savings_inr": savings,
        "net_inr": cost - savings,
        "cost_per_1000_claims": cost / n * 1000.0 if n else 0.0,
        "labels_gained_in_denial_region": float(log.n_explored),
    }


def censoring_report(actions: np.ndarray) -> dict[str, float]:
    """How much of the outcome space the logging policy hides.

    A denied claim yields no label, so this is the fraction of the
    dataset that a naively-trained successor model would never see. It is
    the size of the blind spot ε-exploration exists to shrink.
    """
    actions = np.asarray(actions, dtype=object)
    n = len(actions)
    if n == 0:
        return {"censored_rate": 0.0, "observed_rate": 0.0, "n_censored": 0.0}

    censored = (actions == DENY).sum()
    return {
        "n_censored": float(censored),
        "censored_rate": float(censored / n),
        # APPROVE and REVIEW both yield a label: one from the outcome, one
        # from the human adjudication.
        "observed_rate": float(((actions == APPROVE) | (actions == REVIEW)).sum() / n),
    }
