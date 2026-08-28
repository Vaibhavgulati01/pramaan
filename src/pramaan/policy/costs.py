"""The rupee cost model, and the asymmetry that derives the abstention
band (PRAMAAN_v2_architecture.md Sec.4 L4).

## The load-bearing claim

**At typical Indian order values, a false positive costs MORE than a
false negative once churn is priced in.**

    C_FN = order_value + 180                       refund + COGS +
                                                   reverse freight + dead stock
    C_FP = order_value + 250 + p_churn * ltv       refund + handling +
                                                   the customer you just lost

With `p_churn = 0.35` and `ltv = 3000`, the churn term alone is 1,050
rupees - roughly six times the entire fixed cost of a false negative.
Wrongly denying an honest claimant is the expensive mistake.

This inverts the instinct most fraud systems are built on. A system tuned
to maximise recall destroys more value than the fraud it stops, and the
REVIEW band exists precisely because of it: when the expected cost of
deciding is worse than the cost of asking a human, you ask a human. The
band's width is *derived* from this arithmetic, not chosen.

Note both costs scale with `order_value`, so it does not cancel - it
appears in both and the asymmetry is carried entirely by the fixed terms
plus churn. The crossover is therefore independent of order value:
`C_FP > C_FN` whenever `250 + p_churn*ltv > 180`, which holds for any
plausible churn assumption. `crossover_order_value()` exists to make that
checkable rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class CostModel:
    """Per-decision costs in rupees. Loaded from configs/costs.yaml."""

    fn_fixed: float = 180.0
    fp_fixed: float = 250.0
    p_churn: float = 0.35
    ltv: float = 3000.0
    review_cost: float = 40.0
    epsilon: float = 0.01

    @classmethod
    def from_yaml(cls, path: Path) -> CostModel:
        config = yaml.safe_load(path.read_text())
        return cls(
            fn_fixed=float(config["false_negative"]["fixed_addon"]),
            fp_fixed=float(config["false_positive"]["fixed_addon"]),
            p_churn=float(config["false_positive"]["p_churn"]),
            ltv=float(config["false_positive"]["ltv"]),
            review_cost=float(config["review"]["fixed_cost"]),
            epsilon=float(config["exploration"]["epsilon"]),
        )

    # --- per-outcome costs -------------------------------------------

    def cost_false_negative(self, order_value: np.ndarray | float) -> np.ndarray:
        """Approving a fraudulent claim: refund, goods gone, freight."""
        return np.asarray(order_value, dtype=float) + self.fn_fixed

    def cost_false_positive(self, order_value: np.ndarray | float) -> np.ndarray:
        """Denying an honest claim: refund still owed, handling, and the
        churn-weighted lifetime value of a customer treated as a thief."""
        return (
            np.asarray(order_value, dtype=float)
            + self.fp_fixed
            + self.p_churn * self.ltv
        )

    def cost_review(self, order_value: np.ndarray | float) -> np.ndarray:
        """Human adjudication. Flat: an analyst's time does not scale with
        the order."""
        return np.full_like(np.asarray(order_value, dtype=float), self.review_cost)

    @property
    def churn_component(self) -> float:
        return self.p_churn * self.ltv

    def crossover_order_value(self) -> float | None:
        """Order value at which C_FP overtakes C_FN, or None if FP is
        always dearer.

        Both costs carry `order_value` identically, so it cancels and the
        comparison reduces to the fixed terms. Returning None is the
        expected answer under the spec's parameters, and saying so
        explicitly is better than returning a misleading 0.
        """
        if self.fp_fixed + self.churn_component > self.fn_fixed:
            return None  # false positives are dearer at every order value
        return float("inf")

    # --- expected cost of each action ---------------------------------

    def expected_costs(
        self, p_fraud: np.ndarray, order_value: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Expected rupee cost of each action, given calibrated P(fraud).

        APPROVE is wrong when the claim is fraudulent (probability p);
        DENY is wrong when it is honest (probability 1-p); REVIEW is a
        flat fee and is never "wrong" - it buys a correct answer.
        """
        p = np.asarray(p_fraud, dtype=float)
        value = np.asarray(order_value, dtype=float)
        return {
            "APPROVE": p * self.cost_false_negative(value),
            "DENY": (1.0 - p) * self.cost_false_positive(value),
            "REVIEW": self.cost_review(value),
        }

    def optimal_action(
        self, p_fraud: np.ndarray, order_value: np.ndarray
    ) -> np.ndarray:
        """The cost-minimising action per claim, ignoring the certificate.

        This is the *unconstrained* optimum. The deployed policy
        (policy/selective.py) additionally requires the deny threshold to
        come from the certified set, so it can only ever be more
        conservative than this.
        """
        costs = self.expected_costs(p_fraud, order_value)
        stacked = np.vstack([costs["APPROVE"], costs["DENY"], costs["REVIEW"]])
        names = np.array(["APPROVE", "DENY", "REVIEW"])
        return names[np.argmin(stacked, axis=0)]

    def realised_cost(
        self,
        actions: np.ndarray,
        labels: np.ndarray,
        order_value: np.ndarray,
    ) -> np.ndarray:
        """Actual cost incurred once the truth is known - what the
        rupee-per-1000-claims figure is built from."""
        actions = np.asarray(actions)
        labels = np.asarray(labels).astype(int)
        value = np.asarray(order_value, dtype=float)

        cost = np.zeros(len(actions), dtype=float)
        approved_fraud = (actions == "APPROVE") & (labels == 1)
        denied_legit = (actions == "DENY") & (labels == 0)
        reviewed = actions == "REVIEW"

        cost[approved_fraud] = self.cost_false_negative(value[approved_fraud])
        cost[denied_legit] = self.cost_false_positive(value[denied_legit])
        cost[reviewed] = self.review_cost
        # Correct APPROVE and correct DENY cost nothing beyond business as
        # usual, which is the baseline every option is measured against.
        return cost

    def cost_per_1000(
        self,
        actions: np.ndarray,
        labels: np.ndarray,
        order_value: np.ndarray,
    ) -> float:
        realised = self.realised_cost(actions, labels, order_value)
        return float(realised.sum() / len(realised) * 1000.0) if len(realised) else 0.0
