"""The three-tier selective policy: APPROVE / REVIEW / DENY
(PRAMAAN_v2_architecture.md Sec.4 L4).

Two thresholds, chosen for different reasons and that distinction is the
whole design:

- **`t_deny`** must come from the certified set produced by
  Learn-then-Test. Among the certified options, the cost-minimising one
  is picked. The certificate constrains; cost decides within it. A
  threshold that minimises cost but is not certified is not eligible,
  however attractive its expected loss.
- **`t_approve`** carries no certificate and is chosen purely by cost. It
  bounds the *other* error - approving fraud - which the guarantee says
  nothing about. Being honest that only one side is certified matters:
  the repository claims a bound on false denials, not on everything.

Between them sits REVIEW: the abstention band. Its width is **derived**
from the cost model rather than chosen, because a claim belongs there
exactly when paying a human (`review_cost`) is cheaper in expectation
than acting on the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pramaan.policy.costs import CostModel
from pramaan.risk.certified_set import CertifiedSet

APPROVE = "APPROVE"
REVIEW = "REVIEW"
DENY = "DENY"


@dataclass
class SelectivePolicy:
    """A fitted three-tier decision rule."""

    t_approve: float
    t_deny: float
    certified: bool
    alpha: float | None = None
    delta: float | None = None
    selection_note: str = ""

    def decide(self, p_fraud: np.ndarray) -> np.ndarray:
        p = np.asarray(p_fraud, dtype=float)
        actions = np.full(p.shape, REVIEW, dtype=object)
        actions[p <= self.t_approve] = APPROVE
        actions[p >= self.t_deny] = DENY
        return actions

    def band_width(self) -> float:
        return max(0.0, self.t_deny - self.t_approve)

    def describe(self) -> str:
        certificate = (
            f"certified alpha={self.alpha}, delta={self.delta}"
            if self.certified
            else "UNCERTIFIED (no threshold cleared the ladder)"
        )
        return (
            f"APPROVE if p<={self.t_approve:.3f} | REVIEW | DENY if p>={self.t_deny:.3f} "
            f"(band {self.band_width():.3f}; {certificate})"
        )


@dataclass
class PolicySelectionReport:
    """How the operating point was chosen, including what was rejected."""

    policy: SelectivePolicy
    candidates: list[dict[str, float]] = field(default_factory=list)
    cost_per_1000: float = 0.0
    review_rate: float = 0.0
    deny_rate: float = 0.0
    approve_rate: float = 0.0
    uncertified_best: dict[str, float] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "t_approve": self.policy.t_approve,
            "t_deny": self.policy.t_deny,
            "band_width": self.policy.band_width(),
            "certified": self.policy.certified,
            "alpha": self.policy.alpha,
            "delta": self.policy.delta,
            "selection_note": self.policy.selection_note,
            "cost_per_1000": self.cost_per_1000,
            "review_rate": self.review_rate,
            "deny_rate": self.deny_rate,
            "approve_rate": self.approve_rate,
            "candidates": self.candidates,
            "uncertified_best": self.uncertified_best,
        }


def _cost_of(
    t_approve: float,
    t_deny: float,
    p: np.ndarray,
    labels: np.ndarray,
    order_value: np.ndarray,
    costs: CostModel,
) -> float:
    actions = SelectivePolicy(t_approve, t_deny, certified=False).decide(p)
    return costs.cost_per_1000(actions, labels, order_value)


def select_policy(
    p_fraud: np.ndarray,
    labels: np.ndarray,
    order_value: np.ndarray,
    costs: CostModel,
    certified: CertifiedSet | None,
    approve_grid: np.ndarray | None = None,
) -> PolicySelectionReport:
    """Picks the cost-minimising operating point subject to the certificate.

    `certified` is the Learn-then-Test outcome. If it certified nothing,
    the policy refuses to auto-deny at all (`t_deny = 1.0`, so no claim
    can reach it) and everything above `t_approve` goes to REVIEW.

    That refusal is deliberate. Auto-denying without a certificate is
    exactly the unbounded liability Sec.1 opens by describing, and
    falling back to "the best uncertified threshold" would quietly
    discard the entire point of the layer. The best uncertified candidate
    is still *reported* - as a counterfactual, so a reader can see the
    cost of insisting on the certificate.
    """
    p = np.asarray(p_fraud, dtype=float)
    labels = np.asarray(labels).astype(int)
    order_value = np.asarray(order_value, dtype=float)

    if approve_grid is None:
        # Starts at -1 ("approve nothing") so the search space CONTAINS
        # the trivial baselines. Without it the grid began at 0.01 and
        # already auto-approved ~10% of claims, so review-everything was
        # unreachable - and on the dev split that was genuinely cheaper
        # (Rs 40k/1000) than the "cost-minimising" point the search
        # returned (Rs 107k/1000). A cost-optimal search that cannot
        # express the baselines is not cost-optimal.
        approve_grid = np.concatenate(
            [[-1.0], np.round(np.arange(0.0, 0.51, 0.01), 4)]
        )

    deny_options = list(certified.certified_thresholds) if certified else []
    is_certified = bool(deny_options)

    candidates: list[dict[str, float]] = []
    if is_certified:
        search_space = [(a, d) for d in deny_options for a in approve_grid if a < d]
    else:
        # No certified threshold: auto-deny is off the table entirely.
        search_space = [(a, 1.01) for a in approve_grid]

    best: tuple[float, float, float] | None = None
    for t_approve, t_deny in search_space:
        cost = _cost_of(t_approve, t_deny, p, labels, order_value, costs)
        candidates.append({"t_approve": t_approve, "t_deny": t_deny, "cost_per_1000": cost})
        if best is None or cost < best[0]:
            best = (cost, t_approve, t_deny)

    assert best is not None
    cost, t_approve, t_deny = best

    note = (
        "cost-minimising member of the certified set"
        if is_certified
        else (
            "NO certified threshold: auto-deny disabled (t_deny=1.01, unreachable). "
            "Claims above t_approve go to human review rather than being denied "
            "without a guarantee."
        )
    )
    policy = SelectivePolicy(
        t_approve=t_approve,
        t_deny=t_deny,
        certified=is_certified,
        alpha=certified.alpha if certified else None,
        delta=certified.delta if certified else None,
        selection_note=note,
    )

    actions = policy.decide(p)
    report = PolicySelectionReport(
        policy=policy,
        candidates=sorted(candidates, key=lambda c: c["cost_per_1000"])[:20],
        cost_per_1000=cost,
        review_rate=float((actions == REVIEW).mean()),
        deny_rate=float((actions == DENY).mean()),
        approve_rate=float((actions == APPROVE).mean()),
    )

    # The counterfactual: what an uncertified cost-optimal policy would
    # have done. Reported so the price of insisting on a certificate is
    # visible rather than implied.
    if not is_certified:
        report.uncertified_best = _best_uncertified(
            p, labels, order_value, costs, approve_grid
        )

    return report


def _best_uncertified(
    p: np.ndarray,
    labels: np.ndarray,
    order_value: np.ndarray,
    costs: CostModel,
    approve_grid: np.ndarray,
) -> dict[str, float]:
    """Cost-optimal policy ignoring the certificate - a counterfactual for
    reporting only. Never returned as the deployed policy.

    The search space includes `t_deny = 1.01` (deny nothing). Without it
    this is not "the best uncertified policy" but "the best policy that
    is forced to deny", which is a different and misleading quantity: it
    can come out *more* expensive than the certified policy and make
    insisting on a certificate look free when it is not.

    Including it also makes the comparison sound - the uncertified search
    space is then a strict superset of the certified one, so the
    counterfactual is a genuine lower bound on achievable cost.
    """
    best: dict[str, float] | None = None
    deny_grid = [*np.round(np.arange(0.50, 1.00, 0.02), 4), 1.01]
    for t_deny in deny_grid:
        for t_approve in approve_grid:
            if t_approve >= t_deny:
                continue
            cost = _cost_of(t_approve, t_deny, p, labels, order_value, costs)
            if best is None or cost < best["cost_per_1000"]:
                best = {
                    "t_approve": float(t_approve),
                    "t_deny": float(t_deny),
                    "cost_per_1000": cost,
                }
    assert best is not None
    return best
