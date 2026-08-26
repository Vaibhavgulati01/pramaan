"""Typed, versioned feature schema and the monotone constraints over it
(PRAMAAN_v2_architecture.md Sec.4 L1-L2).

Two jobs:

1. **Version the vector.** `FEATURE_SCHEMA_VERSION` lands in every audit
   record (Sec.4 L5). A model trained on one schema must never silently
   score a vector built by another, so the version is checked rather than
   assumed.

2. **Encode the directions we know a priori.** LightGBM monotone
   constraints force the fitted function to be non-decreasing (or
   non-increasing) in a feature regardless of what a thin data slice
   happens to suggest. Sec.4 L1 asks for this on the features whose
   direction is known, for three reasons: robustness under shift, coherent
   reason codes, and preventing a nonsense inversion learned from noise.

**A monotone constraint is a claim about the world, so each one below
carries its justification.** A wrong constraint is worse than none - it
forces the model to be confidently wrong in a direction it cannot escape.
Features whose direction is genuinely ambiguous are left unconstrained,
which is most of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from pramaan.cascade.cascade import FEATURE_KEYS

# Bump on ANY change to the feature set, ordering, or semantics. Recorded
# in every audit record and asserted at load time in fusion/model.py.
#
# 1.1.0 - dropped `reuse_n_candidates_examined`. It counts prior claims in
#         the index, so it grows with corpus position and acted as a proxy
#         for split membership (691 -> 1597 -> 2041 across
#         train/calibration/test). The model was drawing 10.2% of its gain
#         from what is effectively a timeline, not a claim property.
# 1.0.0 - initial 49-feature schema.
FEATURE_SCHEMA_VERSION = "1.1.0"


class Monotone(IntEnum):
    """LightGBM's encoding: +1 non-decreasing, -1 non-increasing, 0 free."""

    RISK_INCREASES = 1
    UNCONSTRAINED = 0
    RISK_DECREASES = -1


@dataclass(frozen=True)
class ConstrainedFeature:
    name: str
    direction: Monotone
    rationale: str


# Every constraint here must survive the question: "would I still believe
# this if the data disagreed?" If the answer is no, it belongs
# unconstrained instead.
MONOTONE_CONSTRAINTS: tuple[ConstrainedFeature, ...] = (
    # --- reuse: the moat, and where direction is clearest ---
    ConstrainedFeature(
        "reuse_n_distinct_claimants_sharing",
        Monotone.RISK_INCREASES,
        "More unrelated claimants submitting the same image is the "
        "definition of a reuse ring. Sec.4 L1's own worked example.",
    ),
    ConstrainedFeature(
        "reuse_n_matches",
        Monotone.RISK_INCREASES,
        "An image seen more times before is more likely recycled.",
    ),
    ConstrainedFeature(
        "reuse_max_clip_similarity",
        Monotone.RISK_INCREASES,
        "Higher semantic similarity to a prior claim's image is stronger "
        "reuse evidence; it cannot be exculpatory.",
    ),
    ConstrainedFeature(
        "reuse_best_hamming",
        Monotone.RISK_DECREASES,
        "Perceptual DISTANCE to the nearest prior image: larger means less "
        "like anything seen before. The inverse of the similarity above, "
        "and constrained oppositely so the two cannot disagree.",
    ),
    # --- rings ---
    ConstrainedFeature(
        "ring_distinct_claimants",
        Monotone.RISK_INCREASES,
        "A larger set of distinct identities around one image is stronger "
        "evidence of collusion.",
    ),
    ConstrainedFeature(
        "ring_is_first_seen",
        Monotone.RISK_DECREASES,
        "First-seen immunity, enforced structurally rather than hoped for: "
        "being the earliest claimant on a cluster must never RAISE risk, "
        "or submitting a rival's genuine photo first becomes an attack. "
        "The constraint makes that rule inviolable by the model.",
    ),
    # --- provenance: rare but decisive when present ---
    ConstrainedFeature(
        "provenance_declares_ai_generation",
        Monotone.RISK_INCREASES,
        "A signed C2PA manifest declaring synthesis is about as direct as "
        "evidence gets.",
    ),
    ConstrainedFeature(
        "provenance_declares_camera_capture",
        Monotone.RISK_DECREASES,
        "A signed hardware capture is authenticity evidence. Note this is "
        "only ever POSITIVE evidence - its absence is UNKNOWN and encoded "
        "as 0, never as risk (see p1_provenance).",
    ),
    # --- forensics ---
    ConstrainedFeature(
        "forensics_thumbnail_mismatch",
        Monotone.RISK_INCREASES,
        "A wider gap between the embedded EXIF thumbnail and the main "
        "image means more post-capture editing. Zero when no thumbnail "
        "exists, which is the common WhatsApp case and must stay neutral.",
    ),
    # --- behaviour (simulated data; see docs/DATA_CARD.md) ---
    ConstrainedFeature(
        "behaviour_order_value_vs_prior_mean",
        Monotone.RISK_INCREASES,
        "Claiming far above one's own historical order value is classic "
        "escalation. Defaults to 1.0 with no history, so a first-time "
        "claimant is not penalised for lacking one.",
    ),
    ConstrainedFeature(
        "behaviour_n_prior_claims_30d",
        Monotone.RISK_INCREASES,
        "A burst of claims in a short window is higher risk than the same "
        "count spread over a year.",
    ),
)

# Deliberately NOT constrained, with reasons - recorded so the omissions
# read as decisions rather than oversights:
#
#   behaviour_n_prior_claims      a loyal customer files more claims than a
#                                 one-off fraudster; direction genuinely
#                                 depends on volume context.
#   behaviour_claim_to_order_days both extremes are suspicious (instant
#                                 claims and last-minute ones), so the
#                                 relationship is non-monotone by nature.
#   forensics_ela_mean            higher means MORE detail than Q90
#                                 preserves, which is not the same as more
#                                 tampering - see p2_forensics.
#   forensics_estimated_quality   a pristine image and a heavily degraded
#                                 one are suspicious for opposite reasons.
#   ring_span_days                a long-lived cluster may be an organised
#                                 ring or a popular stock photo.
#
# And one feature was REMOVED rather than merely left unconstrained:
# `reuse_n_candidates_examined`, which measured index size and so
# tracked corpus position closely enough to identify a claim's split.
# See the schema version note above.

_CONSTRAINT_BY_NAME = {c.name: c for c in MONOTONE_CONSTRAINTS}


def feature_names() -> tuple[str, ...]:
    """Canonical feature order. LightGBM's monotone_constraints is a
    positional list, so this ordering is load-bearing and must match the
    vector the cascade produces."""
    return FEATURE_KEYS


def monotone_constraint_vector() -> list[int]:
    """Per-feature constraint in `feature_names()` order, for LightGBM."""
    return [int(_CONSTRAINT_BY_NAME.get(name, _UNCONSTRAINED).direction) for name in FEATURE_KEYS]


_UNCONSTRAINED = ConstrainedFeature("", Monotone.UNCONSTRAINED, "")


def validate_schema() -> None:
    """Fails loudly if a constraint names a feature that no longer exists.

    Renaming a feature without updating its constraint would otherwise
    silently drop that constraint - the model would still fit, still
    score, and quietly lose a guarantee we claim in the docs.
    """
    unknown = sorted({c.name for c in MONOTONE_CONSTRAINTS} - set(FEATURE_KEYS))
    if unknown:
        raise ValueError(
            f"monotone constraints reference unknown features: {unknown}. "
            "A renamed feature silently loses its constraint, so this is fatal."
        )
