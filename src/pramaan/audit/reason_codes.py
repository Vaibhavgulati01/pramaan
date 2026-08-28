"""SHAP attributions rendered through a deterministic template layer
(PRAMAAN_v2_architecture.md Sec.4 L5).

## Why templates, not generated text

An adverse-action decision must be explainable in terms that survive
scrutiny months later, from the same inputs, byte for byte. A stochastic
text generator cannot promise that: the same claim can produce two
different explanations, and neither can be audited against the model that
actually made the decision.

So reason codes are **rendered from a fixed template table**. SHAP
supplies the attribution; the template supplies the words. `Sec.10`'s
scope-creep list allows an LLM only for turning already-computed reason
codes into customer-facing prose, clearly separated from the decision -
and nothing in this module does that.

## What is exposed, and what is not

The API returns reason codes and their rank, never raw SHAP values.
Publishing per-feature attributions would hand an attacker a gradient
oracle: submit a claim, read which feature moved the score, adjust,
repeat. `SAFETY.md` commits to that boundary and this module is where it
is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Feature -> (code, template). One entry per feature we are willing to
# explain a decision with. A feature not in this table can influence the
# score but will never be quoted as a reason - deliberately, because a
# reason a claimant cannot act on is not a reason.
REASON_TEMPLATES: dict[str, tuple[str, str]] = {
    "reuse_n_distinct_claimants_sharing": (
        "REUSE_MULTI_CLAIMANT",
        "Submitted image matches {n_matches} prior claim(s) from "
        "{n_distinct_claimants_sharing} distinct claimant(s), first seen "
        "{days_since_first_seen:.0f} days ago.",
    ),
    "reuse_matched_prior_claim": (
        "REUSE_PRIOR_CLAIM",
        "Submitted image matches evidence from an earlier claim "
        "(closest perceptual distance {reuse_min_hamming:.0f}).",
    ),
    "reuse_max_clip_similarity": (
        "REUSE_SEMANTIC_MATCH",
        "Submitted image is semantically near-identical to a prior claim's "
        "image (similarity {reuse_max_clip_similarity:.2f}).",
    ),
    "ring_distinct_claimants": (
        "RING_MEMBERSHIP",
        "Image belongs to a cluster shared by {ring_distinct_claimants:.0f} "
        "claimants across {ring_distinct_merchants:.0f} merchant(s).",
    ),
    "provenance_declares_ai_generation": (
        "PROVENANCE_AI_DECLARED",
        "Image carries a signed C2PA manifest declaring AI generation.",
    ),
    "provenance_declares_camera_capture": (
        "PROVENANCE_CAMERA_CAPTURE",
        "Image carries a signed C2PA manifest declaring hardware capture.",
    ),
    "forensics_thumbnail_mismatch": (
        "METADATA_THUMBNAIL_DESYNC",
        "Embedded thumbnail does not match the image content "
        "(divergence {forensics_thumbnail_mismatch:.2f}), consistent with "
        "post-capture editing.",
    ),
    "forensics_estimated_quality": (
        "COMPRESSION_HISTORY",
        "Compression history is inconsistent with the stated delivery route.",
    ),
    "behaviour_order_value_vs_prior_mean": (
        "VALUE_ESCALATION",
        "Claim value is {behaviour_order_value_vs_prior_mean:.1f}x this "
        "claimant's historical average.",
    ),
    "behaviour_n_prior_claims_30d": (
        "CLAIM_FREQUENCY",
        "{behaviour_n_prior_claims_30d:.0f} prior claim(s) from this claimant "
        "in the last 30 days.",
    ),
}

# Attributions below this contribute nothing a human would act on.
MIN_ABS_SHAP = 0.01


@dataclass(frozen=True)
class ReasonCode:
    code: str
    text: str
    shap: float
    feature: str

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "text": self.text,
            # Rounded: the audit record documents a decision, and
            # publishing full-precision attributions edges toward the
            # gradient oracle SAFETY.md rules out.
            "shap": round(self.shap, 3),
        }


def render_reason_codes(
    shap_values: dict[str, float],
    feature_values: dict[str, float],
    top_k: int = 3,
    min_abs_shap: float = MIN_ABS_SHAP,
) -> list[ReasonCode]:
    """Top-k risk-increasing reasons, rendered deterministically.

    Only *positive* attributions become reason codes: this explains why a
    claim was flagged, not why it might have been cleared. Adverse-action
    reasoning is the regulated direction, and mixing exculpatory factors
    into the same list muddles what the claimant is being told.
    """
    candidates: list[ReasonCode] = []

    for feature, shap in sorted(shap_values.items(), key=lambda kv: -kv[1]):
        if shap < min_abs_shap:
            break  # sorted descending: nothing further can qualify
        entry = REASON_TEMPLATES.get(feature)
        if entry is None:
            continue  # influences the score, but not a reason we will quote
        code, template = entry

        safe_values = {
            key: (0.0 if value is None or np.isnan(value) else value)
            for key, value in feature_values.items()
        }
        try:
            text = template.format(**safe_values)
        except KeyError:
            # A template referencing a feature that no longer exists is a
            # bug, but a decision must still be explainable, so fall back
            # to the code's plain meaning rather than raising.
            text = code.replace("_", " ").capitalize()

        candidates.append(ReasonCode(code=code, text=text, shap=float(shap), feature=feature))
        if len(candidates) >= top_k:
            break

    return candidates


def compute_shap_values(
    model: object,
    features: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    """SHAP attributions for a single claim, as {feature: value}.

    Uses LightGBM's built-in `pred_contrib`, which is exact for tree
    ensembles rather than an approximation - so the explanation matches
    the decision exactly, which is the property an audit needs.
    """
    import numpy as _np

    row = _np.asarray(features, dtype=float).reshape(1, -1)
    contributions = model.predict(row, pred_contrib=True)  # type: ignore[attr-defined]
    # The final column is the expected-value base term, not a feature.
    values = _np.asarray(contributions)[0][:-1]
    return dict(zip(feature_names, values.astype(float), strict=True))
