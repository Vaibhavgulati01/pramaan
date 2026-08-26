"""Cost-ordered evidence cascade with early exit
(PRAMAAN_v2_architecture.md Sec.4 L1).

Pillars are ordered by cost, cheapest first, and after each stage a
cheap interim fusion decides whether the claim is already resolved:

    Stage 1  (~2 ms)   P1 provenance + P4 behaviour (cached aggregates)
    Stage 2  (~40 ms)  P2 container forensics
    Stage 3  (~120 ms) P3 reuse graph (pHash -> LSH -> CLIP)

If the interim probability falls outside `[tau_exit_lo, tau_exit_hi]`
after any stage, the remaining stages are skipped. Most claims are
obviously fine and never need the expensive pillars, which is what makes
a per-claim compute budget realistic at scale.

**The certified guarantee must be computed on the cascade as deployed.**
Sec.4 L1 is explicit about this and it is easy to get wrong: if you
calibrate on full-feature scores but deploy with early exit, the
distribution you certified is not the distribution you serve, and the
guarantee is void. So `CascadeResult` records exactly which stage a
claim exited at and which features were actually observed, and the
feature vector carries an explicit "not computed" marker rather than a
zero for skipped pillars - a zero is a *value*, and a model cannot tell
it apart from a genuine measurement.

**Determinism.** Stage order is fixed, exit thresholds are config, and
no stage consults a clock or RNG. The same claim against the same index
state yields the same result, which Phase 6's byte-identical metrics
test depends on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

import numpy as np

from pramaan.pillars.p1_provenance import ProvenanceFeatures, extract_provenance
from pramaan.pillars.p2_forensics import ForensicsFeatures, extract_forensics
from pramaan.pillars.p3_reuse import ReuseFeatures, TemporalReuseIndex, compute_phash
from pramaan.pillars.p4_behaviour import BehaviourAggregator, BehaviourFeatures
from pramaan.pillars.rings import RingDetector, RingFeatures

# Sentinel for "this pillar did not run". Distinct from every real
# feature value, so the fusion model can learn "unobserved" as its own
# state instead of confusing it with a measured zero. LightGBM handles
# NaN natively as a missing value, which is exactly the semantics wanted.
NOT_COMPUTED = float("nan")


class Stage(IntEnum):
    STAGE_1_CHEAP = 1  # provenance + behaviour
    STAGE_2_FORENSICS = 2
    STAGE_3_REUSE = 3


@dataclass
class CascadeResult:
    claim_id: str
    features: dict[str, float]
    exited_at_stage: Stage
    stages_run: list[Stage]
    compute_ms: float
    interim_probability: float | None = None

    # Raw per-pillar outputs, for the audit record (Sec.4 L5) and for
    # ablations that need to switch a pillar off after the fact.
    provenance: ProvenanceFeatures | None = None
    forensics: ForensicsFeatures | None = None
    reuse: ReuseFeatures | None = None
    ring: RingFeatures | None = None
    behaviour: BehaviourFeatures | None = None

    @property
    def ran_all_stages(self) -> bool:
        return self.exited_at_stage == Stage.STAGE_3_REUSE


@dataclass
class CascadeConfig:
    """Exit band and stage toggles.

    `tau_exit_lo`/`tau_exit_hi` bound the interim probability inside which
    a claim is still ambiguous enough to be worth more compute. They are
    deliberately wide by default: exiting early is an optimisation, and
    exiting *wrongly* costs a misclassification, so the default errs
    toward running more evidence. Phase 5 tunes them against the rupee
    cost model rather than by feel.
    """

    tau_exit_lo: float = 0.02
    tau_exit_hi: float = 0.98
    enable_early_exit: bool = True

    # Ablation switches (Phase 6 leave-one-pillar-out).
    enable_provenance: bool = True
    enable_forensics: bool = True
    enable_reuse: bool = True
    enable_behaviour: bool = True

    # Feature keys every claim carries, so the vector has a fixed schema
    # regardless of which stages ran. Populated lazily on first use.
    _all_keys: tuple[str, ...] | None = field(default=None, repr=False, compare=False)


def _all_feature_keys() -> tuple[str, ...]:
    """The full feature schema. Built from each pillar's empty features so
    it cannot drift from what the pillars actually emit."""
    keys: list[str] = []
    keys.extend(ProvenanceFeatures().as_dict())
    keys.extend(BehaviourFeatures().as_dict())
    keys.extend(ForensicsFeatures().as_dict())
    keys.extend(ReuseFeatures().as_dict())
    keys.extend(RingFeatures().as_dict())
    return tuple(keys)


FEATURE_KEYS = _all_feature_keys()


# Interim fusion. A trained model is not available until Phase 3, and the
# cascade must work before then, so this is a small hand-specified score
# over the signals whose direction is known a priori - the same
# directions the monotone constraints will encode. It is replaced by the
# calibrated model in Phase 3; `interim_scorer` is injectable precisely
# so that swap needs no change here.
#
# MEASURED, and worth being blunt about: with this placeholder scorer,
# **0% of the 3,000-claim dev corpus exits early**. It anchors at 0.5 and
# moves in bounded increments, so an ordinary claim lands mid-band and
# never clears [tau_exit_lo, tau_exit_hi]. The cascade machinery is
# correct - unit tests force exits at both stages and verify indexing
# still happens - but the *optimisation* is inert until a calibrated
# model supplies confident probabilities.
#
# We have deliberately not narrowed the exit band to manufacture exits.
# That would optimise the metric rather than the system: early exit is
# only safe when the interim score is genuinely well-calibrated, which is
# exactly what Phase 3 provides and Phase 5 then tunes against the rupee
# cost model. Until then the honest number to report for "fraction
# resolved at each stage" is 0/0/100.
def default_interim_scorer(features: dict[str, float]) -> float:
    """Crude interim risk in [0, 1]. Only used to decide early exit."""

    def get(key: str, default: float = 0.0) -> float:
        value = features.get(key, default)
        return default if value is None or np.isnan(value) else value

    score = 0.5

    # Reuse is the strongest signal we have, so it moves the score most.
    if get("reuse_matched_prior_claim") > 0:
        score += 0.25
        score += min(0.15, 0.05 * get("reuse_n_distinct_claimants_sharing"))
    if get("ring_in_ring") > 0:
        score += 0.15

    # Provenance only ever contributes positively-identified evidence.
    if get("provenance_declares_ai_generation") > 0:
        score += 0.40
    elif get("provenance_declares_camera_capture") > 0:
        score -= 0.10

    # Forensics: a desynced thumbnail is the cleanest container signal.
    score += min(0.20, 2.0 * get("forensics_thumbnail_mismatch"))

    # Behaviour: escalation relative to the claimant's own history.
    if get("behaviour_n_prior_claims") > 0:
        ratio = get("behaviour_order_value_vs_prior_mean", 1.0)
        if ratio > 3.0:
            score += 0.10
        if get("behaviour_n_prior_claims_30d") >= 3:
            score += 0.10

    return float(np.clip(score, 0.0, 1.0))


class Cascade:
    """Runs the pillars in cost order over a stream of claims.

    Stateful by necessity: P3's index and P4's aggregates both accumulate
    across claims, and both are strictly backward-looking. Claims must be
    fed in timestamp order (`benchmarks.loaders.Corpus.iter_claims`
    guarantees it); P3 raises `TemporalLeakError` if they are not.
    """

    def __init__(
        self,
        config: CascadeConfig | None = None,
        reuse_index: TemporalReuseIndex | None = None,
        ring_detector: RingDetector | None = None,
        behaviour: BehaviourAggregator | None = None,
        interim_scorer: Callable[[dict[str, float]], float] | None = None,
        clip_embedder: object | None = None,
    ) -> None:
        self.config = config or CascadeConfig()
        self.reuse_index = reuse_index or TemporalReuseIndex()
        self.ring_detector = ring_detector or RingDetector()
        self.behaviour = behaviour or BehaviourAggregator()
        self.interim_scorer = interim_scorer or default_interim_scorer
        self.clip_embedder = clip_embedder

    def _should_exit(self, probability: float) -> bool:
        if not self.config.enable_early_exit:
            return False
        return probability < self.config.tau_exit_lo or probability > self.config.tau_exit_hi

    def process(
        self,
        claim_id: str,
        claimant_id: str,
        merchant_id: str,
        category: str,
        timestamp: datetime,
        order_date: datetime,
        order_value: float,
        raw_bytes: bytes,
        device_ua: str | None = None,
        device_screen: str | None = None,
        device_timezone: str | None = None,
        device_fonts: list[str] | None = None,
    ) -> CascadeResult:
        started = time.perf_counter()
        # Every key present from the start, marked unobserved. A stage
        # that runs overwrites its own keys; one that is skipped leaves
        # NaN, which LightGBM reads as missing rather than as zero.
        features: dict[str, float] = dict.fromkeys(FEATURE_KEYS, NOT_COMPUTED)
        stages_run: list[Stage] = []
        result_parts: dict[str, object] = {}

        # --- Stage 1: provenance + behaviour (cheap) ---
        stages_run.append(Stage.STAGE_1_CHEAP)
        if self.config.enable_provenance:
            provenance = extract_provenance(raw_bytes)
            features.update(provenance.as_dict())
            result_parts["provenance"] = provenance
        if self.config.enable_behaviour:
            behaviour = self.behaviour.query_then_add(
                claim_id=claim_id,
                claimant_id=claimant_id,
                merchant_id=merchant_id,
                category=category,
                timestamp=timestamp,
                order_date=order_date,
                order_value=order_value,
                device_ua=device_ua,
                device_screen=device_screen,
                device_timezone=device_timezone,
                device_fonts=device_fonts,
            )
            features.update(behaviour.as_dict())
            result_parts["behaviour"] = behaviour

        probability = self.interim_scorer(features)
        if self._should_exit(probability):
            self._index_only(claim_id, claimant_id, merchant_id, timestamp, raw_bytes)
            return self._finish(
                claim_id, features, Stage.STAGE_1_CHEAP, stages_run, started,
                probability, result_parts,
            )

        # --- Stage 2: container forensics ---
        if self.config.enable_forensics:
            stages_run.append(Stage.STAGE_2_FORENSICS)
            forensics = extract_forensics(raw_bytes)
            features.update(forensics.as_dict())
            result_parts["forensics"] = forensics

            probability = self.interim_scorer(features)
            if self._should_exit(probability):
                self._index_only(claim_id, claimant_id, merchant_id, timestamp, raw_bytes)
                return self._finish(
                    claim_id, features, Stage.STAGE_2_FORENSICS, stages_run, started,
                    probability, result_parts,
                )

        # --- Stage 3: reuse graph (most expensive) ---
        if self.config.enable_reuse:
            stages_run.append(Stage.STAGE_3_REUSE)
            reuse, ring = self._run_reuse(
                claim_id, claimant_id, merchant_id, timestamp, raw_bytes, use_clip=True
            )
            features.update(reuse.as_dict())
            features.update(ring.as_dict())
            result_parts["reuse"] = reuse
            result_parts["ring"] = ring
            probability = self.interim_scorer(features)

        return self._finish(
            claim_id, features, Stage.STAGE_3_REUSE, stages_run, started,
            probability, result_parts,
        )

    def _run_reuse(
        self,
        claim_id: str,
        claimant_id: str,
        merchant_id: str,
        timestamp: datetime,
        raw_bytes: bytes,
        use_clip: bool,
    ) -> tuple[ReuseFeatures, RingFeatures]:
        phash = compute_phash(raw_bytes)
        embedding = None
        if use_clip and self.clip_embedder is not None and self.reuse_index.clip_enabled:
            embedding = self.clip_embedder.embed(raw_bytes)  # type: ignore[attr-defined]
        reuse = self.reuse_index.query_then_add(
            claim_id, claimant_id, merchant_id, timestamp, phash, clip_embedding=embedding
        )
        ring = self.ring_detector.observe(
            claim_id, claimant_id, merchant_id, timestamp, reuse
        )
        return reuse, ring

    def _index_only(
        self,
        claim_id: str,
        claimant_id: str,
        merchant_id: str,
        timestamp: datetime,
        raw_bytes: bytes,
    ) -> None:
        """Record an early-exiting claim in the reuse index, discarding its
        features.

        **Early exit skips using a claim's reuse features; it must never
        skip recording the claim.** An unindexed claim is invisible to
        every later claim, so a fraudster whose first submission exits
        early makes their own recycled follow-up undetectable - the exit
        optimisation would silently create the blind spot the pillar
        exists to close.

        Recording is cheap: pHash is ~12ms, against ~90ms for the CLIP
        embedding, so the embedding is deliberately skipped here. The cost
        is a slightly weaker index (exited claims are matchable by pHash
        but not semantically), which is a real and stated trade rather
        than a silent one.
        """
        if not self.config.enable_reuse:
            return
        self._run_reuse(
            claim_id, claimant_id, merchant_id, timestamp, raw_bytes, use_clip=False
        )

    def _finish(
        self,
        claim_id: str,
        features: dict[str, float],
        exited_at: Stage,
        stages_run: list[Stage],
        started: float,
        probability: float,
        parts: dict[str, object],
    ) -> CascadeResult:
        return CascadeResult(
            claim_id=claim_id,
            features=features,
            exited_at_stage=exited_at,
            stages_run=stages_run,
            compute_ms=(time.perf_counter() - started) * 1000.0,
            interim_probability=probability,
            provenance=parts.get("provenance"),  # type: ignore[arg-type]
            forensics=parts.get("forensics"),  # type: ignore[arg-type]
            reuse=parts.get("reuse"),  # type: ignore[arg-type]
            ring=parts.get("ring"),  # type: ignore[arg-type]
            behaviour=parts.get("behaviour"),  # type: ignore[arg-type]
        )
