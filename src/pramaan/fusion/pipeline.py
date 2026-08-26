"""End-to-end feature extraction and training over a built corpus.

Bridges `benchmarks.loaders` (the corpus on disk) and `fusion.model`
(the trained fusion + calibrator), running the cascade over every claim
in timestamp order and assembling the feature matrix.

Feature extraction is the expensive part (~50ms/claim, dominated by CLIP
and forensics), so the assembled matrix is cached to disk keyed by the
corpus manifest hash. Retraining with different model hyperparameters
then costs seconds rather than minutes, and the cache invalidates
automatically when the corpus changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.loaders import Corpus, load_corpus
from pramaan.cascade.cascade import FEATURE_KEYS, Cascade, CascadeConfig
from pramaan.fusion.calibration import MondrianIsotonicCalibrator
from pramaan.fusion.model import FusionConfig, FusionModel
from pramaan.ingest.identity import ClaimIdentitySignals
from pramaan.pillars.p3_reuse import TemporalReuseIndex

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFeatures:
    features: pd.DataFrame
    labels: np.ndarray
    groups: np.ndarray
    splits: np.ndarray
    claim_ids: list[str]
    exit_stages: np.ndarray
    compute_ms: np.ndarray

    def for_split(self, name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        mask = self.splits == name
        return self.features[mask], self.labels[mask], self.groups[mask]


def _corpus_fingerprint(corpus: Corpus, use_clip: bool) -> str:
    """Cache key: the corpus's own content plus the extraction settings.

    Built from every claim's output SHA rather than the file mtime, so a
    rebuilt-but-identical corpus reuses the cache and a changed one never
    silently does.
    """
    digest = hashlib.sha256()
    for entry in sorted(corpus.manifest["entries"], key=lambda e: e["claim_id"]):
        digest.update(entry["output_sha256"].encode())
    digest.update(json.dumps(sorted(FEATURE_KEYS)).encode())
    digest.update(f"clip={use_clip}".encode())
    return digest.hexdigest()[:16]


def extract_features(
    tier: str,
    data_root: Path = Path("data"),
    cache_dir: Path | None = None,
    use_clip: bool = True,
    cascade_config: CascadeConfig | None = None,
) -> ExtractedFeatures:
    """Runs the cascade over every claim in timestamp order.

    Timestamp order is not a convenience: P3's index and P4's aggregates
    are both strictly backward-looking, and feeding claims out of order
    raises rather than silently leaking (see `TemporalLeakError`).
    """
    corpus = load_corpus(tier, data_root=data_root)
    cache_dir = cache_dir or (data_root / tier / "_features")
    fingerprint = _corpus_fingerprint(corpus, use_clip)
    cache_path = cache_dir / f"features_{fingerprint}.parquet"

    if cache_path.exists():
        logger.info("loading cached features from %s", cache_path)
        frame = pd.read_parquet(cache_path)
        return _unpack(frame)

    embedder = None
    clip_dim = None
    if use_clip:
        from pramaan.pillars.clip_embed import ClipEmbedder

        embedder = ClipEmbedder()
        clip_dim = embedder.dim

    cascade = Cascade(
        config=cascade_config or CascadeConfig(),
        reuse_index=TemporalReuseIndex(clip_dim=clip_dim),
        clip_embedder=embedder,
    )
    cascade.behaviour.register_identities(
        [
            ClaimIdentitySignals(
                claim_id=str(row.claim_id),
                phone_raw=str(row.phone),
                email_raw=str(row.email),
                address_raw=str(row.address),
                pin_raw=str(row.pin),
            )
            for row in corpus.claims.itertuples()
        ]
    )

    by_id = corpus.claims.set_index("claim_id")
    rows: list[dict[str, float]] = []
    meta: list[dict[str, object]] = []
    started = time.time()

    for index, claim in enumerate(corpus.iter_claims()):
        record = by_id.loc[claim.claim_id]
        result = cascade.process(
            claim_id=claim.claim_id,
            claimant_id=claim.claimant_id,
            merchant_id=claim.merchant_id,
            category=str(record["category"]),
            timestamp=claim.claim_timestamp,
            order_date=record["order_date"].to_pydatetime(),
            order_value=float(record["order_value_inr"]),
            raw_bytes=claim.read_image_bytes(),
            device_ua=str(record["device_ua"]),
            device_screen=str(record["device_screen"]),
            device_timezone=str(record["device_timezone"]),
            device_fonts=str(record["device_fonts"]).split(","),
        )
        rows.append(result.features)
        meta.append(
            {
                "claim_id": claim.claim_id,
                "_label": claim.label,
                "_split": claim.split,
                "_group": MondrianIsotonicCalibrator.group_key(
                    str(record["category"]), str(record["price_band"])
                ),
                "_exit_stage": int(result.exited_at_stage),
                "_compute_ms": result.compute_ms,
            }
        )
        if index % 500 == 0:
            logger.info("extracted %d/%d (%.0fs)", index, len(corpus), time.time() - started)

    frame = pd.concat([pd.DataFrame(rows), pd.DataFrame(meta)], axis=1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    logger.info("extracted %d claims in %.0fs -> %s", len(frame), time.time() - started, cache_path)
    return _unpack(frame)


def _unpack(frame: pd.DataFrame) -> ExtractedFeatures:
    return ExtractedFeatures(
        features=frame.loc[:, list(FEATURE_KEYS)],
        labels=frame["_label"].to_numpy(),
        groups=frame["_group"].to_numpy(),
        splits=frame["_split"].to_numpy(),
        claim_ids=frame["claim_id"].tolist(),
        exit_stages=frame["_exit_stage"].to_numpy(),
        compute_ms=frame["_compute_ms"].to_numpy(),
    )


def train_fusion(
    tier: str,
    data_root: Path = Path("data"),
    model_dir: Path | None = None,
    config: FusionConfig | None = None,
    use_clip: bool = True,
) -> tuple[FusionModel, ExtractedFeatures]:
    """Extracts features and fits the fusion model on the TRAIN split only.

    The calibration split is deliberately never passed to `fit` - it is
    reserved for Learn-then-Test in Phase 4 (docs/GUARANTEE.md), and
    `FusionModel.fit` raises if handed it.
    """
    extracted = extract_features(tier, data_root=data_root, use_clip=use_clip)

    train_features, train_labels, train_groups = extracted.for_split("train")
    if len(train_features) == 0:
        raise ValueError(f"no rows in the train split of tier {tier!r}")

    model = FusionModel(config)
    model.fit(
        train_features,
        train_labels,
        train_groups,
        splits=np.full(len(train_features), "train"),
    )

    if model_dir is not None:
        model.save(model_dir)
        logger.info("saved model to %s", model_dir)

    return model, extracted
