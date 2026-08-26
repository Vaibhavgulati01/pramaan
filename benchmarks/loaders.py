"""Reads a built PRAMAAN-Bench-v1 corpus back off disk.

`build_bench.py` writes; this reads. Everything downstream (the pillars,
the cascade, evaluation) goes through here rather than touching
`data/{tier}/` directly, so the on-disk layout stays a private detail of
these two modules.

Claims come back sorted by `claim_timestamp`, because the single most
expensive mistake available in this codebase is letting a claim see
evidence submitted after it (P3's reuse index - see the temporal-leak
test in Phase 2). Iterating in timestamp order by default makes the
correct thing the easy thing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATA_ROOT = Path("data")


class CorpusNotBuiltError(FileNotFoundError):
    """Raised when a tier's corpus is missing, with the command to build it."""


@dataclass(frozen=True)
class Claim:
    claim_id: str
    split: str
    label: int
    fraud_class: str
    generator_family: str | None
    image_group_id: str
    merchant_id: str
    claimant_id: str
    claim_timestamp: pd.Timestamp
    image_path: Path

    def read_image_bytes(self) -> bytes:
        """The claim's submitted bytes, exactly as written - never decoded
        and re-encoded on the way out (see `pramaan.ingest.image`)."""
        return self.image_path.read_bytes()


@dataclass(frozen=True)
class Corpus:
    tier: str
    root: Path
    claims: pd.DataFrame
    manifest: dict[str, Any]

    def __len__(self) -> int:
        return len(self.claims)

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    def split(self, name: str) -> pd.DataFrame:
        """Rows for one split, still in timestamp order."""
        if name not in set(self.claims["split"]):
            available = sorted(set(self.claims["split"]))
            raise KeyError(f"no split {name!r} in this corpus; have {available}")
        return self.claims[self.claims["split"] == name]

    def iter_claims(self, split: str | None = None) -> Iterator[Claim]:
        """Claims in ascending `claim_timestamp` order - the order any
        temporally-aware pillar must consume them in."""
        frame = self.split(split) if split else self.claims
        for _, row in frame.iterrows():
            yield Claim(
                claim_id=str(row["claim_id"]),
                split=str(row["split"]),
                label=int(row["label"]),
                fraud_class=str(row["fraud_class"]),
                generator_family=(
                    str(row["generator_family"]) if pd.notna(row["generator_family"]) else None
                ),
                image_group_id=str(row["image_group_id"]),
                merchant_id=str(row["merchant_id"]),
                claimant_id=str(row["claimant_id"]),
                claim_timestamp=row["claim_timestamp"],
                image_path=self.images_dir / f"{row['claim_id']}.jpg",
            )


def load_corpus(tier: str, data_root: Path = DEFAULT_DATA_ROOT) -> Corpus:
    """Loads a built corpus. Raises CorpusNotBuiltError (naming the build
    command) rather than a bare FileNotFoundError, since "you haven't run
    the builder yet" is by far the most likely cause."""
    root = data_root / tier
    claims_path = root / "claims.csv"
    manifest_path = root / "manifest.json"

    if not claims_path.exists() or not manifest_path.exists():
        build_cmd = "pramaan data-full" if tier == "full" else f"pramaan data --scale {tier}"
        raise CorpusNotBuiltError(
            f"no {tier} corpus at {root}. Build it with: {build_cmd}"
        )

    claims = pd.read_csv(claims_path, dtype={"claim_id": str, "pin": str})
    claims["claim_timestamp"] = pd.to_datetime(claims["claim_timestamp"])
    claims["order_date"] = pd.to_datetime(claims["order_date"])
    claims = claims.sort_values("claim_timestamp", kind="stable").reset_index(drop=True)

    manifest = json.loads(manifest_path.read_text())
    return Corpus(tier=tier, root=root, claims=claims, manifest=manifest)
