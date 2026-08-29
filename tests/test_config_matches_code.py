"""`configs/model.yaml` documents values that live in Python. Pin them.

Nothing loads that file — unlike `data.yaml`, `costs.yaml` and
`risk.yaml`, which all have real loaders. It is a record of the model's
operating parameters, and a record nobody checks is a record that lies.
It already did: before Phase 9 it advertised

  * `min_cell_size: 50` against a real `MIN_CELL_SIZE = 20`,
  * `monotone_constraints_features: []` against 11 applied constraints,
  * and an `hnsw_flat` reuse backend that was never implemented,

while `docs/VM_HANDOFF.md` instructed the reader to edit that last one to
switch backends on the VM. Following the runbook would have changed
nothing at all, silently.

These tests fail when the code moves and the file does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def documented() -> dict:
    return yaml.safe_load((ROOT / "configs" / "model.yaml").read_text(encoding="utf-8"))


def test_lightgbm_values_match_fusion_config(documented: dict) -> None:
    from pramaan.fusion.model import DEFAULT_N_FOLDS, FusionConfig

    cfg = FusionConfig()
    doc = documented["lightgbm"]

    assert doc["num_leaves"] == cfg.num_leaves
    assert doc["learning_rate"] == cfg.learning_rate
    assert doc["n_estimators"] == cfg.n_estimators
    assert doc["min_child_samples"] == cfg.min_child_samples
    assert doc["seed"] == cfg.seed
    assert doc["n_folds"] == DEFAULT_N_FOLDS


def test_determinism_block_matches_the_params_actually_passed(documented: dict) -> None:
    """All four settings are required together; documenting three is a trap."""
    from pramaan.fusion.model import FusionConfig

    params = FusionConfig().to_lgb_params()
    doc = documented["lightgbm"]

    assert doc["num_threads"] == params["num_threads"] == 1
    assert doc["deterministic"] is params["deterministic"] is True
    assert doc["force_row_wise"] is params["force_row_wise"] is True
    assert doc["seed"] == params["seed"]
    assert doc["objective"] == params["objective"]


def test_monotone_constraint_count_matches_the_schema(documented: dict) -> None:
    from pramaan.fusion.schema import monotone_constraint_vector

    vector = monotone_constraint_vector()
    applied = sum(1 for value in vector if value != 0)
    assert documented["lightgbm"]["monotone_constraints_applied"] == applied


def test_calibration_values_match_code(documented: dict) -> None:
    import inspect

    from pramaan.fusion.calibration import MIN_CELL_SIZE, evaluate_calibration

    doc = documented["calibration"]
    assert doc["min_cell_size"] == MIN_CELL_SIZE

    signature = inspect.signature(evaluate_calibration)
    assert doc["ece_bins"] == signature.parameters["n_bins"].default
    assert doc["min_group_size"] == signature.parameters["min_group_size"].default


def test_reuse_index_values_match_code(documented: dict) -> None:
    from pramaan.pillars.p3_reuse import (
        DEFAULT_BAND_BITS,
        DEFAULT_CLIP_THRESHOLD,
        DEFAULT_HAMMING_THRESHOLD,
        DEFAULT_LSH_BANDS,
        DEFAULT_PHASH_BITS,
    )

    doc = documented["reuse_index"]
    assert doc["phash_bits"] == DEFAULT_PHASH_BITS
    assert doc["lsh_bands"] == DEFAULT_LSH_BANDS
    assert doc["lsh_band_bits"] == DEFAULT_BAND_BITS
    assert doc["hamming_threshold"] == DEFAULT_HAMMING_THRESHOLD
    assert doc["clip_threshold"] == pytest.approx(DEFAULT_CLIP_THRESHOLD)


def test_documented_backend_actually_exists(documented: dict) -> None:
    """The specific failure that shipped: a config naming a backend the
    factory has never heard of, with a runbook telling you to select it."""
    from pramaan.pillars.p3_reuse import make_vector_store

    backend = documented["reuse_index"]["backend"]
    store = make_vector_store(8, backend)
    assert store is not None

    with pytest.raises(ValueError, match="unknown vector backend"):
        make_vector_store(8, "hnsw_flat")


def test_only_model_yaml_is_documentation_only() -> None:
    """The other three configs must keep their loaders.

    If one of them ever becomes dead too, it should be caught here rather
    than by someone editing it on a VM and wondering why nothing changed.
    """
    sources = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "src").rglob("*.py")
    )
    sources += "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "eval").rglob("*.py")
    )
    for name in ("data.yaml", "costs.yaml", "risk.yaml"):
        assert name in sources, f"configs/{name} is no longer loaded by any code"
