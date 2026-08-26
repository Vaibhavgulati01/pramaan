"""CLIP ViT-B/32 image embeddings for P3's semantic near-duplicate stage.

Kept separate from `p3_reuse.py` so the index is testable without loading
a ~350MB model: `TemporalReuseIndex` takes embeddings as plain arrays and
never imports this module.

Model choice is `ViT-B-32-quickgelu`, not `ViT-B-32`. The OpenAI weights
were trained with QuickGELU activation, and open_clip warns that pairing
them with the plain config silently produces subtly wrong embeddings -
the kind of defect that degrades results without ever raising.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Protocol

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# open_clip's preprocess is a torchvision Compose; typed here as the
# callable it actually is so mypy doesn't infer it from the model tuple.
Preprocess = Callable[[Image.Image], torch.Tensor]


class ClipModel(Protocol):
    """The slice of open_clip's model we use. `encode_image` is not part
    of torch.nn.Module, so typing the model as Module alone makes mypy
    resolve it through Module.__getattr__ and see a Tensor."""

    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...

    def to(self, device: str) -> ClipModel: ...

    def eval(self) -> ClipModel: ...

CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_EMBED_DIM = 512


@lru_cache(maxsize=1)
def _load_model() -> tuple[ClipModel, Preprocess]:
    """Loads once per process. First call may download ~350MB; afterwards
    it is served from the HF cache."""
    import open_clip

    logger.info("loading CLIP %s (%s)", CLIP_MODEL_NAME, CLIP_PRETRAINED)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model.eval()
    return model, preprocess


class ClipEmbedder:
    """Batched, deterministic CLIP image embeddings.

    Deterministic by construction: `torch.inference_mode` with a
    single-threaded, eval-mode model on CPU. Phase 6's byte-identical
    metrics test depends on repeated runs producing identical embeddings.
    """

    def __init__(self, device: str = "cpu", num_threads: int = 1) -> None:
        self.device = device
        torch.set_num_threads(num_threads)
        self._model, self._preprocess = _load_model()
        self._model = self._model.to(device)

    @property
    def dim(self) -> int:
        return CLIP_EMBED_DIM

    def embed_batch(self, images: list[bytes]) -> np.ndarray:
        """(n, 512) L2-normalised float32 embeddings, one row per input."""
        if not images:
            return np.zeros((0, CLIP_EMBED_DIM), dtype=np.float32)

        tensors = []
        for raw in images:
            with Image.open(io.BytesIO(raw)) as img:
                tensors.append(self._preprocess(img.convert("RGB")))
        batch = torch.stack(tensors).to(self.device)

        with torch.inference_mode():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)

    def embed(self, raw_bytes: bytes) -> np.ndarray:
        """(512,) L2-normalised embedding for one image."""
        return self.embed_batch([raw_bytes])[0]
