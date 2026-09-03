"""BGE-M3 dense embedding wrapper (FlagEmbedding).

Dense-only in v1: 1024-d normalized vectors, cosine similarity. The sparse/
colbert heads of bge-m3 are disabled — the lexical leg of hybrid search is
LanceDB's native BM25 FTS index (see store.py), which handles exact identifiers
better and costs nothing at index time.

Device policy (see docs/retrieval-plan.md "Platform scope"): CUDA on the
RTX 4080 box is the only performance target. device="auto" resolves cuda → cpu;
fp16 defaults on per device (True on cuda, False on cpu). The cpu path exists for
the fixture tests, not as a supported runtime.

VRAM: bge-m3 is ~570M params ≈ 1.2GB fp16; batch 32 × seq 1024 fits 16GB with
large headroom.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def resolve_device(device: str = "auto") -> str:
    """auto → 'cuda' if available else 'cpu'. Pass 'cuda'/'cpu' to force."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Embedder:
    """Lazy BGE-M3 dense encoder. The model loads on first encode call (~5-15s),
    so a bare `--mode fts` query never pays for it."""

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "auto",
        fp16: Optional[bool] = None,
        max_length: int = 1024,
        batch_size: int = 32,
    ):
        self.model_name = model
        self.device = resolve_device(device)
        self.fp16 = (self.device == "cuda") if fp16 is None else fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self.dim = 1024
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.fp16,
                devices=self.device,
            )
        return self._model

    def _encode(self, texts: List[str], batch_size: Optional[int]) -> np.ndarray:
        model = self._ensure_model()
        out = model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        # bge-m3 already L2-normalizes dense output; enforce it so cosine == dot.
        return vecs

    def encode_passages(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """(n, 1024) float32 for a list of chunk texts. Empty in → empty out."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts, batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        """(1024,) float32 for a single query string."""
        return self._encode([text], batch_size=1)[0]
