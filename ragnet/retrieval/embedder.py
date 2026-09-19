"""Dense (bge-m3, GPU) + sparse (BM25 via fastembed, CPU) embeddings. Models load lazily on first use."""
from __future__ import annotations

from typing import Optional

from qdrant_client import models

from ragnet.config import Settings


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_sparse(e) -> models.SparseVector:
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


class Embedder:
    def __init__(self, settings: Settings):
        self.s = settings
        self.device = pick_device(settings.device)
        self._dense = None
        self._sparse = None

    def _load(self):
        if self._dense is not None:
            return
        from fastembed import SparseTextEmbedding
        from sentence_transformers import SentenceTransformer

        self._dense = SentenceTransformer(self.s.embed_model, device=self.device, trust_remote_code=True)
        self._dense.max_seq_length = self.s.embed_max_length
        if self.device == "cuda":
            self._dense.half()
        self._sparse = SparseTextEmbedding(self.s.sparse_model)

    @property
    def dim(self) -> int:
        self._load()
        return int(self._dense.get_sentence_embedding_dimension())

    def embed_docs(self, texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        self._load()
        dense = self._dense.encode(
            texts,
            batch_size=self.s.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sparse = [_to_sparse(e) for e in self._sparse.embed(texts, batch_size=256)]
        return dense.tolist(), sparse

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        self._load()
        dense = self._dense.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        sparse = _to_sparse(next(iter(self._sparse.query_embed(text))))
        return dense.tolist(), sparse

    def embed_queries(self, texts: list[str]) -> list[tuple[list[float], models.SparseVector]]:
        self._load()
        dense = self._dense.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sparse = [_to_sparse(e) for e in self._sparse.query_embed(texts)]
        return list(zip(dense.tolist(), sparse))
