"""Cross-encoder reranker (bge-reranker-v2-m3) on GPU. Loads lazily."""
from __future__ import annotations

from ragnet.config import Settings
from ragnet.retrieval.embedder import pick_device
from ragnet.retrieval.store import Hit


class Reranker:
    def __init__(self, settings: Settings):
        self.s = settings
        self.device = pick_device(settings.device)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.s.rerank_model, device=self.device, max_length=self.s.embed_max_length)
        if self.device == "cuda":
            try:
                self._model.model.half()
            except Exception:
                pass

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        self._load()
        pairs = [(query, (h.context + "\n" + h.text) if h.context else h.text) for h in hits]
        scores = self._model.predict(pairs, batch_size=32, show_progress_bar=False)
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        out = []
        for h, s in ranked:
            h.score = float(s)
            out.append(h)
        return out
