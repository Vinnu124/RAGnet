"""Hybrid (dense + BM25, RRF-fused) retrieval followed by cross-encoder reranking."""
from __future__ import annotations

import asyncio
from typing import Optional

from ragnet.config import Settings
from ragnet.retrieval.embedder import Embedder
from ragnet.retrieval.reranker import Reranker
from ragnet.retrieval.store import Hit, Store


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder, store: Store, reranker: Reranker):
        self.s = settings
        self.embedder = embedder
        self.store = store
        self.reranker = reranker

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        doc_ids: Optional[list[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[Hit]:
        top_k = top_k or self.s.top_k
        dense, sparse = await asyncio.to_thread(self.embedder.embed_query, query)
        cands = await asyncio.to_thread(
            self.store.hybrid_search, dense, sparse, limit=self.s.prefetch_k, doc_ids=doc_ids, page_range=page_range
        )
        return await asyncio.to_thread(self.reranker.rerank, query, cands, top_k)

    async def multi_search(self, queries: list[str], *, top_k: Optional[int] = None, doc_ids: Optional[list[str]] = None) -> list[Hit]:
        """Run several phrasings/sub-queries, pool candidates, rerank once against the first (canonical) query."""
        top_k = top_k or self.s.top_k
        embs = await asyncio.to_thread(self.embedder.embed_queries, queries)
        pooled: dict[str, Hit] = {}
        results = await asyncio.gather(
            *(asyncio.to_thread(self.store.hybrid_search, d, s, limit=self.s.prefetch_k, doc_ids=doc_ids) for d, s in embs)
        )
        for hits in results:
            for h in hits:
                pooled.setdefault(h.id, h)
        return await asyncio.to_thread(self.reranker.rerank, queries[0], list(pooled.values()), top_k)
