"""Qdrant wrapper: named dense + sparse vectors, server-side RRF hybrid fusion, page reads, exact-term grep."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Optional

from qdrant_client import QdrantClient, models

from ragnet.config import Settings
from ragnet.ingest.chunker import Chunk

DENSE = "dense"
SPARSE = "sparse"


@dataclass
class Hit:
    id: str
    score: float
    text: str
    context: str
    doc_id: str
    doc_name: str
    page: int
    page_end: int
    section: str
    index: int

    @property
    def page_label(self) -> str:
        return f"p.{self.page}" if self.page == self.page_end else f"p.{self.page}-{self.page_end}"

    @property
    def locator(self) -> str:
        loc = f"{self.doc_name} · {self.page_label}"
        if self.section:
            loc += f" · §{self.section}"
        return loc

    def to_dict(self) -> dict:
        d = asdict(self)
        d["locator"] = self.locator
        return d


def point_id(doc_id: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ragnet:{doc_id}:{index}"))


def _hit(p) -> Hit:
    pl = p.payload or {}
    return Hit(
        id=str(p.id),
        score=float(getattr(p, "score", 0.0) or 0.0),
        text=pl.get("text", ""),
        context=pl.get("context", ""),
        doc_id=pl.get("doc_id", ""),
        doc_name=pl.get("doc_name", ""),
        page=int(pl.get("page", 0)),
        page_end=int(pl.get("page_end", 0)),
        section=pl.get("section", ""),
        index=int(pl.get("index", 0)),
    )


class Store:
    def __init__(self, settings: Settings):
        self.s = settings
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
        self.collection = settings.collection

    # ------------------------------------------------------------------ schema
    def ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if not self.s.qdrant_url:
            return  # payload indexes are a no-op in embedded mode (filters still work, just unindexed)
        self.client.create_payload_index(self.collection, "doc_id", models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "page", models.PayloadSchemaType.INTEGER)
        self.client.create_payload_index(self.collection, "index", models.PayloadSchemaType.INTEGER)
        self.client.create_payload_index(
            self.collection,
            "text",
            models.TextIndexParams(type="text", tokenizer=models.TokenizerType.WORD, min_token_len=2, max_token_len=40, lowercase=True),
        )

    def drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    # ------------------------------------------------------------------ write
    def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse: list[models.SparseVector], batch: int = 128) -> None:
        points = [
            models.PointStruct(
                id=point_id(c.doc_id, c.index),
                vector={DENSE: d, SPARSE: s},
                payload={
                    "text": c.text,
                    "context": c.context,
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "page_end": c.page_end,
                    "section": c.section,
                    "index": c.index,
                    "tokens": c.tokens,
                },
            )
            for c, d, s in zip(chunks, dense, sparse)
        ]
        for i in range(0, len(points), batch):
            self.client.upsert(self.collection, points=points[i : i + batch], wait=True)

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._filter(doc_ids=[doc_id])),
            wait=True,
        )

    # ------------------------------------------------------------------ read
    @staticmethod
    def _filter(doc_ids: Optional[list[str]] = None, page: Optional[int] = None, page_range: Optional[tuple[int, int]] = None) -> Optional[models.Filter]:
        must: list = []
        if doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))))
        if page is not None:
            must.append(models.FieldCondition(key="page", range=models.Range(lte=page)))
            must.append(models.FieldCondition(key="page_end", range=models.Range(gte=page)))
        if page_range is not None:
            must.append(models.FieldCondition(key="page", range=models.Range(gte=page_range[0], lte=page_range[1])))
        return models.Filter(must=must) if must else None

    def hybrid_search(
        self,
        dense: list[float],
        sparse: models.SparseVector,
        *,
        limit: int,
        doc_ids: Optional[list[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[Hit]:
        flt = self._filter(doc_ids=doc_ids, page_range=page_range)
        res = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, limit=limit, filter=flt),
                models.Prefetch(query=sparse, using=SPARSE, limit=limit, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [_hit(p) for p in res.points]

    def _scroll_all(self, flt: Optional[models.Filter], limit: int = 10_000) -> list[Hit]:
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=min(256, limit - len(out)), offset=offset, with_payload=True)
            out.extend(_hit(p) for p in pts)
            if offset is None or len(out) >= limit:
                break
        return out

    def get_page(self, doc_id: str, page: int) -> list[Hit]:
        hits = self._scroll_all(self._filter(doc_ids=[doc_id], page=page))
        return sorted(hits, key=lambda h: h.index)

    def get_doc(self, doc_id: str) -> list[Hit]:
        return sorted(self._scroll_all(self._filter(doc_ids=[doc_id])), key=lambda h: h.index)

    def grep(self, term: str, *, limit: int, doc_ids: Optional[list[str]] = None) -> list[Hit]:
        flt = self._filter(doc_ids=doc_ids)
        must = list(flt.must) if flt else []
        must.append(models.FieldCondition(key="text", match=models.MatchText(text=term)))
        hits = self._scroll_all(models.Filter(must=must), limit=limit * 4)
        needle = term.lower()
        exact = [h for h in hits if needle in h.text.lower()]
        return (exact or hits)[:limit]
