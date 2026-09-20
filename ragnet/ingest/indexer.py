"""Ingestion pipeline: parse → chunk → (optional LLM context) → embed → upsert. Incremental via file hashes."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Optional

from ragnet.config import Settings
from ragnet.ingest.chunker import chunk_pages
from ragnet.ingest.contextualizer import Contextualizer
from ragnet.ingest.parsers import SUPPORTED, parse
from ragnet.llm.client import LLM
from ragnet.manifest import DocRecord, Manifest
from ragnet.retrieval.embedder import Embedder
from ragnet.retrieval.store import Store

Progress = Callable[[str], None]


def expand_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED))
        elif p.is_file():
            out.append(p)
    return out


class Indexer:
    def __init__(self, settings: Settings, embedder: Embedder, store: Store, manifest: Manifest, llm: Optional[LLM] = None):
        self.s = settings
        self.embedder = embedder
        self.store = store
        self.manifest = manifest
        self.llm = llm

    async def ingest(
        self,
        paths: list[Path],
        *,
        hifi: bool = False,
        contextual: Optional[bool] = None,
        force: bool = False,
        progress: Optional[Progress] = None,
    ) -> list[DocRecord]:
        log = progress or (lambda _msg: None)
        contextual = self.s.contextual if contextual is None else contextual
        use_llm = self.llm is not None and (contextual or self.s.summarize_docs)
        if use_llm and not await self.llm.ping():
            log(f"⚠ LLM at {self.s.llm_base_url} unreachable — skipping summaries/contextualisation")
            use_llm = False
        ctx = Contextualizer(self.s, self.llm) if use_llm else None

        self.store.ensure_collection(self.embedder.dim)
        records: list[DocRecord] = []

        for path in expand_paths(paths):
            log(f"→ {path.name}: parsing ({'docling' if hifi else 'pymupdf4llm'})")
            doc = await asyncio.to_thread(parse, path, hifi)
            existing = self.manifest.by_sha(doc.sha256)
            if existing and not force:
                log(f"  ✓ unchanged, already indexed as {existing.doc_id} ({existing.chunks} chunks) — skipped")
                records.append(existing)
                continue
            if existing:
                self.store.delete_doc(existing.doc_id)
                self.manifest.remove(existing.doc_id)

            chunks = chunk_pages(doc.pages, doc_id=doc.doc_id, doc_name=doc.name, max_tokens=self.s.chunk_tokens, overlap_tokens=self.s.chunk_overlap_tokens)
            log(f"  {len(doc.pages)} pages → {len(chunks)} chunks")
            if not chunks:
                log("  ⚠ no text extracted (scanned PDF? try --hifi for OCR)")
                continue

            summary: dict = {"summary": "", "outline": []}
            if ctx:
                if self.s.summarize_docs:
                    summary = await ctx.summarize(doc)
                if contextual:
                    log(f"  writing chunk contexts with {self.s.router_model_name} …")
                    await ctx.contextualize(doc, chunks)

            log("  embedding (dense + sparse) …")
            dense, sparse = await asyncio.to_thread(self.embedder.embed_docs, [c.embed_text for c in chunks])
            await asyncio.to_thread(self.store.upsert, chunks, dense, sparse)

            rec = DocRecord(
                doc_id=doc.doc_id,
                name=doc.name,
                path=str(path.resolve()),
                sha256=doc.sha256,
                pages=len(doc.pages),
                chunks=len(chunks),
                tokens=sum(c.tokens for c in chunks),
                summary=summary["summary"],
                outline=summary["outline"],
                contextual=bool(contextual and ctx),
            )
            self.manifest.add(rec)
            records.append(rec)
            log(f"  ✓ indexed as {rec.doc_id}")
        return records

    def remove(self, doc_id: str) -> bool:
        rec = self.manifest.get(doc_id)
        if not rec:
            return False
        self.store.delete_doc(doc_id)
        self.manifest.remove(doc_id)
        return True

    def reset(self) -> None:
        self.store.drop()
        self.manifest.clear()
