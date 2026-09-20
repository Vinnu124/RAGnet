"""Ingest-time LLM enrichment.

1. Contextual retrieval: for every chunk, the LLM writes 1–2 sentences that situate the chunk in the
   document; that context is prepended before embedding. Big recall win for chunks like "the table
   below shows…" that are meaningless in isolation.
2. Document summary + outline, used by the agent's `list_documents` tool.

All prompts share the same document prefix so vLLM's automatic prefix caching makes this cheap.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Optional

from ragnet.config import Settings
from ragnet.ingest.chunker import Chunk, count_tokens
from ragnet.ingest.parsers import ParsedDoc
from ragnet.llm.client import LLM

_CONTEXT_PROMPT = """<document>
{doc}
</document>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Write a short context (1-2 sentences) that situates this chunk within the overall document, for the purpose of improving search retrieval of the chunk. State what the document is about and what this specific chunk covers (section, entity, time period, table name, etc). Reply with the context only."""

_SUMMARY_PROMPT = """<document>
{doc}
</document>

Describe this document for a search index. Reply in JSON:
{{"summary": "<2-3 sentences: what it is, who wrote it, what it covers>", "outline": ["<main section or topic>", "..."]}}"""


def _doc_excerpt(doc: ParsedDoc, max_tokens: int) -> str:
    out, used = [], 0
    for p in doc.pages:
        t = count_tokens(p.text)
        if used + t > max_tokens:
            remaining = max_tokens - used
            if remaining > 200:
                out.append(p.text[: remaining * 4])
            break
        out.append(p.text)
        used += t
    return "\n\n".join(out)


class Contextualizer:
    def __init__(self, settings: Settings, llm: LLM):
        self.s = settings
        self.llm = llm

    async def summarize(self, doc: ParsedDoc) -> dict:
        excerpt = _doc_excerpt(doc, self.s.context_doc_tokens)
        try:
            data = await self.llm.json(
                [{"role": "user", "content": _SUMMARY_PROMPT.format(doc=excerpt)}],
                model=self.s.router_model_name,
                max_tokens=600,
            )
            return {"summary": str(data.get("summary", "")), "outline": [str(x) for x in data.get("outline", [])][:20]}
        except Exception as e:
            return {"summary": f"(summary unavailable: {e})", "outline": []}

    async def contextualize(self, doc: ParsedDoc, chunks: list[Chunk], progress: Optional[Callable[[int, int], None]] = None) -> None:
        excerpt = _doc_excerpt(doc, self.s.context_doc_tokens)
        done = 0

        async def one(chunk: Chunk):
            nonlocal done
            try:
                ctx = await self.llm.text(
                    [{"role": "user", "content": _CONTEXT_PROMPT.format(doc=excerpt, chunk=chunk.text)}],
                    model=self.s.router_model_name,
                    max_tokens=120,
                )
                chunk.context = ctx.strip().strip('"')
            except Exception:
                chunk.context = ""
            done += 1
            if progress:
                progress(done, len(chunks))

        await asyncio.gather(*(one(c) for c in chunks))
