"""Tool schemas (OpenAI function-calling format) and their executors."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from ragnet.agent.evidence import Evidence
from ragnet.config import Settings
from ragnet.manifest import Manifest
from ragnet.retrieval.retriever import Retriever
from ragnet.retrieval.store import Store

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Semantic + keyword search over the documents. Returns the most relevant passages with citation ids. Call it several times with different phrasings or sub-questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A specific, self-contained search query."},
                    "doc": {"type": "string", "description": "Optional: restrict to one document (name or id from list_documents)."},
                    "top_k": {"type": "integer", "description": "Number of passages to return (default 6, max 12)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Exact text match across documents. Use for identifiers, codes, part numbers, exact names, dates or numbers that semantic search may miss.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "Exact string to find (case-insensitive)."},
                    "doc": {"type": "string", "description": "Optional: restrict to one document."},
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the full text of one page of a document (use when a passage is cut off or you need the surrounding table/section).",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc": {"type": "string", "description": "Document name or id."},
                    "page": {"type": "integer", "description": "1-based page number."},
                },
                "required": ["doc", "page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all documents in the collection with id, name, page count, summary and outline.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolExecutor:
    def __init__(self, settings: Settings, retriever: Retriever, store: Store, manifest: Manifest, evidence: Evidence):
        self.s = settings
        self.retriever = retriever
        self.store = store
        self.manifest = manifest
        self.evidence = evidence
        self.calls: list[dict] = []  # trace for UI/eval

    def _doc_ids(self, ref: Optional[str]) -> Optional[list[str]]:
        if not ref:
            return None
        rec = self.manifest.resolve(ref)
        if rec is None:
            raise ValueError(f"No document matches {ref!r}. Call list_documents to see valid names.")
        return [rec.doc_id]

    async def run(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "search":
                k = max(1, min(int(args.get("top_k") or self.s.top_k), 12))
                hits = await self.retriever.search(str(args["query"]), top_k=k, doc_ids=self._doc_ids(args.get("doc")))
                out = self.evidence.format(self.evidence.add(hits))
            elif name == "grep":
                hits = await asyncio.to_thread(self.store.grep, str(args["term"]), limit=self.s.grep_limit, doc_ids=self._doc_ids(args.get("doc")))
                out = self.evidence.format(self.evidence.add(hits)) if hits else f"No exact matches for {args['term']!r}."
            elif name == "read_page":
                ids = self._doc_ids(str(args["doc"]))
                hits = await asyncio.to_thread(self.store.get_page, ids[0], int(args["page"]))
                out = self.evidence.format(self.evidence.add(hits)) if hits else f"Page {args['page']} has no indexed text."
            elif name == "list_documents":
                out = self.list_documents()
            else:
                out = f"Unknown tool {name}"
        except Exception as e:  # tool errors go back to the model, not up the stack
            out = f"Tool error: {e}"
        self.calls.append({"tool": name, "args": args, "chars": len(out)})
        return out

    def list_documents(self) -> str:
        if not len(self.manifest):
            return "No documents have been ingested."
        rows = []
        for d in self.manifest:
            row = f"- id={d.doc_id}  name=\"{d.name}\"  pages={d.pages}  chunks={d.chunks}"
            if d.summary:
                row += f"\n  summary: {d.summary}"
            if d.outline:
                row += "\n  outline: " + "; ".join(d.outline[:12])
            rows.append(row)
        return "\n".join(rows)


def parse_tool_args(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        from ragnet.llm.client import parse_json

        try:
            return parse_json(raw)
        except ValueError:
            return {}
