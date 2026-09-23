"""Engine: wires everything together and exposes `ingest()` and the streaming `ask()` used by CLI and API.

Execution paths for a question (chosen by the router unless `mode` is forced):
  chat         → no retrieval, short conversational reply
  long-context → whole corpus fits in the context window: skip retrieval, feed everything (most accurate)
  lookup       → one hybrid search + rerank → streamed answer with citations
  research     → agentic loop with tools (search / grep / read_page / list_documents)
Afterwards (optional): claim-by-claim citation verification.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import AsyncIterator, Optional

from ragnet.agent import prompts
from ragnet.agent.evidence import Evidence
from ragnet.agent.loop import AgentResult, Event, run_agent
from ragnet.agent.router import Router
from ragnet.agent.tools import ToolExecutor
from ragnet.agent.verifier import Verifier
from ragnet.config import Settings, get_settings
from ragnet.ingest.indexer import Indexer
from ragnet.llm.client import LLM
from ragnet.manifest import Manifest
from ragnet.memory.sessions import Sessions
from ragnet.retrieval.embedder import Embedder
from ragnet.retrieval.reranker import Reranker
from ragnet.retrieval.retriever import Retriever
from ragnet.retrieval.store import Hit, Store

MODES = ("auto", "chat", "lookup", "research", "longctx")


class Engine:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLM(self.s)
        self.manifest = Manifest(self.s.manifest_path)
        self.embedder = Embedder(self.s)
        self.store = Store(self.s)
        self.reranker = Reranker(self.s)
        self.retriever = Retriever(self.s, self.embedder, self.store, self.reranker)
        self.indexer = Indexer(self.s, self.embedder, self.store, self.manifest, self.llm)
        self.router = Router(self.s, self.llm, self.manifest)
        self.verifier = Verifier(self.s, self.llm)
        self.sessions = Sessions(self.s.sessions_db)

    def warmup(self) -> None:
        """Load models eagerly (call once at server start so the first query isn't slow)."""
        self.embedder.embed_query("warmup")
        self.reranker.rerank("warmup", [], 1)

    # ------------------------------------------------------------------ ingest
    async def ingest(self, paths: list[Path], **kw):
        return await self.indexer.ingest(paths, **kw)

    # ------------------------------------------------------------------ ask
    def _long_context_ok(self) -> bool:
        budget = min(self.s.long_context_max_tokens, self.s.llm_context_window - self.s.llm_max_tokens - 2000)
        return 0 < self.manifest.total_tokens <= budget

    async def ask(self, question: str, *, session_id: str = "default", mode: str = "auto", verify: Optional[bool] = None) -> AsyncIterator[Event]:
        t0 = time.perf_counter()
        verify = self.s.verify if verify is None else verify
        history = self.sessions.history(session_id, self.s.history_turns)

        if not len(self.manifest):
            yield Event("error", "No documents ingested yet. Run `ragnet ingest <files>` first.")
            return

        # 1. route (condense follow-up + classify)
        if mode == "auto" or mode == "longctx":
            route = await self.router.route(question, history)
            q = route.question
            chosen = route.mode
            if mode == "longctx":
                chosen = "longctx"
            elif chosen != "chat" and self._long_context_ok():
                chosen = "longctx"
        else:
            q, chosen = question, mode
        yield Event("route", {"mode": chosen, "question": q})

        # 2. answer
        evidence = Evidence()
        answer = ""
        if chosen == "chat":
            async for ev in self._chat(question, history):
                if ev.type == "token":
                    answer += ev.data
                yield ev
        elif chosen == "longctx":
            async for ev in self._long_context(q, history, evidence):
                if ev.type == "token":
                    answer += ev.data
                yield ev
        elif chosen == "lookup":
            async for ev in self._fast(q, history, evidence):
                if ev.type == "token":
                    answer += ev.data
                yield ev
        else:
            tools = ToolExecutor(self.s, self.retriever, self.store, self.manifest, evidence)
            async for ev in run_agent(self.s, self.llm, tools, q, history):
                if ev.type == "answer":
                    res: AgentResult = ev.data
                    answer = res.answer
                    yield Event("status", f"done in {res.steps} step(s), {len(res.evidence)} passages read")
                else:
                    yield ev

        yield Event("answer", answer)
        if chosen != "chat":
            yield Event("sources", evidence.sources(answer))

        # 3. verify
        if verify and chosen != "chat" and evidence.cited(answer):
            yield Event("status", "verifying citations")
            claims = await self.verifier.verify(answer, evidence)
            yield Event("verification", [c.to_dict() for c in claims])

        # 4. remember
        self.sessions.add(session_id, "user", question)
        self.sessions.add(session_id, "assistant", answer)
        yield Event("done", {"seconds": round(time.perf_counter() - t0, 2), "mode": chosen})

    # ------------------------------------------------------------------ paths
    async def _chat(self, question: str, history: list[dict]) -> AsyncIterator[Event]:
        docs = ", ".join(d.name for d in self.manifest) or "(none)"
        messages = [{"role": "system", "content": prompts.CHAT_SYSTEM.format(documents=docs)}] + history + [{"role": "user", "content": question}]
        async for tok in self.llm.stream(messages, model=self.s.router_model_name, max_tokens=400):
            yield Event("token", tok)

    async def _fast(self, q: str, history: list[dict], evidence: Evidence) -> AsyncIterator[Event]:
        yield Event("status", f"search: {q}")
        hits = await self.retriever.search(q)
        items = evidence.add(hits)
        user = f"Evidence:\n\n{evidence.format(items)}\n\nQuestion: {q}"
        messages = [{"role": "system", "content": prompts.FAST_SYSTEM}] + history + [{"role": "user", "content": user}]
        async for tok in self.llm.stream(messages):
            yield Event("token", tok)

    async def _long_context(self, q: str, history: list[dict], evidence: Evidence) -> AsyncIterator[Event]:
        yield Event("status", f"corpus fits in context ({self.manifest.total_tokens} tokens) — reading everything")
        page_hits: list[Hit] = []
        for rec in self.manifest:
            chunks = self.store.get_doc(rec.doc_id)
            # one evidence item per page keeps citation ids meaningful (doc + page)
            by_page: dict[int, list[Hit]] = {}
            for c in chunks:
                by_page.setdefault(c.page, []).append(c)
            for page, cs in sorted(by_page.items()):
                first = cs[0]
                page_hits.append(
                    Hit(
                        id=f"{rec.doc_id}:page:{page}",
                        score=0.0,
                        text=_dedupe_overlap([c.text for c in cs]),
                        context="",
                        doc_id=rec.doc_id,
                        doc_name=rec.name,
                        page=page,
                        page_end=max(c.page_end for c in cs),
                        section=first.section,
                        index=first.index,
                    )
                )
        items = evidence.add(page_hits)
        system = prompts.LONG_CONTEXT_SYSTEM.format(documents=evidence.format(items))
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": q}]
        async for tok in self.llm.stream(messages):
            yield Event("token", tok)


def _dedupe_overlap(texts: list[str]) -> str:
    """Chunks overlap by design; when concatenating a page, drop a repeated leading paragraph."""
    out: list[str] = []
    for t in texts:
        if out:
            prev_tail = out[-1].split("\n\n")[-1]
            if t.startswith(prev_tail):
                t = t[len(prev_tail) :].lstrip()
        if t:
            out.append(t)
    return "\n\n".join(out)
