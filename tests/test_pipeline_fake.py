"""End-to-end test of store → tools → agent loop → engine with a fake embedder and a scripted fake LLM.
Exercises real Qdrant (embedded, in a temp dir) including hybrid RRF fusion, page reads, grep, and the
full tool-calling transcript shape — no GPU or LLM server needed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import models

from ragnet.config import Settings
from ragnet.engine import Engine
from ragnet.ingest.chunker import chunk_pages
from ragnet.ingest.parsers import Page
from ragnet.manifest import DocRecord
from ragnet.retrieval.store import Hit

DIM = 16


class FakeEmbedder:
    """Deterministic bag-of-words hashing: dense = hashed word counts, sparse = word ids."""

    device = "cpu"
    dim = DIM

    def _dense(self, text):
        v = np.zeros(DIM)
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def _sparse(self, text):
        idx = sorted({int(hashlib.md5(w.encode()).hexdigest(), 16) % 10_000 for w in text.lower().split()})
        return models.SparseVector(indices=idx, values=[1.0] * len(idx))

    def embed_docs(self, texts):
        return [self._dense(t) for t in texts], [self._sparse(t) for t in texts]

    def embed_query(self, text):
        return self._dense(text), self._sparse(text)

    def embed_queries(self, texts):
        return [self.embed_query(t) for t in texts]


class FakeReranker:
    def rerank(self, query, hits, top_k):
        qw = set(query.lower().split())
        for h in hits:
            h.score = len(qw & set(h.text.lower().split()))
        return sorted(hits, key=lambda h: -h.score)[:top_k]


class FakeLLM:
    """Scripted responses; records what it was asked."""

    def __init__(self):
        self.calls: list[dict] = []
        self.script: list = []

    def _tc(self, name, args):
        return SimpleNamespace(id=f"call_{len(self.calls)}_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    async def chat(self, messages, *, tools=None, model=None, temperature=0.0, max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools})
        step = self.script.pop(0)
        if step["type"] == "tools":
            return SimpleNamespace(content="", tool_calls=[self._tc(n, a) for n, a in step["calls"]])
        return SimpleNamespace(content=step["text"], tool_calls=None)

    async def text(self, messages, **kw):
        return (await self.chat(messages)).content

    async def json(self, messages, **kw):
        self.calls.append({"messages": messages})
        return self.script.pop(0)["json"]

    async def stream(self, messages, **kw):
        self.calls.append({"messages": messages})
        for tok in self.script.pop(0)["text"].split(" "):
            yield tok + " "

    async def ping(self):
        return True


@pytest.fixture
def engine(tmp_path: Path):
    s = Settings(data_dir=tmp_path / "data", verify=True, long_context_max_tokens=0, top_k=3, prefetch_k=10)
    eng = Engine(s)
    eng.embedder = FakeEmbedder()
    eng.reranker = FakeReranker()
    eng.llm = FakeLLM()
    # rewire components that captured the originals
    eng.retriever.embedder, eng.retriever.reranker = eng.embedder, eng.reranker
    eng.indexer.embedder, eng.indexer.llm = eng.embedder, eng.llm
    eng.router.llm = eng.llm
    eng.verifier.llm = eng.llm
    # index two small docs directly (bypass file parsing)
    eng.store.ensure_collection(DIM)
    docs = {
        "manual": [Page(1, "# Warranty\n\nModel A warranty period is 24 months."), Page(2, "Model B warranty period is 36 months. Serial format SN-7781.")],
        "report": [Page(1, "# Revenue\n\nTotal revenue for FY2024 was 12.4 million dollars.")],
    }
    for name, pages in docs.items():
        doc_id = name[:12]
        chunks = chunk_pages(pages, doc_id=doc_id, doc_name=f"{name}.pdf")
        d, sp = eng.embedder.embed_docs([c.embed_text for c in chunks])
        eng.store.upsert(chunks, d, sp)
        eng.manifest.add(DocRecord(doc_id=doc_id, name=f"{name}.pdf", path="/x", sha256=name, pages=len(pages), chunks=len(chunks), tokens=sum(c.tokens for c in chunks), summary=f"{name} summary"))
    return eng


async def _collect(gen):
    evs = []
    async for ev in gen:
        evs.append(ev)
    return evs


async def test_store_hybrid_page_and_grep(engine):
    d, sp = engine.embedder.embed_query("warranty period Model B")
    hits = engine.store.hybrid_search(d, sp, limit=5)
    assert hits and all(isinstance(h, Hit) for h in hits)
    page2 = engine.store.get_page("manual", 2)
    assert page2 and "36 months" in page2[0].text
    g = engine.store.grep("SN-7781", limit=5)[0]
    assert g.page <= 2 <= g.page_end and "SN-7781" in g.text
    assert engine.store.grep("SN-7781", limit=5, doc_ids=["report"]) == []
    assert engine.store.get_doc("report")[0].section == "Revenue"


async def test_lookup_path_streams_and_verifies(engine):
    llm: FakeLLM = engine.llm
    llm.script = [
        {"json": {"question": "What is Model B's warranty period?", "mode": "lookup"}},
        {"text": "Model B has a 36 month warranty [1]."},
        {"json": {"results": [{"id": 1, "supported": True, "note": ""}]}},
    ]
    evs = await _collect(engine.ask("and model B?", session_id="t", mode="auto"))
    types = [e.type for e in evs]
    assert types[0] == "route" and evs[0].data["mode"] == "lookup"
    assert "token" in types and "answer" in types and "sources" in types and "verification" in types
    answer = next(e.data for e in evs if e.type == "answer")
    assert "36 month" in answer
    sources = next(e.data for e in evs if e.type == "sources")
    assert sources[0]["cited"] and sources[0]["n"] == 1
    # the fast-path prompt contained real evidence
    fast_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "[1] manual.pdf" in fast_prompt and "Question: What is Model B's warranty period?" in fast_prompt
    # memory persisted
    assert engine.sessions.history("t")[-1]["content"] == answer


async def test_research_path_tool_loop(engine):
    llm: FakeLLM = engine.llm
    llm.script = [
        {"json": {"question": "Compare warranty of A and B", "mode": "research"}},
        {"type": "tools", "calls": [("search", {"query": "Model A warranty period"}), ("search", {"query": "Model B warranty period"})]},
        {"type": "tools", "calls": [("read_page", {"doc": "manual", "page": 2}), ("grep", {"term": "SN-7781"}), ("list_documents", {})]},
        {"type": "final", "text": "Model A: 24 months [1]. Model B: 36 months [2]. Made up thing [9]."},
        {"json": {"results": [{"id": 1, "supported": True}, {"id": 2, "supported": True}, {"id": 3, "supported": False, "note": "not in evidence"}]}},
    ]
    evs = await _collect(engine.ask("compare them", session_id="r", mode="auto"))
    statuses = [e.data for e in evs if e.type == "status"]
    assert any(s.startswith("search: Model A") for s in statuses)
    assert any("read page 2" in s for s in statuses)
    # transcript shape: system, user, assistant(tool_calls), 2 tool msgs, assistant(tool_calls), 3 tool msgs → final call
    final_call = llm.calls[3]["messages"]
    roles = [m["role"] for m in final_call]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "tool", "tool", "tool"]
    tool_msgs = [m for m in final_call if m["role"] == "tool"]
    assert any("Model A warranty" in m["content"] for m in tool_msgs)
    assert any("manual.pdf" in m["content"] and "report.pdf" in m["content"] for m in tool_msgs)  # list_documents
    assert final_call[2]["tool_calls"][0]["function"]["name"] == "search"
    ver = next(e.data for e in evs if e.type == "verification")
    bad = [c for c in ver if not c["supported"]]
    assert len(bad) == 1 and "non-existent evidence id" in bad[0]["note"]


async def test_long_context_path(engine):
    engine.s.long_context_max_tokens = 5000
    llm: FakeLLM = engine.llm
    llm.script = [
        {"json": {"question": "What was FY2024 revenue?", "mode": "lookup"}},
        {"text": "12.4 million dollars [2]."},
        {"json": {"results": [{"id": 1, "supported": True}]}},
    ]
    evs = await _collect(engine.ask("revenue?", session_id="l"))
    assert evs[0].data["mode"] == "longctx"
    system = llm.calls[1]["messages"][0]["content"]
    assert "[1] manual.pdf · p.1-2" in system and "[2] report.pdf · p.1" in system
    sources = next(e.data for e in evs if e.type == "sources")
    assert sources[0]["doc_name"] == "report.pdf"


async def test_agent_step_cap_forces_answer(engine):
    engine.s.agent_max_steps = 2
    llm: FakeLLM = engine.llm
    llm.script = [
        {"type": "tools", "calls": [("search", {"query": "anything"})]},
        {"type": "final", "text": "Could not determine [1]."},
        {"json": {"results": []}},
    ]
    evs = await _collect(engine.ask("q", session_id="c", mode="research", verify=False))
    assert llm.calls[1]["tools"] is None  # forced final answer without tools
    assert next(e.data for e in evs if e.type == "answer").startswith("Could not")
