"""HTTP API. Streams answers as Server-Sent Events.

  POST /ingest            multipart files, ?hifi=&contextual=   → indexed documents
  GET  /documents
  DELETE /documents/{id}
  POST /ask               {"question", "session_id", "mode", "verify", "stream"}  → SSE (or JSON if stream=false)
  GET  /sessions          GET /sessions/{id}   DELETE /sessions/{id}
  GET  /health
"""
from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ragnet.engine import MODES, Engine

engine: Optional[Engine] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    engine = Engine()
    await asyncio.to_thread(engine.warmup)
    yield


app = FastAPI(title="RAGnet", version="2.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    mode: str = "auto"
    verify: Optional[bool] = None
    stream: bool = True


def _sse(ev) -> str:
    data = ev.data
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    return f"event: {ev.type}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/health")
async def health():
    return {"ok": True, "documents": len(engine.manifest), "llm": await engine.llm.ping()}


@app.get("/documents")
async def documents():
    return [asdict(d) for d in engine.manifest]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not engine.indexer.remove(doc_id):
        raise HTTPException(404, "no such document")
    return {"removed": doc_id}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...), hifi: bool = False, contextual: Optional[bool] = None, force: bool = False):
    engine.s.uploads_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for f in files:
        dest = engine.s.uploads_dir / Path(f.filename or "upload").name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
    log: list[str] = []
    recs = await engine.ingest(paths, hifi=hifi, contextual=contextual, force=force, progress=log.append)
    return {"documents": [asdict(r) for r in recs], "log": log}


@app.post("/ask")
async def ask(req: AskRequest):
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    gen = engine.ask(req.question, session_id=req.session_id, mode=req.mode, verify=req.verify)
    if req.stream:
        async def events():
            async for ev in gen:
                yield _sse(ev)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    result: dict = {"answer": "", "sources": [], "verification": [], "steps": []}
    async for ev in gen:
        if ev.type == "answer":
            result["answer"] = ev.data
        elif ev.type == "status":
            result["steps"].append(ev.data)
        elif ev.type in ("sources", "verification", "route", "done", "error"):
            result[ev.type] = ev.data
    return result


@app.get("/sessions")
async def sessions():
    return engine.sessions.list()


@app.get("/sessions/{session_id}")
async def session(session_id: str, turns: int = 50):
    return engine.sessions.history(session_id, turns)


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    engine.sessions.clear(session_id)
    return {"cleared": session_id}
