"""Golden-set evaluation.

golden.jsonl — one JSON object per line:
  {"question": "...", "answer": "<reference answer>", "doc": "<substring of doc name, optional>", "pages": [3, 4]}

Metrics:
  hit@k   – any returned source is on an expected page (of the expected doc)
  mrr     – 1 / rank of the first such source
  correct – LLM judge score (0 / 0.5 / 1) of the answer vs the reference
  latency – seconds per question
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from ragnet.agent import prompts
from ragnet.engine import Engine


def _retrieval_metrics(sources: list[dict], doc: Optional[str], pages: list[int]) -> tuple[bool, float]:
    if not pages:
        return True, 1.0
    for rank, s in enumerate(sources, 1):
        if doc and doc.lower() not in s["doc_name"].lower():
            continue
        if any(s["page"] <= p <= s["page_end"] for p in pages):
            return True, 1.0 / rank
    return False, 0.0


async def run_eval(engine: Engine, golden: Path, *, mode: str = "auto", limit: Optional[int] = None, out: Optional[Path] = None, console: Optional[Console] = None):
    console = console or Console()
    rows = [json.loads(l) for l in golden.read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    results = []
    for i, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        answer, sources, chosen = "", [], ""
        async for ev in engine.ask(row["question"], session_id=f"eval-{i}", mode=mode, verify=False):
            if ev.type == "answer":
                answer = ev.data
            elif ev.type == "sources":
                sources = ev.data
            elif ev.type == "route":
                chosen = ev.data["mode"]
        latency = time.perf_counter() - t0
        hit, rr = _retrieval_metrics(sources, row.get("doc"), row.get("pages", []))
        try:
            j = await engine.llm.json(
                [{"role": "user", "content": prompts.JUDGE.format(question=row["question"], reference=row.get("answer", ""), answer=answer)}],
                model=engine.s.router_model_name,
                max_tokens=200,
            )
            score, reason = float(j.get("score", 0)), str(j.get("reason", ""))
        except Exception as e:
            score, reason = 0.0, f"judge failed: {e}"
        results.append({**row, "mode": chosen, "answer": answer, "hit": hit, "rr": rr, "score": score, "reason": reason, "latency": latency})
        console.print(f"[dim]{i}/{len(rows)}[/dim] {'✓' if score == 1 else '½' if score == 0.5 else '✗'} hit={'✓' if hit else '✗'} {latency:.1f}s  {row['question'][:70]}")
        engine.sessions.clear(f"eval-{i}")

    n = len(results) or 1
    t = Table(title=f"Eval: {golden.name} ({len(results)} questions, mode={mode})")
    t.add_column("metric")
    t.add_column("value")
    t.add_row("hit@k", f"{sum(r['hit'] for r in results) / n:.2%}")
    t.add_row("MRR", f"{sum(r['rr'] for r in results) / n:.3f}")
    t.add_row("correctness", f"{sum(r['score'] for r in results) / n:.2%}")
    t.add_row("avg latency", f"{sum(r['latency'] for r in results) / n:.1f}s")
    t.add_row("p95 latency", f"{sorted(r['latency'] for r in results)[int(0.95 * (len(results) - 1))]:.1f}s")
    console.print(t)
    if out:
        out.write_text("\n".join(json.dumps(r) for r in results))
        console.print(f"wrote {out}")
    return results
