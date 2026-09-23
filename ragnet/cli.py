"""`ragnet` command line."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from ragnet.engine import MODES, Engine

app = typer.Typer(help="RAGnet — fast, accurate, agentic RAG on your own GPU box.", no_args_is_help=True)
console = Console()


def _engine() -> Engine:
    return Engine()


# --------------------------------------------------------------------------- ingest
@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Files or directories (pdf, md, txt, html, docx)"),
    hifi: bool = typer.Option(False, help="Use Docling (layout model + OCR). Slower, better on scanned/table-heavy PDFs."),
    contextual: Optional[bool] = typer.Option(None, "--contextual/--no-contextual", help="LLM-written context per chunk (contextual retrieval). Default from settings."),
    force: bool = typer.Option(False, help="Re-index even if the file is unchanged."),
):
    """Parse, chunk, embed and index documents (incremental; unchanged files are skipped)."""
    eng = _engine()
    recs = asyncio.run(eng.ingest(paths, hifi=hifi, contextual=contextual, force=force, progress=console.print))
    _print_docs(eng, recs)


@app.command()
def docs():
    """List indexed documents."""
    eng = _engine()
    _print_docs(eng, list(eng.manifest))


def _print_docs(eng: Engine, recs):
    t = Table(title=f"{len(recs)} document(s) · {eng.manifest.total_tokens} tokens total")
    for col in ("id", "name", "pages", "chunks", "tokens", "ctx", "summary"):
        t.add_column(col)
    for d in recs:
        t.add_row(d.doc_id, d.name, str(d.pages), str(d.chunks), str(d.tokens), "✓" if d.contextual else "", (d.summary or "")[:90])
    console.print(t)


@app.command()
def remove(doc_id: str):
    """Remove one document from the index."""
    ok = _engine().indexer.remove(doc_id)
    console.print("removed" if ok else f"no document with id {doc_id}")


@app.command()
def reset(yes: bool = typer.Option(False, "--yes", "-y")):
    """Delete the whole index and manifest."""
    if not yes and not typer.confirm("Delete the entire index?"):
        raise typer.Abort()
    _engine().indexer.reset()
    console.print("index cleared")


# --------------------------------------------------------------------------- ask / chat
async def _ask_and_print(eng: Engine, question: str, session: str, mode: str, verify: Optional[bool], show_sources: bool):
    streamed = False
    async for ev in eng.ask(question, session_id=session, mode=mode, verify=verify):
        if ev.type == "route":
            console.print(f"[dim]mode={ev.data['mode']}  q={ev.data['question']}[/dim]")
        elif ev.type == "status":
            console.print(f"[dim]  ⋯ {ev.data}[/dim]")
        elif ev.type == "token":
            if not streamed:
                console.print()
            streamed = True
            console.print(ev.data, end="", soft_wrap=True, highlight=False, markup=False)
        elif ev.type == "answer":
            if streamed:
                console.print()
            else:
                console.print()
                console.print(Markdown(ev.data))
        elif ev.type == "sources" and show_sources:
            _print_sources(ev.data)
        elif ev.type == "verification":
            _print_verification(ev.data)
        elif ev.type == "error":
            console.print(f"[red]{ev.data}[/red]")
        elif ev.type == "done":
            console.print(f"[dim]({ev.data['seconds']}s, {ev.data['mode']})[/dim]")


def _print_sources(sources: list[dict]):
    if not sources:
        return
    console.print("\n[bold]Sources[/bold]")
    for s in sources:
        snippet = s["text"].replace("\n", " ")
        console.print(f"  [{s['n']}] {s['locator']}\n      [dim]{snippet[:220]}{'…' if len(snippet) > 220 else ''}[/dim]")


def _print_verification(claims: list[dict]):
    if not claims:
        return
    bad = [c for c in claims if not c["supported"]]
    if not bad:
        console.print(f"[green]✓ {len(claims)}/{len(claims)} cited claims verified against sources[/green]")
        return
    console.print(f"[yellow]⚠ {len(bad)}/{len(claims)} cited claims NOT supported by their sources:[/yellow]")
    for c in bad:
        console.print(f"   • {c['text'][:160]}\n     [dim]{c['note']}[/dim]")


@app.command()
def ask(
    question: str,
    mode: str = typer.Option("auto", help="auto | lookup | research | longctx | chat"),
    session: str = typer.Option("cli", help="Conversation id (history is remembered per session)"),
    verify: Optional[bool] = typer.Option(None, "--verify/--no-verify"),
    sources: bool = typer.Option(True, "--sources/--no-sources"),
):
    """Ask one question."""
    if mode not in MODES:
        raise typer.BadParameter(f"mode must be one of {MODES}")
    asyncio.run(_ask_and_print(_engine(), question, session, mode, verify, sources))


@app.command()
def chat(
    session: str = typer.Option("cli", help="Conversation id"),
    mode: str = typer.Option("auto"),
    verify: Optional[bool] = typer.Option(None, "--verify/--no-verify"),
    sources: bool = typer.Option(True, "--sources/--no-sources"),
):
    """Interactive chat with memory. Commands: /mode <m>, /clear, /docs, exit."""
    eng = _engine()
    console.print("[bold]RAGnet[/bold] — type a question, or /mode lookup|research|longctx|auto, /clear, /docs, exit")
    while True:
        try:
            q = console.input("\n[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.startswith("/mode"):
            parts = q.split()
            if len(parts) == 2 and parts[1] in MODES:
                mode = parts[1]
            console.print(f"mode = {mode}")
            continue
        if q == "/clear":
            eng.sessions.clear(session)
            console.print("history cleared")
            continue
        if q == "/docs":
            _print_docs(eng, list(eng.manifest))
            continue
        asyncio.run(_ask_and_print(eng, q, session, mode, verify, sources))


# --------------------------------------------------------------------------- serve / eval
@app.command()
def serve(host: Optional[str] = None, port: Optional[int] = None):
    """Run the HTTP API (FastAPI, SSE streaming)."""
    import uvicorn

    from ragnet.api.server import app as api_app
    from ragnet.config import get_settings

    s = get_settings()
    uvicorn.run(api_app, host=host or s.host, port=port or s.port)


@app.command()
def eval(
    golden: Path = typer.Argument(Path("eval/golden.jsonl")),
    mode: str = typer.Option("auto"),
    limit: Optional[int] = typer.Option(None),
    out: Optional[Path] = typer.Option(None, help="Write per-question results as JSONL"),
):
    """Run the golden-set evaluation (retrieval hit rate, MRR, answer correctness, latency)."""
    from ragnet.eval import run_eval

    asyncio.run(run_eval(_engine(), golden, mode=mode, limit=limit, out=out, console=console))


if __name__ == "__main__":
    app()
