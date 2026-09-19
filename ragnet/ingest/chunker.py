"""Structure-aware, token-bounded chunking.

- Tracks markdown headings so every chunk carries its section path (e.g. "3 Results > 3.2 Ablations").
- Keeps tables atomic where possible.
- Chunks may span page boundaries (so we don't get useless 2-line chunks at page ends), and record page/page_end.
- Overlap is a tail of whole blocks (paragraphs), never mid-sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from ragnet.ingest.parsers import Page

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    try:
        return len(_encoder().encode(text, disallowed_special=()))
    except Exception:  # tiktoken data unavailable (offline) → rough estimate
        return max(1, len(text) // 4)


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    index: int
    page: int
    page_end: int
    section: str
    text: str
    tokens: int
    context: str = ""  # optional LLM-written situating context (contextual retrieval)

    @property
    def embed_text(self) -> str:
        head = f"{self.doc_name}"
        if self.section:
            head += f" > {self.section}"
        parts = [head]
        if self.context:
            parts.append(self.context)
        parts.append(self.text)
        return "\n\n".join(parts)

    @property
    def page_label(self) -> str:
        return f"p.{self.page}" if self.page == self.page_end else f"p.{self.page}-{self.page_end}"


@dataclass
class _Block:
    text: str
    page: int
    tokens: int
    is_heading: bool = False


def _blocks_from_page(page: Page) -> Iterable[_Block]:
    """Split a page's markdown into paragraph/table/heading blocks."""
    lines = page.text.splitlines()
    buf: list[str] = []
    in_table = False

    def flush():
        nonlocal buf
        if buf:
            text = "\n".join(buf).strip()
            if text:
                yield _Block(text=text, page=page.number, tokens=count_tokens(text))
        buf = []

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|")
        heading = _HEADING_RE.match(stripped)
        if heading:
            yield from flush()
            in_table = False
            yield _Block(text=stripped, page=page.number, tokens=count_tokens(stripped), is_heading=True)
            continue
        if not stripped:
            if not in_table:
                yield from flush()
            continue
        if is_table_line != in_table:
            yield from flush()
            in_table = is_table_line
        buf.append(line.rstrip())
    yield from flush()


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Split an oversize block on sentence boundaries (tables: on rows)."""
    if text.lstrip().startswith("|"):
        units = text.splitlines()
        joiner = "\n"
    else:
        units = re.split(r"(?<=[.!?])\s+", text)
        joiner = " "
    # a single unit (one huge sentence / table row) can still exceed the budget → hard-split on whitespace
    expanded: list[str] = []
    for u in units:
        if count_tokens(u) <= max_tokens:
            expanded.append(u)
            continue
        words, buf, buf_tok = u.split(), [], 0
        for w in words:
            t = count_tokens(w + " ")
            if buf and buf_tok + t > max_tokens:
                expanded.append(" ".join(buf))
                buf, buf_tok = [], 0
            buf.append(w)
            buf_tok += t
        if buf:
            expanded.append(" ".join(buf))
    out, cur, cur_tok = [], [], 0
    for u in expanded:
        t = count_tokens(u)
        if cur and cur_tok + t > max_tokens:
            out.append(joiner.join(cur))
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += t
    if cur:
        out.append(joiner.join(cur))
    return out


def chunk_pages(
    pages: list[Page],
    *,
    doc_id: str,
    doc_name: str,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    section_stack: list[tuple[int, str]] = []  # (level, title)
    cur: list[_Block] = []
    cur_tokens = 0
    cur_section = ""

    def section_path() -> str:
        return " > ".join(t for _, t in section_stack)

    def flush(next_blocks_overlap: bool = True):
        nonlocal cur, cur_tokens
        body = [b for b in cur if not (b.is_heading and len(cur) == 1)]
        if not body:
            cur, cur_tokens = [], 0
            return
        text = "\n\n".join(b.text for b in cur).strip()
        chunks.append(
            Chunk(
                doc_id=doc_id,
                doc_name=doc_name,
                index=len(chunks),
                page=cur[0].page,
                page_end=cur[-1].page,
                section=cur_section,
                text=text,
                tokens=count_tokens(text),
            )
        )
        # overlap: keep trailing non-heading blocks up to overlap_tokens
        tail: list[_Block] = []
        tok = 0
        if next_blocks_overlap:
            for b in reversed(cur):
                if b.is_heading or tok + b.tokens > overlap_tokens:
                    break
                tail.insert(0, b)
                tok += b.tokens
        cur, cur_tokens = tail, tok

    for page in pages:
        for block in _blocks_from_page(page):
            if block.is_heading:
                m = _HEADING_RE.match(block.text)
                level, title = len(m.group(1)), m.group(2).strip()
                # a heading starts a new chunk (no overlap across sections)
                if cur and any(not b.is_heading for b in cur):
                    flush(next_blocks_overlap=False)
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, title))
                cur_section = section_path()
                cur, cur_tokens = [block], block.tokens
                continue

            pieces = [block.text] if block.tokens <= max_tokens else _split_long(block.text, max_tokens)
            for piece in pieces:
                pb = _Block(text=piece, page=block.page, tokens=count_tokens(piece))
                if cur and cur_tokens + pb.tokens > max_tokens:
                    flush()
                if not cur:
                    cur_section = section_path()
                cur.append(pb)
                cur_tokens += pb.tokens
    if cur:
        flush(next_blocks_overlap=False)
    return chunks
