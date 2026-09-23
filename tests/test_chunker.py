from ragnet.ingest.chunker import chunk_pages, count_tokens
from ragnet.ingest.parsers import Page


_WORDS = "the quick brown fox jumps over the lazy dog while revenue grows and costs fall".split()


def _para(n_words: int, tag: str) -> str:
    """Realistic prose (~1 token/word) so token budgets behave like real documents; `tag` keeps paragraphs distinct."""
    words = [tag] + [_WORDS[i % len(_WORDS)] for i in range(n_words - 1)]
    return " ".join(words) + "."


def test_sections_pages_and_token_bounds():
    pages = [
        Page(1, "# Intro\n\n" + _para(80, "a") + "\n\n" + _para(80, "b")),
        Page(2, _para(80, "c") + "\n\n## Methods\n\n" + _para(80, "d") + "\n\n| col1 | col2 |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"),
    ]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="doc.pdf", max_tokens=120, overlap_tokens=30)
    assert chunks, "no chunks produced"
    assert all(c.tokens <= 120 + 30 for c in chunks)
    assert chunks[0].section == "Intro"
    assert any(c.section == "Intro > Methods" for c in chunks)
    # heading text is part of the chunk, table kept intact
    assert any("| 3 | 4 |" in c.text and "| col1" in c.text for c in chunks)
    # a chunk that started on page 1 and continued into page 2 reports both
    spanning = [c for c in chunks if c.page != c.page_end]
    assert all(c.page == 1 and c.page_end == 2 for c in spanning)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_oversize_paragraph_is_split_on_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_pages([Page(1, text)], doc_id="d", doc_name="x.txt", max_tokens=100, overlap_tokens=0)
    assert len(chunks) > 3
    assert all(c.tokens <= 110 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_overlap_repeats_trailing_block():
    paras = [_para(40, f"p{i}_") for i in range(6)]
    chunks = chunk_pages([Page(1, "\n\n".join(paras))], doc_id="d", doc_name="x", max_tokens=150, overlap_tokens=60)
    assert len(chunks) >= 2
    last_block_of_first = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].text.startswith(last_block_of_first)


def test_embed_text_includes_header_and_context():
    chunks = chunk_pages([Page(1, "# Title\n\nbody text here.")], doc_id="d", doc_name="report.pdf")
    c = chunks[0]
    c.context = "This is about X."
    assert c.embed_text.startswith("report.pdf > Title")
    assert "This is about X." in c.embed_text
    assert c.page_label == "p.1"


def test_count_tokens_positive():
    assert count_tokens("hello world") >= 1
