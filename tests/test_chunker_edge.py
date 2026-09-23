from ragnet.ingest.chunker import chunk_pages
from ragnet.ingest.parsers import Page


def test_single_giant_sentence_is_hard_split():
    text = " ".join(f"tok{i}" for i in range(600))  # no sentence boundaries at all
    chunks = chunk_pages([Page(1, text)], doc_id="d", doc_name="x", max_tokens=100, overlap_tokens=0)
    assert len(chunks) > 5
    assert all(c.tokens <= 105 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == text.split()  # nothing lost
