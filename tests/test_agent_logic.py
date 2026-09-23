import pytest

from ragnet.agent.evidence import Evidence, extract_citations
from ragnet.agent.tools import parse_tool_args
from ragnet.agent.verifier import split_claims
from ragnet.llm.client import parse_json, strip_thinking
from ragnet.manifest import DocRecord, Manifest
from ragnet.memory.sessions import Sessions
from ragnet.retrieval.store import Hit


def _hit(i: int, doc="a.pdf", page=1) -> Hit:
    return Hit(id=f"id{i}", score=0.5, text=f"text {i}", context="", doc_id="d", doc_name=doc, page=page, page_end=page, section="S", index=i)


def test_evidence_ids_are_stable_across_calls():
    ev = Evidence()
    first = ev.add([_hit(1), _hit(2)])
    second = ev.add([_hit(2), _hit(3)])
    assert [n for n, _ in first] == [1, 2]
    assert [n for n, _ in second] == [2, 3]
    assert len(ev) == 3
    assert ev.get(2).id == "id2"
    assert "[2] a.pdf · p.1 · §S" in ev.format(second)


def test_citations_and_sources():
    ev = Evidence()
    ev.add([_hit(1), _hit(2), _hit(3)])
    assert extract_citations("Revenue rose [2]. Costs fell [3][2]. Bad [99].") == [2, 3, 99]
    cited = ev.cited("x [3] y [1]")
    assert [n for n, _ in cited] == [3, 1]
    src = ev.sources("nothing cited here")
    assert len(src) == 3 and not src[0]["cited"]


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('<think>hmm</think> here: {"mode": "lookup"} thanks') == {"mode": "lookup"}
    assert parse_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        parse_json("no json here")
    assert strip_thinking("<think>a\nb</think>answer") == "answer"


def test_parse_tool_args():
    assert parse_tool_args('{"query": "x"}') == {"query": "x"}
    assert parse_tool_args({"query": "x"}) == {"query": "x"}
    assert parse_tool_args("") == {}
    assert parse_tool_args("garbage") == {}


def test_split_claims_only_cited_sentences():
    answer = "Revenue was $12.4M in 2024 [1]. This is uncited filler. Costs were lower [2][3].\n- Item one grew [4]"
    claims = split_claims(answer)
    assert [c.cites for c in claims] == [[1], [2, 3], [4]]
    assert claims[2].text.startswith("Item one")


def test_sessions_roundtrip(tmp_path):
    s = Sessions(tmp_path / "s.db")
    s.add("a", "user", "hi")
    s.add("a", "assistant", "hello")
    s.add("b", "user", "other")
    assert s.history("a") == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert [x["session_id"] for x in s.list()] == ["b", "a"]
    s.clear("a")
    assert s.history("a") == []


def test_manifest_resolve_and_persist(tmp_path):
    m = Manifest(tmp_path / "m.json")
    m.add(DocRecord(doc_id="abc123", name="Annual Report 2024.pdf", path="/x", sha256="ff", pages=10, chunks=40, tokens=9000))
    m.add(DocRecord(doc_id="def456", name="manual.pdf", path="/y", sha256="ee", pages=3, chunks=8, tokens=1000))
    assert m.resolve("abc123").name.startswith("Annual")
    assert m.resolve("annual").doc_id == "abc123"
    assert m.resolve("manual.pdf").doc_id == "def456"
    assert m.resolve("pdf") is None  # ambiguous
    assert m.by_sha("ee").doc_id == "def456"
    assert m.total_tokens == 10000
    m2 = Manifest(tmp_path / "m.json")
    assert len(m2) == 2
