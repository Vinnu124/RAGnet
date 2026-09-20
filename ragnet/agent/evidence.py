"""Evidence registry: every chunk the model sees gets a stable numeric id, so citations like [3] resolve
to exactly one chunk no matter which tool call surfaced it."""
from __future__ import annotations

import re

from ragnet.retrieval.store import Hit

_CITE_RE = re.compile(r"\[(\d{1,3})\]")


def extract_citations(text: str) -> list[int]:
    seen, out = set(), []
    for m in _CITE_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


class Evidence:
    def __init__(self):
        self._by_num: dict[int, Hit] = {}
        self._num_by_id: dict[str, int] = {}

    def add(self, hits: list[Hit]) -> list[tuple[int, Hit]]:
        out = []
        for h in hits:
            n = self._num_by_id.get(h.id)
            if n is None:
                n = len(self._by_num) + 1
                self._by_num[n] = h
                self._num_by_id[h.id] = n
            out.append((n, h))
        return out

    def get(self, n: int) -> Hit | None:
        return self._by_num.get(n)

    def __len__(self) -> int:
        return len(self._by_num)

    @property
    def all(self) -> list[tuple[int, Hit]]:
        return sorted(self._by_num.items())

    @staticmethod
    def format_one(n: int, h: Hit, max_chars: int = 2500) -> str:
        body = h.text if len(h.text) <= max_chars else h.text[:max_chars] + " …"
        head = f"[{n}] {h.locator}"
        if h.context:
            head += f"\n({h.context})"
        return f"{head}\n{body}"

    def format(self, items: list[tuple[int, Hit]]) -> str:
        if not items:
            return "(no results)"
        return "\n\n---\n\n".join(self.format_one(n, h) for n, h in items)

    def cited(self, answer: str) -> list[tuple[int, Hit]]:
        return [(n, self._by_num[n]) for n in extract_citations(answer) if n in self._by_num]

    def sources(self, answer: str) -> list[dict]:
        cited = self.cited(answer)
        items = cited or self.all
        return [{"n": n, **h.to_dict(), "cited": bool(cited)} for n, h in items]
