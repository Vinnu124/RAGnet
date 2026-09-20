"""Registry of ingested documents (what's in the index, hashes for dedupe, summaries for the agent)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DocRecord:
    doc_id: str
    name: str
    path: str
    sha256: str
    pages: int
    chunks: int
    tokens: int
    summary: str = ""
    outline: list[str] = field(default_factory=list)
    contextual: bool = False


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.docs: dict[str, DocRecord] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.docs = {k: DocRecord(**v) for k, v in raw.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: asdict(v) for k, v in self.docs.items()}, indent=2))

    def clear(self) -> None:
        self.docs = {}
        if self.path.exists():
            self.path.unlink()

    def add(self, rec: DocRecord) -> None:
        self.docs[rec.doc_id] = rec
        self.save()

    def remove(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)
        self.save()

    def get(self, doc_id: str) -> Optional[DocRecord]:
        return self.docs.get(doc_id)

    def by_sha(self, sha256: str) -> Optional[DocRecord]:
        return next((d for d in self.docs.values() if d.sha256 == sha256), None)

    def resolve(self, ref: str) -> Optional[DocRecord]:
        """Find a document by id, exact name, or case-insensitive name substring."""
        if ref in self.docs:
            return self.docs[ref]
        low = ref.lower()
        exact = [d for d in self.docs.values() if d.name.lower() == low]
        if exact:
            return exact[0]
        partial = [d for d in self.docs.values() if low in d.name.lower()]
        return partial[0] if len(partial) == 1 else None

    @property
    def total_tokens(self) -> int:
        return sum(d.tokens for d in self.docs.values())

    def __len__(self) -> int:
        return len(self.docs)

    def __iter__(self):
        return iter(self.docs.values())
