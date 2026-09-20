"""Post-hoc citation verification: every cited sentence is checked against the chunk it cites."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ragnet.agent import prompts
from ragnet.agent.evidence import Evidence, extract_citations
from ragnet.config import Settings
from ragnet.llm.client import LLM

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\-*•])|\n+")


@dataclass
class Claim:
    id: int
    text: str
    cites: list[int]
    supported: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "cites": self.cites, "supported": self.supported, "note": self.note}


def split_claims(answer: str) -> list[Claim]:
    claims = []
    for piece in _SENT_SPLIT.split(answer or ""):
        piece = piece.strip().lstrip("-*• ").strip()
        cites = extract_citations(piece)
        if cites and len(piece) > 15:
            claims.append(Claim(id=len(claims) + 1, text=piece, cites=cites))
    return claims


class Verifier:
    def __init__(self, settings: Settings, llm: LLM):
        self.s = settings
        self.llm = llm

    async def verify(self, answer: str, evidence: Evidence) -> list[Claim]:
        claims = split_claims(answer)
        if not claims:
            return []
        needed = sorted({n for c in claims for n in c.cites if evidence.get(n)})
        ev_text = "\n\n".join(Evidence.format_one(n, evidence.get(n)) for n in needed)
        claims_text = "\n".join(f"{c.id}. {c.text}" for c in claims)
        try:
            data = await self.llm.json(
                [{"role": "user", "content": prompts.VERIFY.format(evidence=ev_text, claims=claims_text)}],
                model=self.s.router_model_name,
                max_tokens=800,
            )
            by_id = {int(r["id"]): r for r in data.get("results", []) if "id" in r}
            for c in claims:
                r = by_id.get(c.id)
                if r is not None:
                    c.supported = bool(r.get("supported", True))
                    c.note = str(r.get("note", "") or "")
                # citations that don't resolve to real evidence are always unsupported
                if any(evidence.get(n) is None for n in c.cites):
                    c.supported = False
                    c.note = (c.note + " " if c.note else "") + "cites a non-existent evidence id"
        except Exception as e:
            for c in claims:
                c.note = f"verification unavailable: {e}"
        return claims
