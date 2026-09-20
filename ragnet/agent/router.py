"""One cheap LLM call: condense the follow-up into a standalone question and pick the execution mode."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ragnet.agent import prompts
from ragnet.config import Settings
from ragnet.llm.client import LLM
from ragnet.manifest import Manifest

Mode = Literal["chat", "lookup", "research"]


@dataclass
class Route:
    question: str
    mode: Mode


def format_history(history: list[dict], max_chars: int = 4000) -> str:
    lines = [f"{m['role']}: {m['content']}" for m in history]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text or "(none)"


class Router:
    def __init__(self, settings: Settings, llm: LLM, manifest: Manifest):
        self.s = settings
        self.llm = llm
        self.manifest = manifest

    async def route(self, question: str, history: list[dict]) -> Route:
        docs = ", ".join(d.name for d in self.manifest) or "(none)"
        prompt = prompts.ROUTER.format(history=format_history(history), question=question, documents=docs)
        try:
            data = await self.llm.json([{"role": "user", "content": prompt}], model=self.s.router_model_name, max_tokens=300)
            mode = str(data.get("mode", "lookup")).lower()
            if mode not in ("chat", "lookup", "research"):
                mode = "lookup"
            q = str(data.get("question") or question).strip() or question
            return Route(question=q, mode=mode)  # type: ignore[arg-type]
        except Exception:
            return Route(question=question, mode="lookup")
