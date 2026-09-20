"""The agentic research loop: model ↔ tools until it answers. Parallel tool calls within a turn."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ragnet.agent import prompts
from ragnet.agent.evidence import Evidence
from ragnet.agent.tools import TOOL_SCHEMAS, ToolExecutor, parse_tool_args
from ragnet.config import Settings
from ragnet.llm.client import LLM


@dataclass
class Event:
    type: str  # status | token | answer | sources | verification | route | error
    data: Any = None


@dataclass
class AgentResult:
    answer: str
    evidence: Evidence
    steps: int
    trace: list[dict] = field(default_factory=list)


def _describe(name: str, args: dict) -> str:
    if name == "search":
        return f"search: {args.get('query', '')!s}" + (f" (in {args['doc']})" if args.get("doc") else "")
    if name == "grep":
        return f"grep: {args.get('term', '')!r}"
    if name == "read_page":
        return f"read page {args.get('page')} of {args.get('doc')}"
    return name


async def run_agent(
    settings: Settings,
    llm: LLM,
    tools: ToolExecutor,
    question: str,
    history: list[dict],
) -> AsyncIterator[Event]:
    """Yields status events while researching, then a final ('answer', AgentResult)."""
    messages: list[dict] = [{"role": "system", "content": prompts.AGENT_SYSTEM}]
    messages += history
    messages.append({"role": "user", "content": question})

    steps = 0
    answer = ""
    while True:
        steps += 1
        force_answer = steps >= settings.agent_max_steps
        if force_answer:
            messages.append({"role": "user", "content": "You have used all your research steps. Write the best final answer you can from the evidence gathered so far, citing ids. State clearly anything you could not find."})
        msg = await llm.chat(messages, tools=None if force_answer else TOOL_SCHEMAS)
        tool_calls = list(msg.tool_calls or [])

        if not tool_calls:
            answer = (msg.content or "").strip()
            if not answer and not force_answer:
                messages.append({"role": "assistant", "content": ""})
                messages.append({"role": "user", "content": "Please write the final answer now, citing evidence ids."})
                continue
            break

        # echo the assistant turn (content + tool calls) back into the transcript
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                    for tc in tool_calls
                ],
            }
        )
        parsed = [(tc, parse_tool_args(tc.function.arguments)) for tc in tool_calls]
        for tc, args in parsed:
            yield Event("status", _describe(tc.function.name, args))
        results = await asyncio.gather(*(tools.run(tc.function.name, args) for tc, args in parsed))
        for (tc, _args), result in zip(parsed, results):
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})

    yield Event("answer", AgentResult(answer=answer, evidence=tools.evidence, steps=steps, trace=tools.calls))
