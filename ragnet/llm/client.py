"""Thin async wrapper over any OpenAI-compatible chat endpoint (vLLM, Ollama, llama.cpp, TGI)."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from ragnet.config import Settings

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks that some servers leak into content."""
    return _THINK_RE.sub("", text or "").strip()


def parse_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM reply (handles fences, leading prose, trailing junk)."""
    text = strip_thinking(text)
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find the outermost {...} or [...]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output: {text[:200]!r}")


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            max_retries=2,
        )
        self._sem = asyncio.Semaphore(settings.llm_concurrency)

    # ------------------------------------------------------------------ helpers
    def _extra_body(self) -> dict:
        # vLLM honours chat_template_kwargs; other servers ignore unknown fields.
        if self.s.llm_disable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def _kwargs(self, model: Optional[str], temperature: float, max_tokens: Optional[int]) -> dict:
        return dict(
            model=model or self.s.llm_model,
            temperature=temperature,
            max_tokens=max_tokens or self.s.llm_max_tokens,
            extra_body=self._extra_body(),
        )

    # ------------------------------------------------------------------ calls
    async def chat(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        """Non-streaming completion. Returns the raw message object (content + tool_calls)."""
        kwargs = self._kwargs(model, temperature, max_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        async with self._sem:
            resp = await self.client.chat.completions.create(messages=messages, **kwargs)
        msg = resp.choices[0].message
        if msg.content:
            msg.content = strip_thinking(msg.content)
        return msg

    async def text(self, messages: list[dict], *, model: Optional[str] = None, temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        msg = await self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        return msg.content or ""

    async def json(self, messages: list[dict], *, model: Optional[str] = None, max_tokens: Optional[int] = None) -> Any:
        """Ask for JSON; retries once with a stern reminder if the first reply doesn't parse."""
        try:
            return parse_json(await self.text(messages, model=model, max_tokens=max_tokens))
        except ValueError:
            retry = messages + [{"role": "user", "content": "Reply with valid JSON only. No prose, no code fences."}]
            return parse_json(await self.text(retry, model=model, max_tokens=max_tokens))

    async def stream(self, messages: list[dict], *, model: Optional[str] = None, temperature: float = 0.0, max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        """Streaming completion yielding text deltas (thinking blocks filtered out)."""
        kwargs = self._kwargs(model, temperature, max_tokens)
        async with self._sem:
            stream = await self.client.chat.completions.create(messages=messages, stream=True, **kwargs)
            in_think = False
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                # crude streaming filter for leaked <think> tags
                if "<think>" in delta:
                    in_think = True
                    delta = delta.split("<think>")[0]
                if in_think:
                    if "</think>" in delta:
                        in_think = False
                        delta = delta.split("</think>", 1)[1]
                    else:
                        continue
                if delta:
                    yield delta

    async def ping(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
