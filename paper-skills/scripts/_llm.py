"""Shared LLM client for the paper-skills scripts — GLM via the Anthropic-compatible endpoint.

All paper-skills plugins used to call the OpenAI SDK against DeepSeek. They now go
through this helper, which speaks the Anthropic Messages protocol against Zhipu's
GLM Coding Plan endpoint (``https://open.bigmodel.cn/api/anthropic``), reachable
with a normal Coding Plan subscription (no pay-as-you-go balance required).

Configuration (read from ``paper-skills/.env`` via python-dotenv, or the process env):
    GLM_API_KEY   — Zhipu API key (the ``id.secret`` string)
    GLM_BASE_URL  — defaults to https://open.bigmodel.cn/api/anthropic
    GLM_MODEL     — defaults to glm-5.2

Three entry points cover every call shape the scripts use:
    chat(...)          — single-turn text completion (forms 1 & 2)
    run_tool_loop(...) — agentic multi-turn function-calling loop (plugin_a2)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import anthropic
from dotenv import load_dotenv

# Load paper-skills/.env (non-secret config) + mlevolve/.env (GLM creds, gitignored).
load_dotenv(Path(__file__).resolve().parent / ".." / ".env")
load_dotenv(Path(__file__).resolve().parents[2] / "mlevolve" / ".env")

DEFAULT_BASE_URL = os.getenv("GLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://open.bigmodel.cn/api/anthropic"
DEFAULT_MODEL = os.getenv("GLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "glm-5.2"


def _api_key() -> str:
    key = os.getenv("GLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "GLM_API_KEY not set (put it in paper-skills/.env or export it)."
        )
    return key


def get_client() -> anthropic.Anthropic:
    """A fresh Anthropic-compatible client. Cheap to construct; safe per-call."""
    return anthropic.Anthropic(api_key=_api_key(), base_url=DEFAULT_BASE_URL, timeout=1200.0)


def _extract_text(content_blocks: list) -> str:
    parts = []
    for block in content_blocks or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    text = "".join(parts)
    # Defensively strip any inline <think>...</think> block some models emit.
    if "</think>" in text:
        text = text[text.find("</think>") + 8:]
    return text


def chat(
    user: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    max_tokens: int = 4096,
    stop_sequences: list[str] | None = None,
) -> str:
    """Single-turn text completion. Returns the assistant's text.

    ``temperature=None`` lets the endpoint use its default (1.0); pass ``0`` for
    deterministic extraction/judgement calls.
    """
    client = get_client()
    params: dict = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    if system is not None:
        params["system"] = system
    if temperature is not None:
        params["temperature"] = temperature
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    resp = client.messages.create(**params)
    return _extract_text(resp.content)


def chat_json(
    user: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict | list:
    """Single-turn completion parsed as JSON (strips markdown fences)."""
    raw = chat(user, system=system, model=model, temperature=temperature, max_tokens=max_tokens)
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(raw)


def run_tool_loop(
    system: str,
    user_msg: str,
    tools: list[dict],
    tool_fns: dict[str, Callable[[dict], object]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    max_tokens: int = 8192,
    max_iters: int = 50,
    on_tool_call: Callable[[str, list], None] | None = None,
) -> str:
    """Agentic multi-turn function-calling loop.

    ``tools`` is the OpenAI-shaped tool list used by the existing plugins:
        [{"type":"function","function":{"name","description","parameters"}}]
    ``tool_fns`` maps tool name -> callable(args_dict) -> result (str or object).
    The loop runs until the model stops requesting tools, then returns its final text.
    """
    client = get_client()
    anthropic_tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]

    messages: list[dict] = [{"role": "user", "content": user_msg}]
    resp = None
    for _ in range(max_iters):
        resp = client.messages.create(
            model=model,
            system=system,
            messages=messages,
            tools=anthropic_tools,
            tool_choice={"type": "auto"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Replay the assistant turn (content blocks) verbatim, then answer tools.
        messages.append({"role": "assistant", "content": resp.content})

        if getattr(resp, "stop_reason", None) != "tool_use":
            return _extract_text(resp.content)

        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            args = block.input if isinstance(block.input, dict) else json.loads(block.input or "{}")
            if on_tool_call is not None:
                on_tool_call(block.name, list(args.keys()))
            result = tool_fns[block.name](args)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": str(result)[:64000]}
            )
        messages.append({"role": "user", "content": tool_results})

    return _extract_text(resp.content if resp is not None else [])
