"""Anthropic Messages-API backend (query + generate).

Speaks the Anthropic Messages protocol so it can target any Anthropic-compatible
endpoint — notably Zhipu's GLM Coding Plan endpoint at
``https://open.bigmodel.cn/api/anthropic`` (models: glm-4.5 / glm-4.6 / glm-5 / glm-5.2).

This mirrors ``openai.py`` exactly in signature and return contract so the
provider dispatch in ``llm/__init__.py`` can route ``glm-*`` models here with no
changes to any caller:

    query(...)  -> (output, req_time, in_tok, out_tok, info)
    generate(...) -> str

Key protocol differences handled here vs the OpenAI backend:
  * the system message is a top-level ``system=`` param, not a message in the list;
  * ``max_tokens`` is required (defaults to 16384);
  * function calling uses ``tools=[{name, description, input_schema}]`` +
    ``tool_choice={"type": "tool", "name": ...}``; the SDK returns a parsed
    ``input`` dict on the ``tool_use`` content block (no manual JSON parsing);
  * JSON-schema structured output is requested via a system instruction +
    post-extraction (the OpenAI ``response_format`` knob has no Anthropic
    equivalent on this compatibility endpoint).
"""

import json
import logging
import time
from typing import Any

import anthropic

from config import Config
from .gemini import FunctionSpec, compile_prompt_to_md
from .model_profiles import get_profile
# Reuse the generic message builders from the OpenAI backend (they are
# provider-agnostic; the only OpenAI-specific branch there is a no-op for glm).
from .openai import _prompt_to_messages, _stage_config_for_model

logger = logging.getLogger("MLEvolve")

OutputType = str | dict


def _extract_text(content_blocks: list) -> str:
    """Concatenate all ``text`` content blocks into a single string."""
    parts = []
    for block in content_blocks or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def _extract_tool_use(content_blocks: list, expected_name: str) -> dict:
    """Find the first ``tool_use`` block and return its parsed ``input`` dict.

    Anthropic models often emit a short text preamble before the tool call; we
    scan all blocks and ignore non-tool content.
    """
    for block in content_blocks or []:
        if getattr(block, "type", None) == "tool_use":
            if getattr(block, "name", None) != expected_name:
                raise ValueError(
                    f"Function name mismatch: expected {expected_name}, "
                    f"got {block.name}"
                )
            inp = getattr(block, "input", None)
            if isinstance(inp, str):
                inp = json.loads(inp or "{}")
            return inp or {}
    raise ValueError("Expected function call (tool_use block), got none")


def _split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """Pull a leading ``system`` message out of the OpenAI-style message list.

    Anthropic takes the system prompt as a separate top-level param.
    """
    system_text = None
    rest: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            # If multiple system messages appear, concatenate them.
            system_text = m["content"] if system_text is None else f"{system_text}\n\n{m['content']}"
        else:
            rest.append(m)
    return system_text, rest


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Anthropic-compatible query (Messages API, optional function calling).

    Same return shape as ``openai.query`` / ``gemini.query``.
    """
    if cfg is None:
        raise ValueError("cfg is required for Anthropic backend")
    filtered = {k: v for k, v in model_kwargs.items() if v is not None}
    model = filtered.get("model", "")
    stage = _stage_config_for_model(cfg, model)
    client = anthropic.Anthropic(
        api_key=stage.api_key,
        base_url=stage.base_url or None,
        timeout=1200.0,
    )

    # Anthropic requires at least one user/assistant message; system is separate.
    messages: list[dict[str, str]] = []
    if user_message:
        messages.append({"role": "user", "content": user_message})
    if not messages:
        # system-only query: synthesize a minimal user turn.
        messages.append({"role": "user", "content": "(proceed)"})
    if not (system_message or user_message):
        raise ValueError("Either system_message or user_message must be provided")

    # Function calling (structured output) is done in non-thinking mode.
    use_thinking = func_spec is None
    profile = get_profile(model, use_thinking=use_thinking)

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": filtered.get("max_tokens", 16384),
        "temperature": profile.get("temperature", filtered.get("temperature", 1.0)),
    }
    if system_message:
        params["system"] = system_message
    if "top_p" in profile:
        params["top_p"] = profile["top_p"]
    if func_spec is not None:
        params["tools"] = [
            {
                "name": func_spec.name,
                "description": func_spec.description,
                "input_schema": func_spec.json_schema,
            }
        ]
        params["tool_choice"] = {"type": "tool", "name": func_spec.name}

    t0 = time.time()
    logger.info(f"Querying Anthropic-compatible API with model: {model}")
    try:
        resp = client.messages.create(**params)
    except Exception as e:
        logger.error(f"Error calling Anthropic-compatible API: {e}")
        raise
    req_time = time.time() - t0

    if getattr(resp, "stop_reason", None) == "max_tokens":
        logger.warning(
            f"Response truncated by max_tokens ({params['max_tokens']}), consider increasing it"
        )

    if func_spec is None:
        output = _extract_text(resp.content)
        # Defensively strip any inline <think>...</think> block.
        if "</think>" in output:
            output = output[output.find("</think>") + 8:]
        logger.info(f"Anthropic response: {output}", extra={"verbose": True})
    else:
        output = _extract_tool_use(resp.content, func_spec.name)
        logger.info(f"Anthropic function call response: {output}", extra={"verbose": True})

    in_tok = getattr(resp.usage, "input_tokens", 0) or 0
    out_tok = getattr(resp.usage, "output_tokens", 0) or 0
    info = {
        "model": getattr(resp, "model", model),
        "created": int(time.time()),
    }
    return output, req_time, in_tok, out_tok, info


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
) -> str:
    """Text generation via the Anthropic Messages API.

    Mirrors ``openai.generate``: converts the prompt to chat messages, streams/
    accumulates the text, strips ``<think>`` blocks, retries on failure.
    """
    stage = cfg.agent.code
    model = stage.model
    messages = _prompt_to_messages(prompt, model=model)
    system_text, msgs = _split_system(messages)
    if not msgs:
        msgs.append({"role": "user", "content": "(proceed)"})

    client = anthropic.Anthropic(
        api_key=stage.api_key,
        base_url=stage.base_url or None,
        timeout=1200.0,
    )

    use_thinking = json_schema is None
    profile = get_profile(model, use_thinking=use_thinking)

    params: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens if max_tokens is not None else 16384,
        "temperature": profile.get("temperature", temperature if temperature is not None else 1.0),
    }
    if "top_p" in profile:
        params["top_p"] = profile["top_p"]
    if stop_tokens:
        params["stop_sequences"] = stop_tokens

    # JSON-schema structured output: the Anthropic compatibility endpoint has no
    # OpenAI-style response_format, so instruct in the system prompt and rely on
    # the model + downstream post-processing. Kept distinct from system_text.
    if json_schema is not None:
        json_instr = (
            "Respond with ONLY valid JSON (no markdown fences, no prose) "
            "matching this schema:\n" + json.dumps(json_schema)
        )
        params["system"] = f"{system_text}\n\n{json_instr}" if system_text else json_instr
    elif system_text:
        params["system"] = system_text

    logger.info(f"generate messages: {len(msgs)} turns", extra={"verbose": True})
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**params)
            full_text = _extract_text(resp.content)
            if "</think>" in full_text:
                full_text = full_text[full_text.find("</think>") + 8:]
            logger.info(f"generate response: {full_text}", extra={"verbose": True})
            return full_text
        except Exception as e:
            logger.warning(f"generate failed, retrying {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                logger.error("generate retry limit reached")
                raise
            time.sleep(retry_delay)
    return ""
