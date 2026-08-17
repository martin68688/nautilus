#!/usr/bin/env python3
"""Live, secret-safe smoke for mlevolve's OpenAI-compatible solver backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from config import _load_cfg
from llm import generate, query
from llm.gemini import FunctionSpec


EXPECTED_MODEL = "gpt-5.6-sol"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    args = _args()
    started = time.time()
    cfg = _load_cfg(use_cli_args=False)
    code = cfg.agent.code
    feedback = cfg.agent.feedback
    strategy_model = cfg.external_skill_memory.memory_strategy_model

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    if code.model != EXPECTED_MODEL or feedback.model != EXPECTED_MODEL:
        raise RuntimeError(f"unexpected solver model: {code.model!r}/{feedback.model!r}")
    if strategy_model != EXPECTED_MODEL:
        raise RuntimeError(f"unexpected Strategy model: {strategy_model!r}")
    if "deepseek" in code.model.lower() or "deepseek" in feedback.model.lower():
        raise RuntimeError("DeepSeek model remained active")

    spec = FunctionSpec(
        name="report_smoke",
        description="Report whether the connectivity smoke passed.",
        json_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )
    structured = query(
        system_message="Use the required function and report success.",
        user_message="The API request reached you successfully.",
        model=feedback.model,
        max_tokens=128,
        func_spec=spec,
        cfg=cfg,
    )
    streamed = generate(
        prompt="Reply with exactly FRAMEWORK_STREAM_OK.",
        cfg=cfg,
        max_tokens=64,
        max_retries=1,
        request_timeout=120,
    ).strip()
    if structured != {"ok": True}:
        raise RuntimeError(f"unexpected structured response: {structured!r}")
    if re.fullmatch(r"FRAMEWORK_STREAM_OK[.!]?", streamed) is None:
        raise RuntimeError(f"unexpected streaming response: {streamed!r}")

    endpoint = urlparse(str(code.base_url))
    receipt = {
        "schema": "mlevolve_openai_compatible_smoke_v1",
        "status": "pass",
        "model": code.model,
        "feedback_model": feedback.model,
        "memory_strategy_model": strategy_model,
        "provider_route": "openai_compatible_chat_completions",
        "endpoint_host": endpoint.hostname,
        "base_url_sha256": _sha256_text(str(code.base_url)),
        "api_key_present": True,
        "api_key_persisted": False,
        "structured_function_call": structured,
        "streaming_generation": streamed,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
