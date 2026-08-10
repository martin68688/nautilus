#!/usr/bin/env python3
"""Inspect provider model metadata and optionally establish a context lower bound.

Only synthetic repeated text is sent by the active probe.  API keys, base URLs,
and response bodies are never written to the report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from openai import OpenAI


CONTEXT_KEY_TOKENS = ("context", "window", "token", "input", "max")


def _interesting_metadata(value: Any, prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(token in str(key).lower() for token in CONTEXT_KEY_TOKENS):
                if isinstance(item, (str, int, float, bool)) or item is None:
                    found[path] = item
            found.update(_interesting_metadata(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_interesting_metadata(item, f"{prefix}[{index}]"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-probe", action="store_true")
    parser.add_argument(
        "--probe-words",
        default="32768,65536,98304,126000",
        help="Synthetic repeated-word counts; stop after the first rejection.",
    )
    args = parser.parse_args()

    api_key = os.environ["DEEPSEEK_API_KEY"]
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or None
    model = os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash"
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=300.0)
    report: dict[str, Any] = {
        "schema": "mlevolve_solver_context_probe_v1",
        "model": model,
        "provider_metadata": {},
        "active_probe": [],
        "secrets_recorded": False,
    }
    try:
        models = list(client.models.list().data)
        matched = next((item for item in models if str(item.id) == model), None)
        if matched is not None:
            dumped = matched.model_dump()
            report["provider_metadata"] = {
                "model_id": str(matched.id),
                "owned_by": str(getattr(matched, "owned_by", "") or ""),
                "context_fields": _interesting_metadata(dumped),
            }
        else:
            report["provider_metadata"] = {
                "model_id": model,
                "listed": False,
                "available_model_ids": sorted(str(item.id) for item in models),
            }
    except Exception as exc:
        report["provider_metadata"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    if args.active_probe:
        for words in [int(value) for value in args.probe_words.split(",") if value.strip()]:
            synthetic = "probe " * words
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": synthetic
                            + "\nReturn exactly the single word OK.",
                        }
                    ],
                    max_tokens=2,
                    temperature=0,
                    stream=False,
                )
                usage = getattr(response, "usage", None)
                report["active_probe"].append(
                    {
                        "synthetic_word_count": words,
                        "synthetic_char_count": len(synthetic),
                        "accepted": True,
                        "provider_prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "provider_completion_tokens": getattr(usage, "completion_tokens", None),
                    }
                )
            except Exception as exc:
                report["active_probe"].append(
                    {
                        "synthetic_word_count": words,
                        "synthetic_char_count": len(synthetic),
                        "accepted": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                break

    report["observed_prompt_token_lower_bound"] = max(
        (
            int(item["provider_prompt_tokens"])
            for item in report["active_probe"]
            if item.get("accepted") and item.get("provider_prompt_tokens") is not None
        ),
        default=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": model,
                "context_fields": report.get("provider_metadata", {}).get(
                    "context_fields", {}
                ),
                "observed_prompt_token_lower_bound": report[
                    "observed_prompt_token_lower_bound"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
