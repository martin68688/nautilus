#!/usr/bin/env python3
"""Run the SOP protocol prompt-exposure probe through the project DeepSeek API."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from run_sop_protocol_prompt_exposure_test import (
    CONDITIONS,
    DEBUG_CASES,
    RANK_CASES,
    SYSTEM,
    debug_prompt,
    extract_choice,
    rank_prompt,
    summarize,
)


def request_one(
    client: OpenAI,
    *,
    model: str,
    row: dict[str, Any],
    temperature: float,
    retries: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    error = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": str(row["prompt"])},
                ],
                temperature=temperature,
                max_tokens=64,
                stream=False,
            )
            message = response.choices[0].message
            output = str(message.content or "").strip()
            choice = extract_choice(output)
            return {
                **row,
                "output": output,
                "choice": choice,
                "correct": choice == row["correct_choice"],
                "forbidden_choice_selected": choice == row["forbidden_choice"],
                "latency_sec": time.perf_counter() - started,
                "attempts": attempt + 1,
                "error": "",
            }
        except Exception as exc:  # API failures are recorded in the receipts.
            error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    return {
        **row,
        "output": "",
        "choice": None,
        "correct": False,
        "forbidden_choice_selected": False,
        "latency_sec": time.perf_counter() - started,
        "attempts": retries,
        "error": error,
    }


def build_rows(repetitions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for condition in CONDITIONS:
            for case_index in range(len(RANK_CASES)):
                prompt, correct, forbidden = rank_prompt(case_index, condition)
                rows.append(
                    {
                        "track": "rank_contamination",
                        "condition": condition,
                        "case_index": case_index,
                        "repetition": repetition,
                        "prompt": prompt,
                        "correct_choice": correct,
                        "forbidden_choice": forbidden,
                    }
                )
            for case_index in range(len(DEBUG_CASES)):
                prompt, correct, forbidden = debug_prompt(case_index, condition)
                rows.append(
                    {
                        "track": "debug_retention",
                        "condition": condition,
                        "case_index": case_index,
                        "repetition": repetition,
                        "prompt": prompt,
                        "correct_choice": correct,
                        "forbidden_choice": forbidden,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", ""))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "")
    if not api_key or not base_url or not args.model:
        raise RuntimeError("DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, and DEEPSEEK_MODEL are required")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=0)
    pending = build_rows(args.repetitions)
    started = time.time()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                request_one,
                client,
                model=args.model,
                row=row,
                temperature=args.temperature,
            )
            for row in pending
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["repetition"], row["track"], row["condition"], row["case_index"]))

    report = {
        "schema": "sop_protocol_prompt_exposure_deepseek_v1",
        "model": args.model,
        "base_url_host": base_url.split("//")[-1].split("/")[0],
        "repetitions": args.repetitions,
        "temperature": args.temperature,
        "workers": args.workers,
        "environment": {
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "elapsed_sec": time.time() - started,
        "request_count": len(rows),
        "error_count": sum(bool(row["error"]) for row in rows),
        "summary": summarize(rows),
        "rows": rows,
        "limitations": [
            "Synthetic multiple-choice prompts do not establish MLEvolve task-level score improvements.",
            "This tests the configured DeepSeek endpoint only; cross-model generalization remains unproven.",
            "Repeated API calls are samples, not independent MLE tasks.",
            "Dual-view and claim-clean prompts are identical at high-risk pre-prompt activation; they differ in retained audit storage.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "model": args.model,
        "request_count": len(rows),
        "error_count": report["error_count"],
        "elapsed_sec": report["elapsed_sec"],
        "summary": report["summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
