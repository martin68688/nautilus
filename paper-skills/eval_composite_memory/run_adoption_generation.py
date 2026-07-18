#!/usr/bin/env python3
"""Call an Agent adapter for frozen T2 prompts and save auditable candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from core import ARTIFACTS, EPISODES, MANIFESTS, REPORTS, read_json, read_jsonl, sha256_value, write_jsonl


def run(
    adapter: list[str], timeout_sec: int, max_episodes: int | None = None,
    conditions: set[str] | None = None, output_path: Path | None = None,
    split: str = "dev", confirm_frozen_test: bool = False,
) -> dict:
    if split == "test" and not confirm_frozen_test:
        raise ValueError("test generation requires --confirm-frozen-test; use dev for pilots")
    matrix = read_json(MANIFESTS / "adoption_matrix_v1.json")
    split_rows = read_jsonl(EPISODES / f"decision_{split}_v1.jsonl")
    episodes = {row["episode_id"]: row for row in split_rows}
    graph = read_json(ARTIFACTS / "memory_snapshot_graph_v1.json")
    nodes = {str(row["id"]): row for row in graph["nodes"]}
    offline = read_jsonl(REPORTS / "offline_test_receipts_v1.jsonl")
    rankings = {(row["episode_id"], row["condition"]): row["ranking"] for row in offline}
    outputs = []
    available_ids = (
        matrix["episodes"]
        if split == "test"
        else [row["episode_id"] for row in split_rows if row["expected_status"] == "rank_candidates"]
    )
    episode_ids = available_ids[:max_episodes] if max_episodes is not None else available_ids
    for episode_id in episode_ids:
        query = episodes[episode_id]
        for condition in matrix["conditions"]:
            if conditions is not None and condition not in conditions:
                continue
            refs = rankings.get((episode_id, condition), [])[:4]
            prompt = {
                "schema": "runforest_composite_agent_request_v1",
                "episode_id": episode_id,
                "condition": condition,
                "task": {key: query[key] for key in ("task_id", "task_family", "stage", "query_text")},
                "memories": [
                    {"id": ref, "title": nodes.get(ref, {}).get("title"), "action": nodes.get(ref, {}).get("action")}
                    for ref in refs
                ],
                "gold_exposed": False,
            }
            started = time.time()
            try:
                proc = subprocess.run(adapter, input=json.dumps(prompt), text=True, capture_output=True, timeout=timeout_sec, check=False)
                payload = json.loads(proc.stdout) if proc.returncode == 0 else {}
                status = "completed" if proc.returncode == 0 else "failed"
                error = "" if proc.returncode == 0 else f"exit_{proc.returncode}:{proc.stderr[-500:]}"
            except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
                payload, status, error = {}, "failed", f"{type(exc).__name__}:{exc}"
            outputs.append(
                {
                    "episode_id": episode_id,
                    "condition": condition,
                    "seed": 0,
                    "selected_memory_ids": refs,
                    "prompt_sha256": sha256_value(prompt),
                    "code": str(payload.get("code") or ""),
                    "adoption_outcome": (
                        payload.get("adoption_outcome", "not_reported")
                        if status == "completed"
                        else "generation_failed"
                    ),
                    "model": payload.get("model"),
                    "input_tokens": payload.get("input_tokens"),
                    "output_tokens": payload.get("output_tokens"),
                    "status": status,
                    "error": error,
                    "latency_sec": time.time() - started,
                    "mock": False,
                }
            )
    path = output_path or (REPORTS / "adoption_candidates_v1.jsonl")
    write_jsonl(path, outputs)
    return {"run_count": len(outputs), "completed_count": sum(row["status"] == "completed" for row in outputs), "path": str(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", nargs="+", required=True)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--conditions", nargs="+", choices=["F00", "F01", "F10", "F11"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--confirm-frozen-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.adapter,
        args.timeout_sec,
        args.max_episodes,
        set(args.conditions) if args.conditions else None,
        args.output,
        args.split,
        args.confirm_frozen_test,
    ), ensure_ascii=False, indent=2))
