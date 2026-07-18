#!/usr/bin/env python3
"""Build the preregistered 12-task x 3-seed T4 execution matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import MANIFESTS, TASK_SPECS, write_json


PRIMARY = ("B0", "F00", "F01", "F10", "F11")


def build(data_root: Path, output_root: Path, isolation_mode: str = "process_only_not_claim_grade") -> dict:
    runs = []
    for task in TASK_SPECS:
        for condition in PRIMARY:
            for seed in (17, 43, 79):
                run_id = f"{task['task_id']}__{condition}__s{seed}"
                runs.append(
                    {
                        "run_id": run_id,
                        "task_id": task["task_id"],
                        "task_family": task["family"],
                        "metric": task["metric"],
                        "direction": task["direction"],
                        "condition": condition,
                        "seed": seed,
                        "train_data_path": str(data_root / task["task_id"] / "train"),
                        "prediction_output_path": str(output_root / run_id / "predictions.jsonl"),
                    }
                )
    matrix = {
        "schema": "runforest_composite_micro_matrix_v1",
        "task_count": len(TASK_SPECS),
        "conditions": list(PRIMARY),
        "seeds": [17, 43, 79],
        "run_count": len(runs),
        "budget": {
            "wall_clock_sec": 3600,
            "gpu_count": 1,
            "root_slots": 3,
            "model_call_limit": 20,
            "output_token_limit": 80000,
            "failure_is_worst_valid_score": True,
        },
        "holdout_isolation_mode": isolation_mode,
        "runs": runs,
    }
    write_json(MANIFESTS / "micro_execution_matrix_v1.json", matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/benchmark/data"))
    parser.add_argument("--output-root", type=Path, default=Path("/benchmark/predictions"))
    parser.add_argument("--isolation-mode", choices=("process_only_not_claim_grade", "container_filesystem"), default="process_only_not_claim_grade")
    args = parser.parse_args()
    print(json.dumps(build(args.data_root, args.output_root, args.isolation_mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
