#!/usr/bin/env python3
"""Freeze per-task B0 median/IQR from a completed dev-only T4 pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from core import MANIFESTS, read_jsonl, sha256_file, write_json


def freeze(path: Path) -> dict:
    rows = read_jsonl(path)
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("split") == "dev"
            and row.get("condition") == "B0"
            and row.get("analysis_eligible") is True
            and isinstance(row.get("metric"), (int, float))
        ):
            values[str(row["task_id"])].append(float(row["metric"]))
    tasks = {}
    for task_id, task_values in sorted(values.items()):
        if len(task_values) < 3:
            raise ValueError(f"need at least 3 dev B0 seeds for {task_id}")
        iqr = float(np.subtract(*np.percentile(task_values, [75, 25])))
        if iqr <= 1e-12:
            raise ValueError(f"dev B0 IQR is non-positive for {task_id}")
        tasks[task_id] = {"median": float(np.median(task_values)), "iqr": iqr, "seed_count": len(task_values)}
    artifact = {
        "schema": "runforest_composite_baseline_reference_v1",
        "split": "dev",
        "frozen": True,
        "source_receipts_sha256": sha256_file(path),
        "tasks": tasks,
    }
    write_json(MANIFESTS / "baseline_reference_v1.json", artifact)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-receipts", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.dev_receipts), ensure_ascii=False, indent=2))
