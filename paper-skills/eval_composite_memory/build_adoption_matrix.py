#!/usr/bin/env python3
"""Freeze all available normal T2 episodes (minimum target: 60)."""

from __future__ import annotations

import json
from collections import Counter

from core import EPISODES, MANIFESTS, read_jsonl, write_json


def build() -> dict:
    queries = [row for row in read_jsonl(EPISODES / "decision_test_v1.jsonl") if row["expected_status"] == "rank_candidates"]
    selected = queries
    matrix = {
        "schema": "runforest_composite_adoption_matrix_v1",
        "episode_count": len(selected),
        "conditions": ["F00", "F01", "F10", "F11"],
        "run_count": len(selected) * 4,
        "episodes": [row["episode_id"] for row in selected],
        "episodes_by_task_family": dict(sorted(Counter(row["task_family"] for row in selected).items())),
        "episodes_by_stage": dict(sorted(Counter(row["stage"] for row in selected).items())),
        "generation_budget": {"attempts": 1, "temperature": 0.0, "output_tokens": 5000},
        "gold_exposed_to_generator": False,
    }
    write_json(MANIFESTS / "adoption_matrix_v1.json", matrix)
    return matrix


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
