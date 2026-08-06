#!/usr/bin/env python3
"""Aggregate the 40-cell exploratory Pilot across frozen v21/v22/v23 releases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
V23_MANIFESTS = ROOT / "manifests_v23"
DEFAULT_ROOTS = {
    "v21": Path("/workspace/experiment-end2end-memory-agent-v21/runs"),
    "v22": Path("/workspace/experiment-end2end-memory-agent-v22/runs"),
    "v23": Path("/workspace/experiment-end2end-memory-agent-v23/runs"),
}
LEAF_V22_SYSTEMS = {"sop_only", "dynamic_hybrid"}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    if payload.get(field) != payload_hash(payload, field):
        raise ValueError(f"{label} {field} mismatch")


def official_cells(
    pilot: Mapping[str, Any], roots: Mapping[str, Path]
) -> list[dict[str, Any]]:
    cells = []
    for row in pilot["runs"]:
        task_id = str(row["task_id"])
        system_id = str(row["system_id"])
        if task_id == "leaf-classification":
            release = "v22" if system_id in LEAF_V22_SYSTEMS else "v21"
            logical_run_id = (
                f"e2e-pilot-agentic-three-role-{release}__"
                f"{task_id}__{system_id}__seed-1"
            )
        else:
            release = "v23"
            logical_run_id = str(row["logical_run_id"])
        cells.append(
            {
                "task_id": task_id,
                "system_id": system_id,
                "seed": 1,
                "release": release,
                "logical_run_id": logical_run_id,
                "condition_root": roots[release] / logical_run_id,
            }
        )
    if len(cells) != 40 or len(
        {(row["task_id"], row["system_id"]) for row in cells}
    ) != 40:
        raise ValueError("Composite result plan is not an exact 10×4 matrix")
    return cells


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("attempt-"))
    except ValueError as error:
        raise ValueError(f"Invalid immutable attempt directory: {path}") from error


def load_cell_attempts(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(cell["condition_root"])
    attempts = []
    if not root.is_dir():
        return attempts
    for directory in sorted(root.glob("attempt-*"), key=_attempt_number):
        measurement_path = directory / "MEASUREMENT.json"
        if not measurement_path.is_file() or measurement_path.is_symlink():
            continue
        row = read_object(measurement_path)
        verify_hash(row, "measurement_hash", str(measurement_path))
        if row.get("logical_run_id") != cell["logical_run_id"]:
            raise ValueError(f"Logical run mismatch: {measurement_path}")
        if row.get("task_id") != cell["task_id"]:
            raise ValueError(f"Task mismatch: {measurement_path}")
        if row.get("system_id") != cell["system_id"]:
            raise ValueError(f"System mismatch: {measurement_path}")
        row = dict(row)
        row["_measurement_path"] = str(measurement_path)
        row["_release"] = str(cell["release"])
        attempts.append(row)
    return attempts


def select_outcome(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [row for row in attempts if row.get("completed") is True]
    if len(completed) > 1:
        raise ValueError("A logical condition has multiple completed attempts")
    return completed[0] if completed else (attempts[-1] if attempts else None)


def cumulative_value(row: Mapping[str, Any], cumulative: str, local: str) -> Any:
    value = row.get(cumulative)
    return value if value is not None else row.get(local)


def normalized_delta(score: float, baseline: float, direction: str) -> float:
    raw = score - baseline if direction == "maximize" else baseline - score
    return raw / max(abs(baseline), 1e-12)


def rank_task(rows: list[dict[str, Any]], direction: str) -> dict[str, float]:
    completed = [row for row in rows if row["completed"]]
    ordered = sorted(
        completed,
        key=lambda row: float(row["terminal_score"]),
        reverse=direction == "maximize",
    )
    ranks: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor
        score = float(ordered[cursor]["terminal_score"])
        while end + 1 < len(ordered) and math.isclose(
            float(ordered[end + 1]["terminal_score"]), score
        ):
            end += 1
        rank = (cursor + 1 + end + 1) / 2.0
        for row in ordered[cursor : end + 1]:
            ranks[row["system_id"]] = rank
        cursor = end + 1
    for row in rows:
        ranks.setdefault(row["system_id"], 11.0)
    return ranks


def build_summary(
    cells: list[dict[str, Any]],
    task_directions: Mapping[str, str],
    *,
    allow_incomplete: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = []
    inventory = []
    for cell in cells:
        attempts = load_cell_attempts(cell)
        outcome = select_outcome(attempts)
        inventory.append(
            {
                **{key: value for key, value in cell.items() if key != "condition_root"},
                "condition_root": str(cell["condition_root"]),
                "attempts": [
                    {
                        "attempt": row.get("attempt"),
                        "status": row.get("status"),
                        "failure_class": row.get("failure_class"),
                        "completed": row.get("completed") is True,
                        "measurement_hash": row.get("measurement_hash"),
                        "measurement_path": row.get("_measurement_path"),
                    }
                    for row in attempts
                ],
            }
        )
        if outcome is None:
            if not allow_incomplete:
                raise ValueError(f"Missing terminal outcome: {cell['logical_run_id']}")
            selected.append(
                {
                    **cell,
                    "completed": False,
                    "status": "missing",
                    "failure_class": "missing",
                    "terminal_score": None,
                    "attempt": None,
                }
            )
            continue
        completed = outcome.get("completed") is True
        score = outcome.get("terminal_score")
        if completed and (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"Completed outcome has no finite score: {cell['logical_run_id']}")
        selected.append(
            {
                **cell,
                "attempt": outcome.get("attempt"),
                "completed": completed,
                "status": outcome.get("status"),
                "failure_class": outcome.get("failure_class"),
                "terminal_score": score,
                "time_to_first_valid_seconds": cumulative_value(
                    outcome,
                    "cumulative_time_to_first_valid_seconds",
                    "time_to_first_valid_seconds",
                ),
                "agent_wall_seconds": cumulative_value(
                    outcome, "cumulative_agent_wall_seconds", "agent_wall_seconds"
                ),
                "allocated_gpu_hours": cumulative_value(
                    outcome,
                    "cumulative_allocated_gpu_hours",
                    "allocated_gpu_hours",
                ),
                "llm_token_usage": outcome.get("llm_token_usage"),
                "llm_cost_usd": outcome.get("llm_cost_usd"),
                "selected_candidate_id": outcome.get("selected_candidate_id"),
                "journal_path": outcome.get("journal_path"),
                "terminal_report_sha256": outcome.get("terminal_report_sha256"),
                "measurement_path": outcome.get("_measurement_path"),
            }
        )

    by_cell = {(row["task_id"], row["system_id"]): row for row in selected}
    task_ranks = {}
    for task_id, direction in task_directions.items():
        rows = [row for row in selected if row["task_id"] == task_id]
        task_ranks[task_id] = rank_task(rows, direction)
    terminal_cells = []
    for row in selected:
        baseline = by_cell[(row["task_id"], "no_memory")]
        delta = None
        negative = None
        if baseline["completed"]:
            if row["completed"]:
                delta = normalized_delta(
                    float(row["terminal_score"]),
                    float(baseline["terminal_score"]),
                    task_directions[row["task_id"]],
                )
                negative = delta < 0.0
            else:
                negative = row["system_id"] != "no_memory"
        terminal_cells.append(
            {
                **{key: value for key, value in row.items() if key != "condition_root"},
                "condition_root": str(row["condition_root"]),
                "direction": task_directions[row["task_id"]],
                "normalized_delta_vs_no_memory": delta,
                "negative_transfer": negative,
                "task_rank": task_ranks[row["task_id"]][row["system_id"]],
            }
        )

    systems = []
    system_ids = sorted({row["system_id"] for row in terminal_cells})
    for system_id in system_ids:
        rows = [row for row in terminal_cells if row["system_id"] == system_id]
        observed_negative = [
            row["negative_transfer"]
            for row in rows
            if row["negative_transfer"] is not None
        ]
        systems.append(
            {
                "system_id": system_id,
                "completed": sum(row["completed"] for row in rows),
                "completion_rate": sum(row["completed"] for row in rows) / 4.0,
                "negative_transfer_rate": (
                    sum(value is True for value in observed_negative)
                    / len(observed_negative)
                    if observed_negative
                    else None
                ),
                "mean_task_rank": sum(float(row["task_rank"]) for row in rows) / 4.0,
                "total_allocated_gpu_hours": sum(
                    float(row["allocated_gpu_hours"] or 0.0) for row in rows
                ),
                "total_agent_wall_seconds": sum(
                    float(row["agent_wall_seconds"] or 0.0) for row in rows
                ),
            }
        )
    summary = {
        "schema": "mlevolve_end2end_composite_terminal_summary_v1",
        "exploratory_pilot": True,
        "seed": 1,
        "statistical_significance_claim_allowed": False,
        "expected_cells": 40,
        "observed_terminal_outcomes": sum(
            row["status"] != "missing" for row in terminal_cells
        ),
        "completed_cells": sum(row["completed"] for row in terminal_cells),
        "analysis_order": ["terminal", "mechanism_after_terminal"],
        "cells": terminal_cells,
        "systems": systems,
        "summary_hash": "",
    }
    summary["summary_hash"] = payload_hash(summary, "summary_hash")
    attempt_inventory = {
        "schema": "mlevolve_end2end_composite_attempt_inventory_v1",
        "all_attempts_retained": True,
        "cells": inventory,
        "inventory_hash": "",
    }
    attempt_inventory["inventory_hash"] = payload_hash(
        attempt_inventory, "inventory_hash"
    )
    return summary, attempt_inventory


def write_outputs(root: Path, summary: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "terminal_summary.json").write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "attempt_inventory.json").write_text(
        json.dumps(inventory, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "task_id", "system_id", "release", "attempt", "completed", "status",
        "failure_class", "terminal_score", "direction",
        "normalized_delta_vs_no_memory", "negative_transfer", "task_rank",
        "time_to_first_valid_seconds", "agent_wall_seconds",
        "allocated_gpu_hours", "llm_token_usage", "llm_cost_usd",
    ]
    with (root / "terminal_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["cells"]:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, default=V23_MANIFESTS / "pilot_manifest.json")
    parser.add_argument("--tasks-manifest", type=Path, default=V23_MANIFESTS / "tasks.json")
    parser.add_argument("--v21-root", type=Path, default=DEFAULT_ROOTS["v21"])
    parser.add_argument("--v22-root", type=Path, default=DEFAULT_ROOTS["v22"])
    parser.add_argument("--v23-root", type=Path, default=DEFAULT_ROOTS["v23"])
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    pilot = read_object(args.pilot_manifest)
    verify_hash(pilot, "manifest_hash", "Pilot manifest")
    tasks = read_object(args.tasks_manifest)
    verify_hash(tasks, "manifest_hash", "Tasks manifest")
    directions = {
        str(row["task_id"]): str(row["direction"]) for row in tasks["tasks"]
    }
    cells = official_cells(
        pilot, {"v21": args.v21_root, "v22": args.v22_root, "v23": args.v23_root}
    )
    summary, inventory = build_summary(
        cells, directions, allow_incomplete=args.allow_incomplete
    )
    write_outputs(args.analysis_root, summary, inventory)
    print(
        json.dumps(
            {
                "expected_cells": 40,
                "observed_terminal_outcomes": summary["observed_terminal_outcomes"],
                "completed_cells": summary["completed_cells"],
                "exploratory_pilot": True,
                "summary_hash": summary["summary_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
