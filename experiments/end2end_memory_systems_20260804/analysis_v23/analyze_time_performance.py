#!/usr/bin/env python3
"""Export internal candidate score versus active time for the 40-cell Pilot.

The terminal score remains the primary outcome.  These curves use RunForest
development metrics only and are explicitly labelled diagnostic.  Both the
search-lineage clock and the all-retained-attempt operational clock are kept,
so an infrastructure/adapter attempt is never silently erased from cost.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping


TERMINAL_MODULE_PATH = Path(__file__).with_name("analyze_composite_terminal.py")
_spec = importlib.util.spec_from_file_location(
    "composite_terminal", TERMINAL_MODULE_PATH
)
terminal = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(terminal)


def _journal_nodes(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("nodes", "journal", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _metric_value(node: Mapping[str, Any]) -> float | None:
    value = node.get("metric")
    if isinstance(value, Mapping):
        value = value.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _observed_epoch(node: Mapping[str, Any]) -> float | None:
    ctime = node.get("ctime")
    if not isinstance(ctime, (int, float)) or isinstance(ctime, bool):
        return None
    ctime = float(ctime)
    if not math.isfinite(ctime):
        return None
    duration = None
    created = node.get("created_time")
    finished = node.get("finish_time")
    if isinstance(created, str) and isinstance(finished, str):
        try:
            duration = (
                datetime.fromisoformat(finished) - datetime.fromisoformat(created)
            ).total_seconds()
        except ValueError:
            duration = None
    if duration is None:
        execution = node.get("exec_time")
        if isinstance(execution, (int, float)) and not isinstance(execution, bool):
            execution = float(execution)
            if math.isfinite(execution):
                duration = execution
    return ctime + max(0.0, float(duration or 0.0))


def _resolve_journal(measurement: Mapping[str, Any], measurement_path: Path) -> Path | None:
    declared = measurement.get("journal_path")
    if isinstance(declared, str) and declared:
        path = Path(declared)
        if path.is_file() and not path.is_symlink():
            return path
    candidates = sorted(
        path
        for path in measurement_path.parent.glob("agent/logs/*/journal.json")
        if path.is_file() and not path.is_symlink()
    )
    if len(candidates) > 1:
        raise ValueError(f"Attempt has multiple journals: {measurement_path.parent}")
    return candidates[0] if candidates else None


def _nonnegative(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None:
        return 0.0
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field} is invalid")
    return value


def build_cell_curve(
    cell: Mapping[str, Any],
    *,
    direction: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    operational_wall_prior = 0.0
    operational_gpu_prior = 0.0
    attempts = sorted(cell.get("attempts") or [], key=lambda row: int(row["attempt"]))
    for attempt in attempts:
        raw_path = attempt.get("measurement_path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        measurement_path = Path(raw_path)
        measurement = terminal.read_object(measurement_path)
        terminal.verify_hash(measurement, "measurement_hash", str(measurement_path))
        local_wall = _nonnegative(measurement, "agent_wall_seconds")
        local_gpu = _nonnegative(measurement, "allocated_gpu_hours")
        cumulative_wall = _nonnegative(
            measurement,
            "cumulative_agent_wall_seconds"
            if measurement.get("cumulative_agent_wall_seconds") is not None
            else "agent_wall_seconds",
        )
        cumulative_gpu = _nonnegative(
            measurement,
            "cumulative_allocated_gpu_hours"
            if measurement.get("cumulative_allocated_gpu_hours") is not None
            else "allocated_gpu_hours",
        )
        search_wall_prior = max(0.0, cumulative_wall - local_wall)
        search_gpu_prior = max(0.0, cumulative_gpu - local_gpu)
        receipt_path = measurement_path.parent / "LAUNCH_RECEIPT.json"
        journal_path = _resolve_journal(measurement, measurement_path)
        if receipt_path.is_file() and journal_path is not None:
            receipt = terminal.read_object(receipt_path)
            terminal.verify_hash(receipt, "receipt_hash", str(receipt_path))
            started_ns = receipt.get("started_at_ns")
            if not isinstance(started_ns, int) or isinstance(started_ns, bool) or started_ns <= 0:
                raise ValueError(f"Invalid launch timestamp: {receipt_path}")
            started = started_ns / 1_000_000_000.0
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            nodes = sorted(
                _journal_nodes(journal),
                key=lambda node: (
                    _observed_epoch(node) or float("inf"),
                    int(node.get("step") or 0),
                ),
            )
            for node in nodes:
                node_id = str(node.get("id") or "")
                if not node_id or node_id in seen_nodes:
                    continue
                score = _metric_value(node)
                observed = _observed_epoch(node)
                if score is None or observed is None or node.get("is_buggy") is True:
                    continue
                active_in_attempt = observed - started
                # A resumed journal contains restored source nodes.  Their wall
                # timestamps predate this attempt and were already counted in
                # the source attempt; never count them twice.
                if active_in_attempt < -1.0:
                    continue
                seen_nodes.add(node_id)
                active_in_attempt = min(local_wall, max(0.0, active_in_attempt))
                gpu_fraction = (active_in_attempt / local_wall) if local_wall else 0.0
                points.append(
                    {
                        "logical_run_id": cell["logical_run_id"],
                        "task_id": cell["task_id"],
                        "system_id": cell["system_id"],
                        "release": cell["release"],
                        "attempt": int(attempt["attempt"]),
                        "node_id": node_id,
                        "step": int(node.get("step") or 0),
                        "stage": str(node.get("stage") or "unknown"),
                        "candidate_internal_metric": score,
                        "search_active_seconds": search_wall_prior + active_in_attempt,
                        "operational_active_seconds": (
                            operational_wall_prior + active_in_attempt
                        ),
                        "search_gpu_hours": search_gpu_prior + local_gpu * gpu_fraction,
                        "operational_gpu_hours": (
                            operational_gpu_prior + local_gpu * gpu_fraction
                        ),
                        "direction": direction,
                        "internal_metric_not_terminal": True,
                        "formal_result_eligible": bool(
                            attempt.get("formal_result_eligible", True)
                        ),
                    }
                )
        operational_wall_prior += local_wall
        operational_gpu_prior += local_gpu

    points.sort(
        key=lambda row: (
            float(row["operational_active_seconds"]),
            int(row["attempt"]),
            int(row["step"]),
            str(row["node_id"]),
        )
    )
    incumbent = None
    formal_incumbent = None
    for row in points:
        score = float(row["candidate_internal_metric"])
        if incumbent is None:
            incumbent = score
        elif direction == "maximize":
            incumbent = max(incumbent, score)
        else:
            incumbent = min(incumbent, score)
        row["best_internal_metric_so_far"] = incumbent
        if row["formal_result_eligible"]:
            if formal_incumbent is None:
                formal_incumbent = score
            elif direction == "maximize":
                formal_incumbent = max(formal_incumbent, score)
            else:
                formal_incumbent = min(formal_incumbent, score)
        row["best_formal_internal_metric_so_far"] = formal_incumbent
    return points


def build_report(
    terminal_summary: Mapping[str, Any],
    attempt_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    terminal.verify_hash(terminal_summary, "summary_hash", "Terminal summary")
    terminal.verify_hash(attempt_inventory, "inventory_hash", "Attempt inventory")
    summary_cells = {
        str(row["logical_run_id"]): row for row in terminal_summary.get("cells") or []
    }
    rows = []
    curves = []
    for cell in attempt_inventory.get("cells") or []:
        logical_run_id = str(cell["logical_run_id"])
        summary_cell = summary_cells.get(logical_run_id)
        if summary_cell is None:
            raise ValueError(f"Inventory cell missing from terminal summary: {logical_run_id}")
        points = build_cell_curve(cell, direction=str(summary_cell["direction"]))
        rows.extend(points)
        curves.append(
            {
                "logical_run_id": logical_run_id,
                "task_id": cell["task_id"],
                "system_id": cell["system_id"],
                "point_count": len(points),
                "formal_point_count": sum(
                    row["formal_result_eligible"] is True for row in points
                ),
                "attempt_count": len(cell.get("attempts") or []),
            }
        )
    report = {
        "schema": "mlevolve_end2end_time_performance_v1",
        "exploratory_pilot": True,
        "seed": 1,
        "metric_scope": "internal_development_metric_not_terminal_score",
        "terminal_summary_hash": terminal_summary["summary_hash"],
        "attempt_inventory_hash": attempt_inventory["inventory_hash"],
        "definitions": {
            "search_active_seconds": (
                "active MLEvolve time along the selected resume lineage"
            ),
            "operational_active_seconds": (
                "active time across every retained attempt, including failed "
                "adapters and retries"
            ),
            "best_internal_metric_so_far": (
                "incumbent internal development metric across every retained "
                "attempt; not the fixed-holdout terminal score"
            ),
            "best_formal_internal_metric_so_far": (
                "incumbent internal development metric from formal-result-eligible "
                "attempts only"
            ),
        },
        "curves": curves,
        "points": rows,
        "report_hash": "",
    }
    report["report_hash"] = terminal.payload_hash(report, "report_hash")
    return report


def write_outputs(root: Path, report: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "time_performance.json").write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "logical_run_id", "task_id", "system_id", "release", "attempt",
        "node_id", "step", "stage", "candidate_internal_metric",
        "best_internal_metric_so_far", "best_formal_internal_metric_so_far",
        "direction", "search_active_seconds",
        "operational_active_seconds", "search_gpu_hours", "operational_gpu_hours",
        "internal_metric_not_terminal", "formal_result_eligible",
    ]
    with (root / "time_performance_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["points"]:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-summary", type=Path, required=True)
    parser.add_argument("--attempt-inventory", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        terminal.read_object(args.terminal_summary),
        terminal.read_object(args.attempt_inventory),
    )
    write_outputs(args.analysis_root, report)
    print(
        json.dumps(
            {
                "curves": len(report["curves"]),
                "points": len(report["points"]),
                "exploratory_pilot": True,
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
