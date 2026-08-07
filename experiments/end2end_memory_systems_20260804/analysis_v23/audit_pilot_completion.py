#!/usr/bin/env python3
"""Fail closed unless the exploratory 10×4 Pilot has complete evidence."""

from __future__ import annotations

import argparse
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


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _existing_journals(cell: Mapping[str, Any]) -> list[Path]:
    values = cell.get("retained_journal_paths") or [cell.get("journal_path")]
    return [
        Path(str(value))
        for value in values
        if value and Path(str(value)).is_file() and not Path(str(value)).is_symlink()
    ]


def audit(
    terminal_summary: Mapping[str, Any],
    attempt_inventory: Mapping[str, Any],
    mechanism_summary: Mapping[str, Any],
) -> dict[str, Any]:
    terminal.verify_hash(terminal_summary, "summary_hash", "Terminal summary")
    terminal.verify_hash(attempt_inventory, "inventory_hash", "Attempt inventory")
    terminal.verify_hash(mechanism_summary, "summary_hash", "Mechanism summary")

    cells = [dict(row) for row in terminal_summary.get("cells") or []]
    inventory_cells = [
        dict(row) for row in attempt_inventory.get("cells") or []
    ]
    mechanism_rows = [
        dict(row) for row in mechanism_summary.get("runs") or []
    ]
    cell_ids = [str(row.get("logical_run_id") or "") for row in cells]
    inventory_by_id = {
        str(row.get("logical_run_id") or ""): row for row in inventory_cells
    }
    mechanism_by_id = {
        str(row.get("logical_run_id") or ""): row for row in mechanism_rows
    }
    condition_pairs = {
        (str(row.get("task_id") or ""), str(row.get("system_id") or ""))
        for row in cells
    }

    attempt_files_valid = True
    every_cell_has_attempt = True
    for logical_run_id in cell_ids:
        attempts = inventory_by_id.get(logical_run_id, {}).get("attempts") or []
        every_cell_has_attempt &= bool(attempts)
        for attempt in attempts:
            path_value = attempt.get("measurement_path")
            path = Path(str(path_value or ""))
            if not path.is_file() or path.is_symlink():
                attempt_files_valid = False
                continue
            try:
                measurement = terminal.read_object(path)
                terminal.verify_hash(
                    measurement, "measurement_hash", str(path)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                attempt_files_valid = False

    terminal_scores_consistent = True
    cost_fields_valid = True
    all_journals_present = True
    for cell in cells:
        completed = cell.get("completed") is True
        score = cell.get("terminal_score")
        terminal_scores_consistent &= (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            if completed
            else score is None
        )
        cost_fields_valid &= all(
            _finite_nonnegative(cell.get(field, 0.0))
            for field in (
                "agent_wall_seconds",
                "allocated_gpu_hours",
                "retained_attempt_agent_wall_seconds",
                "retained_attempt_gpu_hours",
                "retry_overhead_agent_wall_seconds",
                "retry_overhead_gpu_hours",
            )
        )
        all_journals_present &= bool(_existing_journals(cell))

    memory_on_trace_complete = True
    no_memory_is_empty = True
    unobserved_activation_is_not_zero = True
    for cell in cells:
        row = mechanism_by_id.get(str(cell.get("logical_run_id") or ""), {})
        if cell.get("system_id") == "no_memory":
            no_memory_is_empty &= (
                int(row.get("routing_routes") or 0) > 0
                and int(row.get("raw_candidates") or 0) == 0
                and int(row.get("prompt_visible") or 0) == 0
            )
        else:
            memory_on_trace_complete &= (
                int(row.get("routing_routes") or 0) > 0
                and int(row.get("raw_candidates") or 0) > 0
                and int(row.get("prompt_visible") or 0) > 0
            )
        if int(row.get("plan_covered") or 0) == 0:
            unobserved_activation_is_not_zero &= (
                row.get("static_adoption_rate") is None
                and row.get("runtime_activation_rate") is None
            )

    checks = {
        "exploratory_seed_one_only": (
            terminal_summary.get("exploratory_pilot") is True
            and terminal_summary.get("seed") == 1
            and terminal_summary.get("statistical_significance_claim_allowed")
            is False
        ),
        "exact_10_by_4_matrix": (
            terminal_summary.get("expected_cells") == 40
            and len(cells) == 40
            and len(condition_pairs) == 40
            and len({row["task_id"] for row in cells}) == 4
            and len({row["system_id"] for row in cells}) == 10
        ),
        "all_terminal_outcomes_observed": (
            terminal_summary.get("observed_terminal_outcomes") == 40
            and all(row.get("status") != "missing" for row in cells)
        ),
        "terminal_scores_consistent": terminal_scores_consistent,
        "attempt_inventory_covers_all_cells": (
            attempt_inventory.get("all_attempts_retained") is True
            and len(inventory_cells) == 40
            and set(inventory_by_id) == set(cell_ids)
            and every_cell_has_attempt
        ),
        "all_attempt_measurements_exist_and_hash": attempt_files_valid,
        "cost_fields_nonnegative": cost_fields_valid,
        "all_cells_have_retained_journal": all_journals_present,
        "mechanism_bound_to_terminal_summary": (
            mechanism_summary.get("terminal_summary_hash")
            == terminal_summary.get("summary_hash")
        ),
        "mechanism_covers_all_cells": (
            mechanism_summary.get("observed_terminal_outcomes") == 40
            and len(mechanism_rows) == 40
            and set(mechanism_by_id) == set(cell_ids)
        ),
        "memory_on_has_real_retrieval_and_prompt_trace": (
            memory_on_trace_complete
        ),
        "no_memory_has_zero_external_memory": no_memory_is_empty,
        "unobserved_activation_is_not_zero": (
            unobserved_activation_is_not_zero
        ),
    }
    report = {
        "schema": "mlevolve_end2end_pilot_completion_audit_v1",
        "exploratory_pilot": True,
        "seed": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "terminal_summary_hash": terminal_summary["summary_hash"],
        "attempt_inventory_hash": attempt_inventory["inventory_hash"],
        "mechanism_summary_hash": mechanism_summary["summary_hash"],
        "audit_hash": "",
    }
    report["audit_hash"] = terminal.payload_hash(report, "audit_hash")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-summary", type=Path, required=True)
    parser.add_argument("--attempt-inventory", type=Path, required=True)
    parser.add_argument("--mechanism-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        terminal.read_object(args.terminal_summary),
        terminal.read_object(args.attempt_inventory),
        terminal.read_object(args.mechanism_summary),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
