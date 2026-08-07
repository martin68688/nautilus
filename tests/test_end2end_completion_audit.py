from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "analysis_v23"
    / "audit_pilot_completion.py"
)
SPEC = importlib.util.spec_from_file_location("end2end_completion_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _fixtures(tmp_path: Path) -> tuple[dict, dict, dict]:
    pilot = MODULE.terminal.read_object(
        SCRIPT.parents[1] / "manifests_v23" / "pilot_manifest.json"
    )
    cells = []
    inventories = []
    mechanisms = []
    for index, run in enumerate(pilot["runs"]):
        run_root = tmp_path / str(index)
        run_root.mkdir()
        journal = run_root / "journal.json"
        journal.write_text('{"nodes": []}', encoding="utf-8")
        measurement = {
            "logical_run_id": run["logical_run_id"],
            "attempt": 0,
            "measurement_hash": "",
        }
        measurement["measurement_hash"] = MODULE.terminal.payload_hash(
            measurement, "measurement_hash"
        )
        measurement_path = run_root / "MEASUREMENT.json"
        measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
        cells.append(
            {
                "logical_run_id": run["logical_run_id"],
                "task_id": run["task_id"],
                "system_id": run["system_id"],
                "status": "scored_terminal_result",
                "completed": True,
                "terminal_score": 1.0,
                "agent_wall_seconds": 10.0,
                "allocated_gpu_hours": 10.0 / 3600.0,
                "retained_attempt_agent_wall_seconds": 10.0,
                "retained_attempt_gpu_hours": 10.0 / 3600.0,
                "retry_overhead_agent_wall_seconds": 0.0,
                "retry_overhead_gpu_hours": 0.0,
                "journal_path": str(journal),
                "formal_journal_paths": [str(journal)],
                "retained_journal_paths": [str(journal)],
            }
        )
        inventories.append(
            {
                "logical_run_id": run["logical_run_id"],
                "attempts": [
                    {
                        "attempt": 0,
                        "measurement_path": str(measurement_path),
                        "formal_result_eligible": True,
                    }
                ],
            }
        )
        no_memory = run["system_id"] == "no_memory"
        mechanisms.append(
            {
                "logical_run_id": run["logical_run_id"],
                "routing_routes": 1,
                "raw_candidates": 0 if no_memory else 6,
                "prompt_visible": 0 if no_memory else 6,
                "plan_covered": 0,
                "static_adoption_rate": None,
                "runtime_activation_rate": None,
            }
        )
    terminal_summary = {
        "schema": "mlevolve_end2end_composite_terminal_summary_v1",
        "exploratory_pilot": True,
        "seed": 1,
        "statistical_significance_claim_allowed": False,
        "expected_cells": 40,
        "observed_terminal_outcomes": 40,
        "cells": cells,
        "summary_hash": "",
    }
    terminal_summary["summary_hash"] = MODULE.terminal.payload_hash(
        terminal_summary, "summary_hash"
    )
    inventory = {
        "schema": "mlevolve_end2end_composite_attempt_inventory_v1",
        "all_attempts_retained": True,
        "cells": inventories,
        "inventory_hash": "",
    }
    inventory["inventory_hash"] = MODULE.terminal.payload_hash(
        inventory, "inventory_hash"
    )
    mechanism = {
        "schema": "mlevolve_end2end_composite_mechanism_summary_v1",
        "terminal_summary_hash": terminal_summary["summary_hash"],
        "observed_terminal_outcomes": 40,
        "runs": mechanisms,
        "retained_operational_analysis": {
            "runs": mechanisms,
        },
        "summary_hash": "",
    }
    mechanism["summary_hash"] = MODULE.terminal.payload_hash(
        mechanism, "summary_hash"
    )
    return terminal_summary, inventory, mechanism


def test_completion_audit_passes_only_with_full_40_cell_evidence(tmp_path) -> None:
    terminal_summary, inventory, mechanism = _fixtures(tmp_path)

    report = MODULE.audit(terminal_summary, inventory, mechanism)

    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["audit_hash"] == MODULE.terminal.payload_hash(
        report, "audit_hash"
    )


def test_completion_audit_rejects_memory_on_without_prompt_exposure(tmp_path) -> None:
    terminal_summary, inventory, mechanism = _fixtures(tmp_path)
    row = next(
        row
        for row in mechanism["runs"]
        if row["logical_run_id"].endswith("dynamic_hybrid__seed-1")
    )
    row["prompt_visible"] = 0
    mechanism["summary_hash"] = MODULE.terminal.payload_hash(
        mechanism, "summary_hash"
    )

    report = MODULE.audit(terminal_summary, inventory, mechanism)

    assert report["passed"] is False
    assert report["checks"]["memory_on_has_real_retrieval_and_prompt_trace"] is False
