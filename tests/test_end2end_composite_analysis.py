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
    / "analyze_composite_terminal.py"
)
SPEC = importlib.util.spec_from_file_location("composite_terminal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

MECHANISM_SCRIPT = SCRIPT.with_name("analyze_composite_mechanism.py")
MECHANISM_SPEC = importlib.util.spec_from_file_location(
    "composite_mechanism", MECHANISM_SCRIPT
)
MECHANISM = importlib.util.module_from_spec(MECHANISM_SPEC)
assert MECHANISM_SPEC.loader is not None
MECHANISM_SPEC.loader.exec_module(MECHANISM)


def _write_measurement(root: Path, cell: dict, score: float, *, cumulative: float) -> None:
    attempt = root / cell["logical_run_id"] / "attempt-001"
    attempt.mkdir(parents=True)
    payload = {
        "logical_run_id": cell["logical_run_id"],
        "task_id": cell["task_id"],
        "system_id": cell["system_id"],
        "attempt": 1,
        "status": "scored_terminal_result",
        "failure_class": "none",
        "completed": True,
        "terminal_score": score,
        "time_to_first_valid_seconds": 5.0,
        "cumulative_time_to_first_valid_seconds": cumulative,
        "agent_wall_seconds": 10.0,
        "cumulative_agent_wall_seconds": cumulative + 10.0,
        "allocated_gpu_hours": 1.0,
        "cumulative_allocated_gpu_hours": 2.5,
        "llm_token_usage": None,
        "llm_cost_usd": None,
        "measurement_hash": "",
    }
    payload["measurement_hash"] = MODULE.payload_hash(payload, "measurement_hash")
    (attempt / "MEASUREMENT.json").write_text(json.dumps(payload), encoding="utf-8")


def test_composite_plan_is_40_cells_and_uses_cumulative_resume_metrics(tmp_path) -> None:
    pilot = MODULE.read_object(MODULE.V23_MANIFESTS / "pilot_manifest.json")
    roots = {name: tmp_path / name for name in ("v21", "v22", "v23")}
    cells = MODULE.official_cells(pilot, roots)
    assert len(cells) == 40
    assert next(
        row for row in cells
        if row["task_id"] == "leaf-classification"
        and row["system_id"] == "dynamic_hybrid"
    )["release"] == "v22"
    assert next(
        row for row in cells
        if row["task_id"] == "leaf-classification"
        and row["system_id"] == "flat_retrieval"
    )["release"] == "v21"

    for index, cell in enumerate(cells):
        score = 1.0 + index / 100.0
        _write_measurement(
            roots[cell["release"]], cell, score, cumulative=100.0 + index
        )
    tasks = MODULE.read_object(MODULE.V23_MANIFESTS / "tasks.json")
    directions = {row["task_id"]: row["direction"] for row in tasks["tasks"]}
    summary, inventory = MODULE.build_summary(
        cells, directions, allow_incomplete=False
    )

    assert summary["expected_cells"] == 40
    assert summary["observed_terminal_outcomes"] == 40
    assert summary["completed_cells"] == 40
    assert summary["statistical_significance_claim_allowed"] is False
    assert len(inventory["cells"]) == 40
    first = summary["cells"][0]
    assert first["time_to_first_valid_seconds"] == 100.0
    assert first["agent_wall_seconds"] == 110.0
    assert first["allocated_gpu_hours"] == 2.5
    assert summary["summary_hash"] == MODULE.payload_hash(summary, "summary_hash")


def test_mechanism_aggregation_keeps_stage_system_and_activation_counts() -> None:
    row = {
        "logical_run_id": "run-1",
        "task_id": "leaf-classification",
        "system_id": "dynamic_hybrid",
        **MECHANISM._empty_counts(),
        "routing_routes": 2,
        "raw_candidates": 12,
        "prompt_visible": 6,
        "suppressed": 6,
        "static_adopted": 3,
        "runtime_activated": 2,
        "adopted": 1,
        "partially_adopted": 1,
        "plan_covered": 6,
        "by_stage": {
            "debug": {
                **MECHANISM._empty_counts(),
                "routing_routes": 2,
                "raw_candidates": 12,
                "prompt_visible": 6,
                "suppressed": 6,
                "runtime_activated": 2,
                "plan_covered": 6,
            }
        },
    }
    aggregate = MECHANISM.aggregate_runs([row])
    assert aggregate["totals"]["suppression_rate"] == 0.5
    assert aggregate["totals"]["static_adoption_rate"] == 0.5
    assert aggregate["totals"]["runtime_activation_rate"] == 2 / 6
    assert aggregate["by_system"]["dynamic_hybrid"]["runs"] == 1
    assert aggregate["by_stage"]["debug"]["runtime_activated"] == 2


def test_missing_runtime_probe_is_unobserved_not_zero_activation() -> None:
    counts = MECHANISM._empty_counts()
    counts["prompt_visible"] = 6
    counts["raw_candidates"] = 12
    rates = MECHANISM._rates(counts)
    assert rates["adoption_observable"] is False
    assert rates["static_adoption_rate"] is None
    assert rates["runtime_activation_rate"] is None
    assert rates["prompt_visible_without_adoption_plan"] == 6
