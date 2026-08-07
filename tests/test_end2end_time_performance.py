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
    / "analyze_time_performance.py"
)
SPEC = importlib.util.spec_from_file_location("end2end_time_performance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_attempt(
    root: Path,
    *,
    logical_run_id: str,
    attempt: int,
    started: float,
    local_wall: float,
    cumulative_wall: float,
    local_gpu: float,
    cumulative_gpu: float,
    nodes: list[dict] | None,
) -> dict:
    attempt_root = root / f"attempt-{attempt:03d}"
    attempt_root.mkdir(parents=True)
    receipt = {
        "started_at_ns": int(started * 1_000_000_000),
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = MODULE.terminal.payload_hash(receipt, "receipt_hash")
    (attempt_root / "LAUNCH_RECEIPT.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    journal_path = None
    if nodes is not None:
        journal_path = attempt_root / "agent" / "logs" / "run" / "journal.json"
        journal_path.parent.mkdir(parents=True)
        journal_path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
    measurement = {
        "logical_run_id": logical_run_id,
        "attempt": attempt,
        "agent_wall_seconds": local_wall,
        "cumulative_agent_wall_seconds": cumulative_wall,
        "allocated_gpu_hours": local_gpu,
        "cumulative_allocated_gpu_hours": cumulative_gpu,
        "journal_path": str(journal_path) if journal_path is not None else "",
        "measurement_hash": "",
    }
    measurement["measurement_hash"] = MODULE.terminal.payload_hash(
        measurement, "measurement_hash"
    )
    measurement_path = attempt_root / "MEASUREMENT.json"
    measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
    return {"attempt": attempt, "measurement_path": str(measurement_path)}


def test_curve_deduplicates_restored_nodes_and_keeps_two_time_clocks(tmp_path) -> None:
    logical_run_id = "run__leaf-classification__flat_retrieval__seed-1"
    root = tmp_path / logical_run_id
    original = {
        "id": "node-a",
        "step": 1,
        "stage": "draft",
        "ctime": 1010.0,
        "exec_time": 5.0,
        "metric": {"value": 0.3, "maximize": False},
        "is_buggy": False,
    }
    improved = {
        "id": "node-b",
        "step": 2,
        "stage": "improve",
        "ctime": 3010.0,
        "exec_time": 5.0,
        "metric": 0.2,
        "is_buggy": False,
    }
    attempts = [
        {
            **_write_attempt(
                root,
                logical_run_id=logical_run_id,
                attempt=0,
                started=1000.0,
                local_wall=100.0,
                cumulative_wall=100.0,
                local_gpu=1.0,
                cumulative_gpu=1.0,
                nodes=[original],
            ),
            "formal_result_eligible": True,
        },
        {
            **_write_attempt(
                root,
                logical_run_id=logical_run_id,
                attempt=1,
                started=2000.0,
                local_wall=10.0,
                cumulative_wall=110.0,
                local_gpu=0.1,
                cumulative_gpu=1.1,
                nodes=None,
            ),
            "formal_result_eligible": False,
        },
        {
            **_write_attempt(
                root,
                logical_run_id=logical_run_id,
                attempt=2,
                started=3000.0,
                local_wall=100.0,
                cumulative_wall=200.0,
                local_gpu=1.0,
                cumulative_gpu=2.0,
                nodes=[original, improved],
            ),
            "formal_result_eligible": False,
        },
    ]
    cell = {
        "logical_run_id": logical_run_id,
        "task_id": "leaf-classification",
        "system_id": "flat_retrieval",
        "release": "v21",
        "attempts": attempts,
    }

    points = MODULE.build_cell_curve(cell, direction="minimize")

    assert [row["node_id"] for row in points] == ["node-a", "node-b"]
    assert [row["best_internal_metric_so_far"] for row in points] == [0.3, 0.2]
    assert [row["formal_result_eligible"] for row in points] == [True, False]
    assert [row["best_formal_internal_metric_so_far"] for row in points] == [
        0.3,
        0.3,
    ]
    assert points[1]["search_active_seconds"] == 115.0
    assert points[1]["operational_active_seconds"] == 125.0
    assert points[1]["search_gpu_hours"] == 1.15
    assert points[1]["operational_gpu_hours"] == 1.25


def test_real_archived_journal_can_export_internal_score_curve() -> None:
    attempt = (
        REPO
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "smoke_attempts"
        / "20260806_leaf_layered_recipe_v27_attempt_000"
    )
    cell = {
        "logical_run_id": (
            "e2e-smoke-leaf-layered-recipe-v4__leaf-classification__"
            "dynamic_hybrid__seed-1"
        ),
        "task_id": "leaf-classification",
        "system_id": "dynamic_hybrid",
        "release": "v13-smoke",
        "attempts": [
            {
                "attempt": 0,
                "measurement_path": str(attempt / "MEASUREMENT.json"),
            }
        ],
    }

    points = MODULE.build_cell_curve(cell, direction="minimize")

    assert len(points) == 1
    assert points[0]["node_id"] == "d2fccc688085447c9ad84356deac9194"
    assert points[0]["candidate_internal_metric"] == 0.05326
    assert points[0]["internal_metric_not_terminal"] is True
    assert points[0]["formal_result_eligible"] is True
