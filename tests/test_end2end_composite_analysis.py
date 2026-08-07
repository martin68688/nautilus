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


def _write_measurement(
    root: Path,
    cell: dict,
    score: float | None,
    *,
    cumulative: float,
    attempt_number: int = 1,
    completed: bool = True,
    status: str = "scored_terminal_result",
    failure_class: str = "none",
    agent_wall_seconds: float = 10.0,
    cumulative_agent_wall_seconds: float | None = None,
    allocated_gpu_hours: float = 1.0,
    cumulative_allocated_gpu_hours: float | None = None,
) -> None:
    attempt = root / cell["logical_run_id"] / f"attempt-{attempt_number:03d}"
    attempt.mkdir(parents=True)
    payload = {
        "logical_run_id": cell["logical_run_id"],
        "task_id": cell["task_id"],
        "system_id": cell["system_id"],
        "attempt": attempt_number,
        "status": status,
        "failure_class": failure_class,
        "completed": completed,
        "terminal_score": score,
        "time_to_first_valid_seconds": 5.0,
        "cumulative_time_to_first_valid_seconds": cumulative,
        "agent_wall_seconds": agent_wall_seconds,
        "cumulative_agent_wall_seconds": (
            cumulative + agent_wall_seconds
            if cumulative_agent_wall_seconds is None
            else cumulative_agent_wall_seconds
        ),
        "allocated_gpu_hours": allocated_gpu_hours,
        "cumulative_allocated_gpu_hours": (
            2.5
            if cumulative_allocated_gpu_hours is None
            else cumulative_allocated_gpu_hours
        ),
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


def test_flat_resume_selects_completed_attempt_and_retains_failed_attempts(tmp_path) -> None:
    pilot = MODULE.read_object(MODULE.V23_MANIFESTS / "pilot_manifest.json")
    roots = {name: tmp_path / name for name in ("v21", "v22", "v23")}
    flat = next(
        row
        for row in MODULE.official_cells(pilot, roots)
        if row["task_id"] == "leaf-classification"
        and row["system_id"] == "flat_retrieval"
    )
    assert flat["release"] == "v21"
    _write_measurement(
        roots["v21"],
        flat,
        None,
        cumulative=100.0,
        attempt_number=0,
        completed=False,
        status="retained_infrastructure_or_timeout_failure",
        failure_class="infrastructure",
        agent_wall_seconds=100.0,
        cumulative_agent_wall_seconds=100.0,
        allocated_gpu_hours=1.0,
        cumulative_allocated_gpu_hours=1.0,
    )
    _write_measurement(
        roots["v21"],
        flat,
        None,
        cumulative=110.0,
        attempt_number=1,
        completed=False,
        status="retained_adapter_failure",
        failure_class="agent",
        agent_wall_seconds=10.0,
        cumulative_agent_wall_seconds=110.0,
        allocated_gpu_hours=0.1,
        cumulative_allocated_gpu_hours=1.1,
    )
    _write_measurement(
        roots["v21"],
        flat,
        0.031,
        cumulative=220.0,
        attempt_number=2,
        agent_wall_seconds=120.0,
        cumulative_agent_wall_seconds=220.0,
        allocated_gpu_hours=1.2,
        cumulative_allocated_gpu_hours=2.2,
    )

    attempts = MODULE.load_cell_attempts(flat)
    assert [row["attempt"] for row in attempts] == [0, 1, 2]
    selected = MODULE.select_outcome(attempts)
    assert selected is not None
    assert selected["attempt"] == 2
    assert selected["terminal_score"] == 0.031

    tasks = MODULE.read_object(MODULE.V23_MANIFESTS / "tasks.json")
    directions = {row["task_id"]: row["direction"] for row in tasks["tasks"]}
    summary, _ = MODULE.build_summary(
        MODULE.official_cells(pilot, roots), directions, allow_incomplete=True
    )
    flat_cell = next(
        row
        for row in summary["cells"]
        if row["task_id"] == "leaf-classification"
        and row["system_id"] == "flat_retrieval"
    )
    assert flat_cell["attempt_count"] == 3
    assert flat_cell["failed_attempt_count"] == 2
    assert flat_cell["agent_wall_seconds"] == 220.0
    assert flat_cell["retained_attempt_agent_wall_seconds"] == 230.0
    assert flat_cell["retry_overhead_agent_wall_seconds"] == 10.0
    assert flat_cell["allocated_gpu_hours"] == 2.2
    assert flat_cell["retained_attempt_gpu_hours"] == 2.3
    assert abs(flat_cell["retry_overhead_gpu_hours"] - 0.1) < 1e-12


def test_allow_incomplete_materializes_zero_cost_missing_cells(tmp_path) -> None:
    pilot = MODULE.read_object(MODULE.V23_MANIFESTS / "pilot_manifest.json")
    roots = {name: tmp_path / name for name in ("v21", "v22", "v23")}
    tasks = MODULE.read_object(MODULE.V23_MANIFESTS / "tasks.json")
    directions = {row["task_id"]: row["direction"] for row in tasks["tasks"]}
    summary, inventory = MODULE.build_summary(
        MODULE.official_cells(pilot, roots), directions, allow_incomplete=True
    )

    assert summary["observed_terminal_outcomes"] == 0
    assert summary["completed_cells"] == 0
    assert len(summary["cells"]) == 40
    assert len(inventory["cells"]) == 40
    assert all(row["status"] == "missing" for row in summary["cells"])
    assert all(row["allocated_gpu_hours"] == 0.0 for row in summary["cells"])
    assert all(
        row["retained_attempt_gpu_hours"] == 0.0 for row in summary["cells"]
    )


def test_dynamic_official_cell_ignores_v21_diagnostic_resume(tmp_path) -> None:
    pilot = MODULE.read_object(MODULE.V23_MANIFESTS / "pilot_manifest.json")
    roots = {name: tmp_path / name for name in ("v21", "v22", "v23")}
    dynamic = next(
        row
        for row in MODULE.official_cells(pilot, roots)
        if row["task_id"] == "leaf-classification"
        and row["system_id"] == "dynamic_hybrid"
    )
    assert dynamic["release"] == "v22"
    _write_measurement(
        roots["v22"],
        dynamic,
        None,
        cumulative=21600.0,
        completed=False,
        status="retained_agent_partial",
        failure_class="agent",
    )

    diagnostic = {
        **dynamic,
        "release": "v21",
        "logical_run_id": (
            "e2e-pilot-agentic-three-role-v21__leaf-classification__"
            "dynamic_hybrid__seed-1"
        ),
        "condition_root": roots["v21"]
        / (
            "e2e-pilot-agentic-three-role-v21__leaf-classification__"
            "dynamic_hybrid__seed-1"
        ),
    }
    _write_measurement(
        roots["v21"],
        diagnostic,
        0.011208,
        cumulative=22000.0,
        attempt_number=2,
    )

    attempts = MODULE.load_cell_attempts(dynamic)
    assert len(attempts) == 1
    assert attempts[0]["logical_run_id"].endswith(
        "v22__leaf-classification__dynamic_hybrid__seed-1"
    )
    assert MODULE.select_outcome(attempts)["terminal_score"] is None


def test_resume_manifest_snapshots_match_publication_receipts() -> None:
    experiment = SCRIPT.parents[1]
    releases = (
        ("v23", "20260807_v21_resume_adapter_v23.json"),
        ("v24", "20260807_v21_resume_adapter_v24.json"),
    )
    for release, receipt_name in releases:
        manifests = experiment / f"manifests_resume_{release}"
        receipt = MODULE.read_object(
            experiment / "infrastructure_attempts" / receipt_name
        )
        pilot = MODULE.read_object(manifests / "pilot_manifest.json")
        source_lock = MODULE.read_object(manifests / "source_lock.json")
        MODULE.verify_hash(pilot, "manifest_hash", f"{release} Pilot manifest")
        MODULE.verify_hash(
            source_lock, "manifest_hash", f"{release} source-lock manifest"
        )
        assert pilot["manifest_hash"] == receipt["pilot_manifest_hash"]
        assert (
            source_lock["manifest_hash"]
            == receipt["source_lock_manifest_hash"]
        )
        assert len(pilot["runs"]) == 40
        assert len(
            {(row["task_id"], row["system_id"]) for row in pilot["runs"]}
        ) == 40

    v23 = MODULE.read_object(
        experiment / "manifests_resume_v23" / "pilot_manifest.json"
    )
    expected = {
        17: "rcr_router_style_port",
        18: "runforest_only",
        19: "macla_style_port",
    }
    assert {index: v23["runs"][index]["system_id"] for index in expected} == expected


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


def test_mechanism_reads_all_attempt_journals_and_latest_node_wins(tmp_path) -> None:
    def route(stage: str) -> dict:
        return {
            "schema": "mlevolve_memory_routing_trace_v1",
            "stage_route": {"stage": stage},
            "raw_candidates": [],
            "suppressed_candidates": [],
            "final_prompt_candidate_ids": [],
        }

    old_journal = tmp_path / "attempt-000-journal.json"
    new_journal = tmp_path / "attempt-002-journal.json"
    old_journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "shared-node",
                        "stage": "draft",
                        "memory_routing_trace": route("draft"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    new_journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "shared-node",
                        "stage": "improve",
                        "memory_routing_trace": route("improve"),
                    },
                    {
                        "id": "new-node",
                        "stage": "debug",
                        "memory_routing_trace": route("debug"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    outcome = {
        "logical_run_id": "run-1",
        "task_id": "leaf-classification",
        "system_id": "dynamic_hybrid",
        "journal_path": str(new_journal),
        "retained_journal_paths": [str(old_journal), str(new_journal)],
    }

    report = MECHANISM.base_analysis.mechanism_summary(
        [outcome], manifests=MECHANISM.MANIFESTS
    )
    row = report["runs"][0]
    assert row["routing_routes"] == 2
    assert "draft" not in row["by_stage"]
    assert row["by_stage"]["improve"]["routing_routes"] == 1
    assert row["by_stage"]["debug"]["routing_routes"] == 1
