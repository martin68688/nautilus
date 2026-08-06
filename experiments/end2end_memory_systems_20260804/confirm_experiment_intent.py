#!/usr/bin/env python3
"""Print the one lightweight human-facing confirmation before End2End launch.

This command reads only committed local configuration.  It does not open a
Memory Bundle, hash data, compile candidate code, start a subprocess, call an
LLM, contact Kubernetes, or create a run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MANIFESTS = ROOT / "manifests"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFESTS / "pilot_manifest.json",
    )
    args = parser.parse_args()

    manifest = read_object(args.manifest.resolve(strict=True))
    systems = read_object(MANIFESTS / "systems.json")
    tasks = read_object(MANIFESTS / "tasks.json")
    budget = read_object(MANIFESTS / "budget.json")
    memory = read_object(MANIFESTS / "memory_bundles.json")
    replay_targets = read_object(
        REPO
        / "paper-skills"
        / "eval_skill_memory"
        / "clean_replay_targets.json"
    )

    rows = list(manifest.get("runs") or [])
    all_system_ids = [str(row["system_id"]) for row in systems.get("systems") or []]
    all_task_ids = [str(row["task_id"]) for row in tasks.get("tasks") or []]
    system_ids = [str(value) for value in manifest.get("system_ids") or []]
    task_ids = [str(value) for value in manifest.get("task_ids") or []]
    seeds = sorted({int(row["seed"]) for row in rows})
    expected = {
        (task_id, system_id, seed)
        for task_id in task_ids
        for system_id in system_ids
        for seed in seeds
    }
    observed = {
        (str(row["task_id"]), str(row["system_id"]), int(row["seed"]))
        for row in rows
    }

    kind = str(manifest.get("kind") or "")
    require(kind in {"smoke", "pilot"}, "intent confirmation expects Smoke or Pilot")
    require(set(system_ids).issubset(all_system_ids), "Manifest contains an unknown system")
    require(set(task_ids).issubset(all_task_ids), "Manifest contains an unknown task")
    require(seeds == [1], "Experiment seed must be exactly 1")
    require(len(rows) == len(expected) and observed == expected, "Manifest is not its declared Cartesian matrix")
    if kind == "pilot":
        require(len(task_ids) == 4, "Pilot must contain exactly four tasks")
        require(len(system_ids) == 10, "Pilot must contain exactly ten systems")
        require(len(rows) == 40, "Pilot must be the full 10×4×1 matrix")
        require(
            memory.get("excluded_run_ids") == [],
            "Pilot must allow same-task historical memory",
        )
        task_bundles = dict(memory.get("task_bundles") or {})
        require(set(task_bundles) == set(task_ids), "Pilot Bundle task set mismatch")
        require(
            all(
                task_bundles[task_id].get("same_task_history_enabled") is True
                and str(task_bundles[task_id].get("same_task_best_node_id") or "")
                for task_id in task_ids
            ),
            "Every Pilot task must freeze a same-task clean best record",
        )
        pilot_job = yaml.safe_load(
            (ROOT / "jobs" / "pilot-all-40-indexed-job.yaml").read_text(
                encoding="utf-8"
            )
        )
        pilot_spec = pilot_job["spec"]
        pilot_args = pilot_spec["template"]["spec"]["containers"][0]["args"]
        require(pilot_spec["completions"] == 40, "Pilot Job must cover 40 indices")
        require(pilot_spec["parallelism"] == 1, "Pilot Job must use one A100 at a time")
        require("--resume" in pilot_args, "Pilot Job must enable condition resume")
    else:
        require(manifest.get("formal_result_eligible") is False, "Smoke must not be a formal result")

    sys.path.insert(0, str(REPO / "mlevolve"))
    sys.path.insert(0, str(REPO))
    try:
        from config import _load_cfg

        dynamic = _load_cfg(ROOT / "systems" / "dynamic_hybrid.yaml", use_cli_args=False)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)

    roles = list(dynamic.agent.draft_role_policy.roles)
    require(
        roles == ["coldstart_baseline", "memory_reproduction", "novel_exploration"],
        "Dynamic Hybrid must use the frozen three roles",
    )
    require(dynamic.evaluation_authority.mode == "off", "Host authority must be off")
    require(not dynamic.agent.protocol_preflight.enabled, "Protocol preflight must be off")
    require(not dynamic.agent.protocol_repair.enabled, "Protocol repair must be off")
    require(not dynamic.agent.check_data_leakage, "Leakage audit must be off")
    require(not dynamic.adoption_verifier.enabled, "Adoption verifier must be off")
    require(not dynamic.prospective_audit.enabled, "Prospective audit must be off")
    require(
        not dynamic.fixed_holdout.preflight_validate_train_view,
        "Per-run train-view integrity scan must be off",
    )
    require(
        dynamic.external_skill_memory.experiment_r_agentic_retrieval_enabled,
        "Dynamic Hybrid Retrieval Agent must be on",
    )
    replay_by_task = {
        str(row.get("task_id") or ""): row
        for row in replay_targets.get("targets") or []
    }
    require(
        all(
            task_id in replay_by_task
            and replay_by_task[task_id].get("audit_status") == "verified_clean"
            and str(replay_by_task[task_id].get("code_sha256") or "")
            for task_id in task_ids
        ),
        "Every Pilot task must freeze one clean exact Replay implementation",
    )

    runtime = dict(budget["runtime"])
    run_budget = dict(budget[kind])
    require(runtime["gpu_resource_key"] == "nvidia.com/a100", "GPU request must be A100")
    require(run_budget["gpu_count"] == 1, "Each run must request one GPU")
    require(run_budget["parallel_search_num"] == 1, "Each run must use one candidate worker")
    require(run_budget["cpu_count"] == 16, "Each run must request 16 CPU")
    require(run_budget["memory_gib"] == 64, "Each run must request 64 GiB")

    per_task_order = {
        task_id: [
            str(row["system_id"])
            for row in sorted(
                (item for item in rows if item["task_id"] == task_id),
                key=lambda item: int(item["task_launch_position"]),
            )
        ]
        for task_id in task_ids
    }
    report = {
        "schema": "mlevolve_end2end_intent_confirmation_v2",
        "status": "ready_for_user_confirmation",
        "launches_training": False,
        "experiment": {
            "kind": kind,
            "runs": len(rows),
            "systems": system_ids,
            "tasks": task_ids,
            "seeds": seeds,
            "exploratory_only": True,
            "full_pilot_matrix": kind == "pilot",
            "system_order_by_task": per_task_order,
        },
        "dynamic_hybrid": {
            "included_in_this_manifest": "dynamic_hybrid" in system_ids,
            "roles": roles,
            "retrieval_agent": True,
            "same_task_best_policy": (
                "memory_reproduction loads the task-specific frozen exact target; "
                "Leaf is bound to the best sealed terminal implementation"
            ),
            "exact_replay_target_by_task": {
                task_id: {
                    "graph_node_id": replay_by_task[task_id].get(
                        "graph_node_id"
                    ),
                    "run_id": replay_by_task[task_id].get("run_id"),
                    "original_node_id": replay_by_task[task_id].get(
                        "original_node_id"
                    ),
                    "historical_metric": replay_by_task[task_id].get(
                        "historical_metric"
                    ),
                    "metric_status": replay_by_task[task_id].get(
                        "metric_status"
                    ),
                    "method_family": replay_by_task[task_id].get(
                        "method_family"
                    ),
                    "code_sha256": replay_by_task[task_id].get("code_sha256"),
                }
                for task_id in task_ids
            },
            "prompt_selection": (
                "memory_reproduction loads the frozen exact implementation; "
                "novel_exploration uses Retrieval Agent selection; any invalid-Agent "
                "fallback is explicit and retained"
            ),
        },
        "memory_bundle": {
            "binding_path": memory["production_binding_path"],
            "binding_sha256": memory["production_binding_sha256"],
            "same_task_history_enabled": memory.get("excluded_run_ids") == [],
            "same_task_best_node_by_task": {
                task_id: memory["task_bundles"][task_id].get(
                    "same_task_best_node_id"
                )
                for task_id in task_ids
            },
        },
        "resources_per_run": {
            "gpu": "1×A100",
            "cpu": 16,
            "memory_gib": 64,
            "parallel_runs_per_job": 1,
        },
        "formal_job": (
            {
                "name": "mlevolve-e2e-agentic-pilot-all-40-v22",
                "completions": 40,
                "parallelism": 1,
                "condition_level_resume": True,
                "epoch_checkpoint_guaranteed": False,
            }
            if kind == "pilot"
            else None
        ),
        "runtime_checks": {
            "host_protocol": False,
            "host_receipts": False,
            "data_tree_hash": False,
            "bundle_artifact_traversal": False,
            "source_lock_gate": False,
            "adoption_gate": False,
            "prospective_audit": False,
        },
        "still_required_to_get_a_score": [
            "candidate program can actually run",
            "submission has the task-required columns and rows",
            "terminal evaluator can compute the frozen metric",
        ],
    }
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
