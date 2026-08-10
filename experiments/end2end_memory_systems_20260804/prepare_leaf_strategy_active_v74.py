#!/usr/bin/env python3
"""Build an immutable six-hour v74 Leaf Strategy Smoke release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from prepare_leaf_strategy_active_v73 import (
    extract_overlay,
    freeze,
    make_writable,
    payload_hash,
    read_object,
    sha256_file,
    write_object,
)


def _six_hour_smoke_budget(budget: dict[str, Any]) -> None:
    profile = dict(budget["smoke"])
    profile.update(
        {
            "agent_steps": 8,
            "agent_time_limit_seconds": 21600,
            "cpu_count": 16,
            "execution_timeout_seconds": 3600,
            "finalize_reserve_seconds": 600,
            "gpu_count": 1,
            "initial_drafts": 3,
            "max_replacement_drafts": 0,
            "memory_gib": 64,
            "parallel_search_num": 1,
        }
    )
    budget["smoke"] = profile
    budget["manifest_hash"] = payload_hash(budget, "manifest_hash")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.resolve(strict=True)
    destination = args.destination.resolve()
    overlay = args.overlay.resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable source: {destination}")
    shutil.copytree(base, destination, symlinks=False)
    make_writable(destination)
    extract_overlay(overlay, destination)

    exp_root = destination / "experiments/end2end_memory_systems_20260804"
    source_manifests = exp_root / "manifests_v61"
    manifests = exp_root / "manifests_v74"
    if manifests.exists():
        raise FileExistsError(f"overlay unexpectedly created {manifests}")
    shutil.copytree(source_manifests, manifests)
    make_writable(manifests)

    config_path = exp_root / "systems_v74/dynamic_hybrid.yaml"
    systems_path = manifests / "systems.json"
    systems = read_object(systems_path)
    systems["experimental_axis"] = (
        "Leaf required Strategy with method-complete evidence and staged Atomic actuation"
    )
    systems["systems"] = [
        {
            "config_path": "systems_v74/dynamic_hybrid.yaml",
            "config_sha256": sha256_file(config_path),
            "description": (
                "Dynamic Router plus required active Strategy with Planner feedback "
                "decomposition for Improve and Debug"
            ),
            "kind": "internal_exploratory",
            "label": "S5-v74-active-decomposed",
            "limitation": "single exploratory six-hour Leaf Smoke",
            "primary_reference": None,
            "system_id": "dynamic_hybrid",
        }
    ]
    systems["system_count"] = 1
    systems["manifest_hash"] = payload_hash(systems, "manifest_hash")
    write_object(systems_path, systems)

    budget_path = manifests / "budget.json"
    budget = read_object(budget_path)
    _six_hour_smoke_budget(budget)
    write_object(budget_path, budget)

    replay_source = source_manifests / "leaf_official_replay_targets.json"
    replay_target = manifests / "leaf_official_replay_targets.json"
    shutil.copy2(replay_source, replay_target)

    source_lock_path = manifests / "source_lock.json"
    source_lock = read_object(source_lock_path)
    source_lock["git_head"] = args.git_head
    source_lock["git_dirty"] = False
    source_lock["files"] = []
    source_lock["complete_runtime_file_hash_lock"] = True
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path in {source_lock_path, destination / "SOURCE_FILES.sha256"}:
            continue
        source_lock["files"].append(
            {
                "path": str(path.relative_to(destination)),
                "sha256": sha256_file(path),
            }
        )
    source_lock["manifest_hash"] = payload_hash(source_lock, "manifest_hash")
    write_object(source_lock_path, source_lock)

    smoke_source = source_manifests / "leaf_official_smoke_manifest.json"
    smoke_path = manifests / "leaf_strategy_active_smoke_manifest.json"
    smoke = read_object(smoke_source)
    logical_id = (
        "e2e-smoke-leaf-strategy-active-decomposed-v74__leaf-classification__"
        "dynamic_hybrid__seed-1"
    )
    row = dict(smoke["runs"][0])
    row.update(
        {
            "logical_run_id": logical_id,
            "launch_position": 0,
            "task_launch_position": 0,
            "formal_result_eligible": False,
            "exploratory_pilot": True,
        }
    )
    bindings = dict(smoke["bindings"])
    bindings["systems_manifest_hash"] = systems["manifest_hash"]
    bindings["budget_manifest_hash"] = budget["manifest_hash"]
    bindings["source_lock_manifest_hash"] = source_lock["manifest_hash"]
    row["bindings"] = dict(bindings)
    row["row_hash"] = payload_hash(row, "row_hash")
    smoke.update(
        {
            "release_id": "end2end-leaf-strategy-active-decomposed-v74-smoke",
            "kind": "smoke",
            "run_count": 1,
            "runs": [row],
            "system_ids": ["dynamic_hybrid"],
            "task_ids": ["leaf-classification"],
            "first_parallel_batch": ["dynamic_hybrid"],
            "launch_order_randomization": "single active Dynamic Leaf Smoke",
            "formal_result_eligible": False,
            "exploratory_pilot": True,
            "bindings": bindings,
        }
    )
    smoke["manifest_hash"] = payload_hash(smoke, "manifest_hash")
    write_object(smoke_path, smoke)

    file_count, source_manifest_sha = freeze(destination)
    receipt = {
        "schema": "mlevolve_leaf_strategy_active_source_release_v2",
        "git_head": args.git_head,
        "base_source": str(base),
        "frozen_source": str(destination),
        "source_file_count": file_count,
        "source_manifest_sha256": source_manifest_sha,
        "systems_manifest_sha256": systems["manifest_hash"],
        "budget_manifest_sha256": budget["manifest_hash"],
        "source_lock_manifest_sha256": source_lock["manifest_hash"],
        "smoke_manifest_sha256": smoke["manifest_hash"],
        "smoke_manifest": str(smoke_path),
        "agent_time_limit_seconds": 21600,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
