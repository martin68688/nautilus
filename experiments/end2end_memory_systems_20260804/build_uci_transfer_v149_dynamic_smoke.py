#!/usr/bin/env python3
"""Freeze v149 projected dynamic retrieval beside preserved v147/v148 runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import build_uci_transfer_v147_smoke as base


RUNTIME = "/workspace/nautilus-exp-end2end-agent-v149-dynamic-transfer-smoke-r1"
DATA_ROOT = (
    "/workspace/experiment-end2end-uci100-dynamic-transfer-v149-r1/"
    "data/public"
)
OUTPUT_ROOT = (
    "/workspace/experiment-end2end-uci100-dynamic-transfer-v149-r1/runs"
)
RELEASE_ID = "uci100-cross-task-dynamic-transfer-gpt56sol-v149-smoke-r1"
EXISTING_POD_NAME = "mlevolve-uci100-gpt56sol-v147-transfer-smoke-r6-dev"
EXISTING_POD_UID = "6ffe637e-d7f7-4b89-8653-127698a4fd97"
SOURCE_DATA_ROOT = (
    "/workspace/experiment-end2end-uci100-architecture-transfer-v148-r1/"
    "data/public"
)
V149_OVERLAYS = (
    "mlevolve/agents/memory/cross_task_dynamic_retrieval.py",
    "tests/test_cross_task_dynamic_retrieval_v149.py",
    "experiments/end2end_memory_systems_20260804/"
    "build_uci_transfer_v147_r7_smoke.py",
    "experiments/end2end_memory_systems_20260804/"
    "build_uci_transfer_v147_r8_smoke.py",
    "experiments/end2end_memory_systems_20260804/"
    "build_uci_transfer_v148_architecture_smoke.py",
    "experiments/end2end_memory_systems_20260804/"
    "build_uci_transfer_v149_dynamic_smoke.py",
    "experiments/end2end_memory_systems_20260804/"
    "systems_v147_transfer_smoke_r7/dynamic_cross_task_transfer.yaml",
    "experiments/end2end_memory_systems_20260804/"
    "systems_v147_transfer_smoke_r8/dynamic_cross_task_transfer.yaml",
    "experiments/end2end_memory_systems_20260804/"
    "systems_v148_architecture_transfer_smoke/dynamic_cross_task_transfer.yaml",
    "experiments/end2end_memory_systems_20260804/"
    "systems_v149_dynamic_transfer_smoke/dynamic_cross_task_transfer.yaml",
)


def configure_base() -> None:
    base.RUNTIME = RUNTIME
    base.DATA_ROOT = DATA_ROOT
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.RELEASE_ID = RELEASE_ID
    base.OVERLAYS = tuple(dict.fromkeys((*base.OVERLAYS, *V149_OVERLAYS)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    configure_base()
    if args.output_runtime.exists():
        raise FileExistsError(f"Refusing to reuse runtime: {args.output_runtime}")
    shutil.copytree(
        args.base_runtime,
        args.output_runtime,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "RELEASE_SOURCE_LOCK.json",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "._*",
        ),
    )
    for relative in base.OVERLAYS:
        base.copy_overlay(args.output_runtime, relative)
    lock = base.source_lock(args.output_runtime)
    receipt = {
        "schema": "uci100_cross_task_dynamic_transfer_v149_build_v1",
        "release_id": RELEASE_ID,
        "git_head": base.git_head(),
        "runtime": RUNTIME,
        "local_runtime": str(args.output_runtime.resolve()),
        "source_lock_hash": lock["manifest_hash"],
        "source_lock_file_count": len(lock["files"]),
        "source_data_root": SOURCE_DATA_ROOT,
        "data_root": DATA_ROOT,
        "output_root": OUTPUT_ROOT,
        "source_task_id": "leaf-classification",
        "target_task_id": "uci-one-hundred-leaves",
        "transfer_contract": {
            "retrieval_route": (
                "irreversible_projection_search_judge_resolver"
            ),
            "projected_levels": ["L1_recipe", "L2_tactic", "L3_repair"],
            "fixed_layer_cardinality": False,
            "maximum_selected_architecture_families": 1,
            "source_scores_allowed": False,
            "source_code_allowed": False,
            "source_artifacts_allowed": False,
            "source_dimensions_allowed": False,
            "exact_task_replay_changed": False,
            "independent_novel_source_memory_allowed": False,
        },
        "draft_roles": ["memory_transfer", "novel_exploration"],
        "agent_steps": 8,
        "entrypoint": "mlevolve/run.py",
        "existing_dev_pod_reused": True,
        "existing_dev_pod_name": EXISTING_POD_NAME,
        "existing_dev_pod_uid": EXISTING_POD_UID,
        "pod_manifest_must_not_be_created": True,
        "preserve_v147_and_v148_workloads_and_outputs": True,
    }
    base.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
