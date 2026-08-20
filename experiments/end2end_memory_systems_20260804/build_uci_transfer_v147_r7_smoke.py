#!/usr/bin/env python3
"""Freeze v147-r7 for reuse inside the existing r6 A100 Dev Pod."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import build_uci_transfer_v147_smoke as base


RUNTIME = "/workspace/nautilus-exp-end2end-agent-v147-transfer-smoke-r7"
DATA_ROOT = "/workspace/experiment-end2end-uci100-transfer-v147-r7/data/public"
OUTPUT_ROOT = "/workspace/experiment-end2end-uci100-transfer-v147-r7/runs"
RELEASE_ID = "uci100-same-type-cross-task-transfer-gpt56sol-v147-smoke-r7"
EXISTING_POD_NAME = "mlevolve-uci100-gpt56sol-v147-transfer-smoke-r6-dev"
EXISTING_POD_UID = "6ffe637e-d7f7-4b89-8653-127698a4fd97"
R7_OVERLAYS = (
    "experiments/end2end_memory_systems_20260804/build_uci_transfer_v147_r7_smoke.py",
    "experiments/end2end_memory_systems_20260804/systems_v147_transfer_smoke_r7/dynamic_cross_task_transfer.yaml",
)


def configure_base() -> None:
    base.RUNTIME = RUNTIME
    base.DATA_ROOT = DATA_ROOT
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.RELEASE_ID = RELEASE_ID
    base.OVERLAYS = tuple(dict.fromkeys((*base.OVERLAYS, *R7_OVERLAYS)))


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
            "__pycache__", "*.pyc", "*.pyo", ".DS_Store", "._*"
        ),
    )
    for relative in base.OVERLAYS:
        base.copy_overlay(args.output_runtime, relative)
    lock = base.source_lock(args.output_runtime)
    receipt = {
        "schema": "uci100_cross_task_transfer_v147_reused_devpod_build_v1",
        "release_id": RELEASE_ID,
        "git_head": base.git_head(),
        "runtime": RUNTIME,
        "local_runtime": str(args.output_runtime.resolve()),
        "source_lock_hash": lock["manifest_hash"],
        "source_lock_file_count": len(lock["files"]),
        "data_root": DATA_ROOT,
        "output_root": OUTPUT_ROOT,
        "source_task_id": "leaf-classification",
        "target_task_id": "uci-one-hundred-leaves",
        "draft_roles": ["memory_transfer", "novel_exploration"],
        "agent_steps": 8,
        "existing_dev_pod_reused": True,
        "existing_dev_pod_name": EXISTING_POD_NAME,
        "existing_dev_pod_uid": EXISTING_POD_UID,
        "pod_manifest_must_not_be_created": True,
    }
    base.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
