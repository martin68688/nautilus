#!/usr/bin/env python3
"""Freeze endpoint-wire-compatible r2 runtimes without mutating clip-r1."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import build_uci_transfer_3h_clip_r1 as base


R2_OVERLAYS = (
    *base.SHARED_OVERLAYS,
    "mlevolve/llm/gemini.py",
    "mlevolve/llm/openai.py",
    "tests/test_gpt_openai_compatible_config.py",
    "tests/test_nrp_clip_tool_choice_compat.py",
    "experiments/end2end_memory_systems_20260804/build_uci_transfer_3h_clip_r2.py",
)

VARIANTS = deepcopy(base.VARIANTS)
for name, variant in VARIANTS.items():
    variant["release_id"] = variant["release_id"].replace("clip-r1", "clip-r2")
    variant["runtime"] = variant["runtime"].replace("clip-r1", "clip-r2")
    variant["output_root"] = variant["output_root"].replace("clip-r1", "clip-r2")
    variant["data_root"] = base.VARIANTS[name]["data_root"]
    if name == "v147":
        variant["config"] = (
            "experiments/end2end_memory_systems_20260804/"
            "systems_v147_transfer_3h_clip_r2/dynamic_cross_task_transfer.yaml"
        )
    else:
        variant["config"] = (
            "experiments/end2end_memory_systems_20260804/"
            "systems_v149_dynamic_transfer_3h_clip_r2/"
            "dynamic_cross_task_transfer.yaml"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    overlays = tuple(
        dict.fromkeys((*R2_OVERLAYS, variant["config"], variant["pod_manifest"]))
    )
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
    for relative in overlays:
        base._copy_overlay(args.output_runtime, relative)

    r1_variants = base.VARIANTS
    try:
        base.VARIANTS = VARIANTS
        lock = base._source_lock(
            args.output_runtime,
            variant=args.variant,
            overlays=overlays,
        )
    finally:
        base.VARIANTS = r1_variants

    receipt = {
        "schema": "uci100_cross_task_transfer_three_hour_clip_r2",
        "variant": args.variant,
        "release_id": variant["release_id"],
        "git_head": base._git_head(),
        "runtime": variant["runtime"],
        "local_runtime": str(args.output_runtime.resolve()),
        "source_lock_hash": lock["manifest_hash"],
        "source_lock_file_count": len(lock["files"]),
        "data_root": variant["data_root"],
        "output_root": variant["output_root"],
        "pod_name": variant["pod_name"],
        "pod_manifest": variant["pod_manifest"],
        "retrieval_mode": variant["retrieval_mode"],
        "entrypoint": "mlevolve/run.py",
        "model": base.MODEL,
        "endpoint": base.ENDPOINT,
        "credential_secret": base.SECRET,
        "time_limit_seconds": base.TIME_LIMIT_SECONDS,
        "step_limit_mode": "time_only_practical_unbounded",
        "step_ceiling": base.STEP_CEILING,
        "pod_active_deadline_seconds": base.POD_ACTIVE_DEADLINE_SECONDS,
        "preserved_failed_attempt": "clip-r1/attempt-000",
        "preserve_all_prior_tasks_pods_outputs_checkpoints": True,
    }
    base._write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
