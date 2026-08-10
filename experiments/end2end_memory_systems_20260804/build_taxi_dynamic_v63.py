#!/usr/bin/env python3
"""Build the immutable Taxi Dynamic-only v63 capsule-replay rerun packet."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import build_taxi_cpu_v51 as builder


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
VERSION = "v63"
MANIFESTS = ROOT / f"manifests_{VERSION}"
TARGET_NAME = "taxi_dynamic_replay_targets.json"
MANIFEST_NAME = "taxi_dynamic_pilot_manifest.json"
WORKLOAD = "mlevolve-e2e-taxi-dynamic-replay-v63"


def configure() -> None:
    builder.VERSION = VERSION
    builder.MANIFESTS = MANIFESTS
    builder.SYSTEMS_V51 = ROOT / f"systems_{VERSION}"
    builder.SYSTEM_IDS = ["dynamic_hybrid"]
    builder.RELEASE_ID = "end2end-taxi-dynamic-replay-v63"
    builder.WORKLOAD = WORKLOAD
    builder.EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v63"
    builder.CLUSTER_REPO = "/workspace/nautilus-exp-end2end-agent-v63"
    builder.CLUSTER_ROOT = (
        f"{builder.CLUSTER_REPO}/experiments/end2end_memory_systems_20260804"
    )
    builder.OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v63/runs"
    builder.SOURCE_PARENT = "/workspace/nautilus-exp-end2end-agent-v62"
    builder.EXECUTION_MODE = "gpu"
    builder.GPU_RESOURCE_KEY = "nvidia.com/gpu"
    builder.GPU_TYPE = "Any schedulable NVIDIA GPU"
    builder.COMPUTE_LABEL = "dynamic-anygpu"
    builder.MANIFEST_FILENAME = MANIFEST_NAME


def repaired_source_lock() -> dict:
    source_lock = builder.read_json(ROOT / "manifests_v62" / "source_lock.json")
    source_lock["release_id"] = builder.RELEASE_ID
    source_lock["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    source_lock["source_parent"] = builder.SOURCE_PARENT
    source_lock["worktree_patch"] = "taxi_dynamic_capsule_replay_binding_v1"
    paths = {str(row["path"]) for row in source_lock["files"]}
    paths.update(
        {
            "experiments/end2end_memory_systems_20260804/systems_v63/dynamic_hybrid.yaml",
            f"experiments/end2end_memory_systems_20260804/manifests_v63/{TARGET_NAME}",
        }
    )
    files = []
    for relative in sorted(paths):
        path = REPO / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v63 source-lock input unavailable: {relative}")
        files.append({"path": relative, "sha256": builder.sha256_file(path)})
    source_lock["files"] = files
    builder.self_hash(source_lock)
    return source_lock


def build() -> dict:
    configure()
    target = builder.read_json(MANIFESTS / TARGET_NAME)
    if MANIFESTS.exists():
        shutil.rmtree(MANIFESTS)
    MANIFESTS.mkdir(parents=True)
    builder.write_json(MANIFESTS / TARGET_NAME, target)

    components = builder.build_components()
    components["source_lock"] = repaired_source_lock()
    for name, payload in components.items():
        builder.write_json(MANIFESTS / f"{name}.json", payload)

    manifest = builder.build_manifest(components)
    manifest["formal_result_eligible"] = False
    manifest["launch_order_randomization"] = (
        "single Taxi Dynamic exploratory rerun after pre-search capsule binding repair"
    )
    for row in manifest["runs"]:
        row["formal_result_eligible"] = False
        row["row_hash"] = builder.payload_hash(row, "row_hash")
    builder.self_hash(manifest)
    builder.write_json(MANIFESTS / MANIFEST_NAME, manifest)

    job = builder.build_job(manifest)
    job["spec"].update(
        {
            "completions": 1,
            "parallelism": 1,
            "maxFailedIndexes": 1,
            "activeDeadlineSeconds": 27000,
        }
    )
    annotations = job["metadata"]["annotations"]
    annotations["mlevolve.ai/max-total-cpu-parallelism"] = "1"
    annotations["mlevolve.ai/global-deadline-seconds"] = "27000"
    annotations["mlevolve.ai/replay-target-repair"] = (
        "bind-clean-taxi-best-to-frozen-implementation-capsule"
    )
    builder.write_json(builder.JOB_DIR / f"{WORKLOAD}.yaml", job)

    packet = {
        "schema": "mlevolve_end2end_taxi_dynamic_launch_packet_v63",
        "status": "generated_not_submitted",
        "release_id": builder.RELEASE_ID,
        "source_root": builder.CLUSTER_REPO,
        "source_parent": builder.SOURCE_PARENT,
        "output_root": builder.OUTPUT_ROOT,
        "manifest": f"manifests_v63/{MANIFEST_NAME}",
        "manifest_sha256": manifest["manifest_hash"],
        "job": f"jobs/{WORKLOAD}.yaml",
        "workload": WORKLOAD,
        "task": builder.TASK_ID,
        "systems": ["dynamic_hybrid"],
        "completions": 1,
        "parallelism": 1,
        "per_condition_search_budget_seconds": 21600,
        "compute_contract": "nvidia.com/gpu=1; 16 CPU and 64Gi RAM",
        "memory_binding_sha256": builder.TAXI_BINDING["binding_sha256"],
        "replaces_failed_workload": WORKLOAD.replace("v63", "v62"),
        "checkpoint_policy": "fresh attempt-000; prior failures retained; no resume",
        "packet_hash": "",
    }
    builder.self_hash(packet, "packet_hash")
    builder.write_json(MANIFESTS / "launch_packet.json", packet)
    return packet


def check() -> None:
    configure()
    target = builder.read_json(MANIFESTS / TARGET_NAME)["targets"][0]
    if target.get("source_kind") != "recipe_implementation_capsule":
        raise RuntimeError("Taxi exact replay must use the frozen implementation capsule")
    if target["sop_ids"] != ["sop::sg_0081"]:
        raise RuntimeError("Taxi replay SOP repair drift")
    expected_node = (
        "run::20260726_022228_new-york-city-taxi-fare-prediction-host-shadow-r7"
        "::node::eeb6e2364829449ba6e1ce6c1600fc3d"
    )
    if target.get("graph_node_id") != expected_node:
        raise RuntimeError("Taxi capsule graph-node binding drift")
    for name in (
        "systems",
        "tasks",
        "budget",
        "memory_bundles",
        "evaluators",
        "schemas",
        "source_lock",
        Path(MANIFEST_NAME).stem,
    ):
        payload = builder.read_json(MANIFESTS / f"{name}.json")
        if builder.payload_hash(payload, "manifest_hash") != payload.get("manifest_hash"):
            raise RuntimeError(f"v63 self-hash mismatch: {name}")
    manifest = builder.read_json(MANIFESTS / MANIFEST_NAME)
    if manifest["system_ids"] != ["dynamic_hybrid"] or manifest["run_count"] != 1:
        raise RuntimeError("v63 must contain only Taxi Dynamic")
    if manifest["formal_result_eligible"] is not False:
        raise RuntimeError("seed-1 v63 must remain exploratory")
    source_lock = builder.read_json(MANIFESTS / "source_lock.json")
    for row in source_lock["files"]:
        if builder.sha256_file(REPO / row["path"]) != row["sha256"]:
            raise RuntimeError(f"v63 source drift: {row['path']}")
    job = builder.read_json(builder.JOB_DIR / f"{WORKLOAD}.yaml")
    if job["spec"]["completions"] != 1 or job["spec"]["parallelism"] != 1:
        raise RuntimeError("v63 Job shape drift")
    resources = job["spec"]["template"]["spec"]["containers"][0]["resources"]
    if resources["requests"] != resources["limits"]:
        raise RuntimeError("v63 resource requests/limits drift")
    if resources["limits"].get("nvidia.com/gpu") != "1":
        raise RuntimeError("v63 GPU contract drift")
    if "--resume" in job["spec"]["template"]["spec"]["containers"][0]["args"]:
        raise RuntimeError("v63 must start fresh")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))
