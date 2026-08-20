#!/usr/bin/env python3
"""Freeze the isolated UCI-100-Leaves cross-task transfer Dev-Pod smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNTIME = "/workspace/nautilus-exp-end2end-agent-v147-transfer-smoke-r6"
DATA_ROOT = "/workspace/experiment-end2end-uci100-transfer-v147-r6/data/public"
OUTPUT_ROOT = "/workspace/experiment-end2end-uci100-transfer-v147-r6/runs"
POD_NAME = "mlevolve-uci100-gpt56sol-v147-transfer-smoke-r6-dev"
EXPERIMENT_LABEL = "experiment-end2end-uci100-transfer-v147-smoke-r6"
RELEASE_ID = "uci100-same-type-cross-task-transfer-gpt56sol-v147-smoke-r6"
IMAGE = (
    "docker.io/haomingwang22/mlevolve@"
    "sha256:fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc"
)

OVERLAYS = (
    "mlevolve/agents/draft_agent.py",
    "mlevolve/agents/adoption.py",
    "mlevolve/agents/memory/cross_task_transfer.py",
    "mlevolve/agents/memory/stage_aware_hybrid_memory.py",
    "mlevolve/config/__init__.py",
    "mlevolve/engine/agent_search.py",
    "mlevolve/engine/conditions.py",
    "tests/test_cross_task_transfer_v147.py",
    "tests/test_two_role_coverage_fusion_v144.py",
    "experiments/end2end_memory_systems_20260804/prepare_uci_one_hundred_leaves_v147.py",
    "experiments/end2end_memory_systems_20260804/build_uci_transfer_v147_smoke.py",
    "experiments/end2end_memory_systems_20260804/systems_v147_transfer_smoke/dynamic_cross_task_transfer.yaml",
    "experiments/end2end_memory_systems_20260804/systems_v147_transfer_smoke_r3/dynamic_cross_task_transfer.yaml",
    "experiments/end2end_memory_systems_20260804/systems_v147_transfer_smoke_r4/dynamic_cross_task_transfer.yaml",
    "experiments/end2end_memory_systems_20260804/systems_v147_transfer_smoke_r5/dynamic_cross_task_transfer.yaml",
    "experiments/end2end_memory_systems_20260804/systems_v147_transfer_smoke_r6/dynamic_cross_task_transfer.yaml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def copy_overlay(output: Path, relative: str) -> None:
    source = REPO / relative
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def source_lock(output: Path) -> dict:
    lock_path = output / "RELEASE_SOURCE_LOCK.json"
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink() or path == lock_path:
            continue
        relative = path.relative_to(output).as_posix()
        if (
            relative.endswith((".pyc", ".pyo"))
            or "/__pycache__/" in f"/{relative}/"
            or Path(relative).name.startswith("._")
            or Path(relative).name == ".DS_Store"
        ):
            raise ValueError(f"Forbidden runtime artifact: {relative}")
        rows.append({"path": relative, "sha256": sha256_file(path)})
    payload = {
        "schema": "mlevolve_end2end_source_lock_v1",
        "release_id": RELEASE_ID,
        "git_head": git_head(),
        "complete_runtime_file_hash_lock": True,
        "control_file_exclusions": ["RELEASE_SOURCE_LOCK.json"],
        "overlay_scope": list(OVERLAYS),
        "files": rows,
        "manifest_hash": "",
    }
    canonical = json.dumps(
        {**payload, "manifest_hash": ""},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    write_json(lock_path, payload)
    return payload


def build_pod(lock_hash: str) -> dict:
    labels = {
        "app": "mlevolve-end2end-dev",
        "experiment": EXPERIMENT_LABEL,
        "mlevolve.ai/system": "dynamic_cross_task_transfer",
        "mlevolve.ai/release-mode": "smoke",
        "ecepxie.nrp/owner": "haoming",
        "app.kubernetes.io/managed-by": "codex-nrp-training",
    }
    resources = {
        "cpu": "16",
        "memory": "32Gi",
        "ephemeral-storage": "64Gi",
        "nvidia.com/a100": "1",
    }
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": POD_NAME,
            "namespace": "ecepxie",
            "labels": labels,
            "annotations": {
                "mlevolve.ai/purpose": "same-type-cross-task-transfer-eight-step-smoke",
                "mlevolve.ai/runtime-source-lock-sha256": lock_hash,
                "mlevolve.ai/source-task": "leaf-classification",
                "mlevolve.ai/target-task": "uci-one-hundred-leaves",
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 180,
            "containers": [
                {
                    "name": "dev",
                    "image": IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/bash", "-lc"],
                    "args": ["exec sleep infinity"],
                    "env": [
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                        {"name": "OPENAI_MODEL", "value": "gpt-5.6-sol"},
                        {
                            "name": "PYTHONPATH",
                            "value": f"{RUNTIME}/mlevolve:{RUNTIME}",
                        },
                    ],
                    "envFrom": [
                        {"secretRef": {"name": "cliproxyapi-haoming-client"}}
                    ],
                    "resources": {"limits": resources, "requests": resources},
                    "volumeMounts": [
                        {"name": "workspace", "mountPath": "/workspace"},
                        {"name": "shm", "mountPath": "/dev/shm"},
                    ],
                }
            ],
            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
            "volumes": [
                {
                    "name": "workspace",
                    "persistentVolumeClaim": {"claimName": "haoming-storage"},
                },
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--pod-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
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
    for relative in OVERLAYS:
        copy_overlay(args.output_runtime, relative)
    lock = source_lock(args.output_runtime)
    write_json(args.pod_out, build_pod(lock["manifest_hash"]))
    receipt = {
        "schema": "uci100_cross_task_transfer_v147_smoke_build_v1",
        "release_id": RELEASE_ID,
        "git_head": git_head(),
        "runtime": RUNTIME,
        "local_runtime": str(args.output_runtime.resolve()),
        "source_lock_hash": lock["manifest_hash"],
        "source_lock_file_count": len(lock["files"]),
        "pod_name": POD_NAME,
        "experiment_label": EXPERIMENT_LABEL,
        "source_task_id": "leaf-classification",
        "target_task_id": "uci-one-hundred-leaves",
        "draft_roles": ["memory_transfer", "novel_exploration"],
        "agent_steps": 8,
        "data_root": DATA_ROOT,
        "output_root": OUTPUT_ROOT,
        "gpu_resource": "nvidia.com/a100",
        "gpu_count": 1,
        "cpu": 16,
        "memory_gib": 32,
    }
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
