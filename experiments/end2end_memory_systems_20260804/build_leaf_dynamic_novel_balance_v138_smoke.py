#!/usr/bin/env python3
"""Build the isolated Dynamic v138 six-step Dev-Pod smoke release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import build_leaf_dynamic_retrieval_v137_runtime as v137
import build_leaf_ten_system_gpt_v135_runtime as v135


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
SUFFIX = "v138-smoke"
MANIFEST_DIR = EXPERIMENT / "manifests_v138_smoke"
SYSTEM_DIR = EXPERIMENT / "systems_v138_smoke"
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v138-smoke"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v138-smoke/runs"
EVALUATOR_ROOT = "/workspace/experiment-end2end-leaf-official-evaluator-v138-smoke"
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v138-smoke"
POD_NAME = "mlevolve-leaf-gpt56sol-v138-smoke-dev"
DEV_MEMORY_GIB = 64

OVERLAY_FILES = (
    Path("mlevolve/llm/openai.py"),
    Path("mlevolve/agents/memory/multigranular_grep.py"),
    Path("mlevolve/config/__init__.py"),
    Path("mlevolve/engine/agent_search.py"),
    Path("mlevolve/engine/executor.py"),
    Path("mlevolve/engine/node_selection.py"),
    Path("mlevolve/engine/role_balance.py"),
)
TEST_FILES = (
    Path("tests/test_gpt_openai_compatible_config.py"),
    Path("tests/test_multigranular_grep_retrieval.py"),
    Path("tests/test_experiment_r_dynamic_routing.py"),
    Path("tests/test_l3_grep_search_agent.py"),
    Path("tests/test_executor_host_preamble_v138.py"),
    Path("tests/test_role_resource_balance_v138.py"),
)
TEST_SUPPORT_FILES = (
    Path("experiments/dynamic_memory_routing_injection_20260731/design.py"),
    Path("tests/test_stage_aware_hybrid_memory.py"),
)


def spec() -> dict:
    return {
        "mode": "smoke",
        "suffix": SUFFIX,
        "kind": "smoke",
        "smoke_agent_steps": 6,
        "manifest_dir": MANIFEST_DIR,
        "system_dir": SYSTEM_DIR,
        "execution_name": "leaf_dynamic_smoke_manifest.json",
        "release_id": f"end2end-leaf-dynamic-novel-balance-gpt56sol-{SUFFIX}",
        "cluster_runtime": CLUSTER_RUNTIME,
        "output_root": OUTPUT_ROOT,
        "evaluator_root": EVALUATOR_ROOT,
        "experiment_label": EXPERIMENT_LABEL,
        "workload": POD_NAME,
        "stager": "unused-v138-dev-stager",
        "logical_run_id": (
            f"e2e-smoke-leaf-dynamic-novel-balance-official-gpt56sol-{SUFFIX}__"
            "leaf-classification__dynamic_hybrid__seed-1"
        ),
    }


def write_dynamic_config(output: Path) -> Path:
    target = output / SYSTEM_DIR / "dynamic_hybrid.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        "\n".join(
            [
                "# v138 changes only Dynamic: repair Host preamble composition and",
                "# protect equal startup resources for Cold Start, Replay, and Novel.",
                "extends: ../systems_v137_full/dynamic_hybrid.yaml",
                "",
                "agent:",
                "  draft_role_policy:",
                "    ensure_valid_candidate_per_role: true",
                "    role_balance_min_valid_candidates: 1",
                f"    replay_targets_path: {CLUSTER_RUNTIME}/{MANIFEST_DIR}/leaf_official_replay_targets.json",
                "",
                "external_skill_memory:",
                f"  transition_evidence_capsules_path: {CLUSTER_RUNTIME}/{EXPERIMENT}/transition_evidence_v122/transition_evidence_capsules.json",
                "",
                "run_identity:",
                f"  memory_version: leaf_dynamic_novel_host_fix_role_balance_gpt56sol_{SUFFIX.replace('-', '_')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def build_dev_pod(manifest_hash: str, source_lock_hash: str) -> dict:
    labels = {
        "app": "mlevolve-end2end-dev",
        "experiment": EXPERIMENT_LABEL,
        "mlevolve.ai/system": "dynamic_hybrid",
        "mlevolve.ai/release-mode": "smoke",
        "ecepxie.nrp/owner": "haoming",
        "app.kubernetes.io/managed-by": "codex-nrp-training",
    }
    resources = {
        "cpu": "16",
        "memory": f"{DEV_MEMORY_GIB}Gi",
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
                "mlevolve.ai/manifest-sha256": manifest_hash,
                "mlevolve.ai/runtime-source-lock-sha256": source_lock_hash,
                "mlevolve.ai/purpose": "three-role-dynamic-smoke",
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 180,
            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
            "containers": [
                {
                    "name": "dev",
                    "image": v135.RUNTIME_IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/bash", "-lc"],
                    "args": ["exec sleep infinity"],
                    "envFrom": [{"secretRef": {"name": v135.LLM_SECRET}}],
                    "env": [
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                        {
                            "name": "PYTHONPATH",
                            "value": f"{CLUSTER_RUNTIME}/mlevolve:{CLUSTER_RUNTIME}",
                        },
                    ],
                    "resources": {
                        "requests": dict(resources),
                        "limits": dict(resources),
                    },
                    "volumeMounts": [
                        {"name": "workspace", "mountPath": "/workspace"},
                        {"name": "shm", "mountPath": "/dev/shm"},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "workspace",
                    "persistentVolumeClaim": {"claimName": "haoming-storage"},
                },
                {
                    "name": "shm",
                    "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"},
                },
            ],
        },
    }


def build(base_runtime: Path, output_runtime: Path, pod_out: Path) -> dict:
    if output_runtime.exists():
        raise FileExistsError(f"fresh runtime already exists: {output_runtime}")
    if pod_out.exists():
        raise FileExistsError(f"fresh Pod manifest already exists: {pod_out}")
    shutil.copytree(base_runtime.resolve(strict=True), output_runtime, symlinks=True)
    v137.remove_runtime_caches(output_runtime)
    for relative in (
        *OVERLAY_FILES,
        *TEST_FILES,
        *TEST_SUPPORT_FILES,
        Path(__file__).relative_to(REPO),
    ):
        v135.copy_file(REPO / relative, output_runtime / relative)

    run_spec = spec()
    v137.OVERLAY_FILES = OVERLAY_FILES
    dynamic_config = write_dynamic_config(output_runtime)
    bindings = v137.build_components(output_runtime, run_spec)
    execution = v137.build_execution(run_spec, bindings)
    runtime_manifests = output_runtime / MANIFEST_DIR
    execution_hash = v135.write_hashed(
        runtime_manifests / run_spec["execution_name"],
        execution,
        "manifest_hash",
    )
    source_lock = v135.read_json(runtime_manifests / "source_lock.json")
    pod = build_dev_pod(execution_hash, source_lock["manifest_hash"])
    v135.write_json(pod_out, pod)
    return {
        "schema": "mlevolve_leaf_dynamic_novel_balance_v138_smoke_build_v1",
        "status": "complete",
        "release_id": run_spec["release_id"],
        "runtime_root": str(output_runtime),
        "cluster_runtime": CLUSTER_RUNTIME,
        "output_root": OUTPUT_ROOT,
        "evaluator_root": EVALUATOR_ROOT,
        "logical_run_id": run_spec["logical_run_id"],
        "pod_name": POD_NAME,
        "source_lock_hash": source_lock["manifest_hash"],
        "source_lock_file_count": len(source_lock["files"]),
        "execution_manifest_hash": execution_hash,
        "dynamic_config_sha256": v135.sha256_file(dynamic_config),
        "agent_steps": 6,
        "role_balance_min_valid_candidates": 1,
        "dev_memory_gib": DEV_MEMORY_GIB,
        "llm_model": v135.LLM_MODEL,
        "llm_base_url": v135.LLM_BASE_URL,
        "llm_secret_reference": v135.LLM_SECRET,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--pod-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--generation", type=int, default=1)
    args = parser.parse_args()
    configure_generation(args.generation)
    receipt = build(args.base_runtime, args.output_runtime, args.pod_out)
    v135.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def configure_generation(generation: int) -> None:
    """Select a fresh release identity after a pre-Pod staging failure."""

    if generation < 1:
        raise ValueError("generation must be positive")
    if generation == 1:
        return
    revision = f"-r{generation}"
    manifest_revision = f"_r{generation}"
    global SUFFIX, MANIFEST_DIR, SYSTEM_DIR, CLUSTER_RUNTIME
    global OUTPUT_ROOT, EVALUATOR_ROOT, EXPERIMENT_LABEL, POD_NAME, DEV_MEMORY_GIB
    SUFFIX = f"v138-smoke{revision}"
    MANIFEST_DIR = EXPERIMENT / f"manifests_v138_smoke{manifest_revision}"
    SYSTEM_DIR = EXPERIMENT / f"systems_v138_smoke{manifest_revision}"
    CLUSTER_RUNTIME = f"/workspace/nautilus-exp-end2end-agent-{SUFFIX}"
    OUTPUT_ROOT = f"/workspace/experiment-end2end-memory-agent-{SUFFIX}/runs"
    EVALUATOR_ROOT = (
        f"/workspace/experiment-end2end-leaf-official-evaluator-{SUFFIX}"
    )
    EXPERIMENT_LABEL = f"experiment-end2end-memory-agent-{SUFFIX}"
    POD_NAME = f"mlevolve-leaf-gpt56sol-{SUFFIX}-dev"
    DEV_MEMORY_GIB = 32


if __name__ == "__main__":
    raise SystemExit(main())
