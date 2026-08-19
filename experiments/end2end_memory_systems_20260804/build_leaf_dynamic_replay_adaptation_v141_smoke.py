#!/usr/bin/env python3
"""Build the isolated 16-step Dynamic v141 Dev-Pod smoke release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import build_leaf_dynamic_retrieval_v137_runtime as v137
import build_leaf_ten_system_gpt_v135_runtime as v135


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
SUFFIX = "v141-smoke16"
MANIFEST_DIR = EXPERIMENT / "manifests_v141_smoke16"
SYSTEM_DIR = EXPERIMENT / "systems_v141_smoke16"
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v141-smoke16"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v141-smoke16/runs"
EVALUATOR_ROOT = "/workspace/experiment-end2end-leaf-official-evaluator-v141-smoke16"
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v141-smoke16"
POD_NAME = "mlevolve-leaf-gpt56sol-v141-smoke16-dev"
DEV_MEMORY_GIB = 64
MEMORY_ROOT = (
    "/workspace/experiment-end2end-memory-agent-v140-r6/"
    "memory-leaf-llm-redistilled-v10-r6/leaf-classification"
)
MEMORY_BUNDLE = (
    f"{MEMORY_ROOT}/bundles/v10-r6-leaf-llm-redistilled-20260819"
)

OVERLAY_FILES = (
    Path("mlevolve/llm/openai.py"),
    Path("mlevolve/agents/aggregation_agent.py"),
    Path("mlevolve/agents/debug_agent.py"),
    Path("mlevolve/agents/fusion_agent.py"),
    Path("mlevolve/agents/memory/multigranular_grep.py"),
    Path("mlevolve/agents/result_log_facts.py"),
    Path("mlevolve/agents/result_parse_agent.py"),
    Path("mlevolve/agents/triggers.py"),
    Path("mlevolve/config/__init__.py"),
    Path("mlevolve/engine/agent_search.py"),
    Path("mlevolve/engine/conditions.py"),
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
    Path("tests/test_preexperiment_repairs.py"),
    Path("tests/test_result_parse_full_output.py"),
    Path("tests/test_role_resource_balance_v138.py"),
    Path("tests/test_run_forest_memory.py"),
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
        "smoke_agent_steps": 16,
        "manifest_dir": MANIFEST_DIR,
        "system_dir": SYSTEM_DIR,
        "execution_name": "leaf_dynamic_smoke_manifest.json",
        "release_id": f"end2end-leaf-dynamic-replay-adaptation-gpt56sol-{SUFFIX}",
        "cluster_runtime": CLUSTER_RUNTIME,
        "output_root": OUTPUT_ROOT,
        "evaluator_root": EVALUATOR_ROOT,
        "experiment_label": EXPERIMENT_LABEL,
        "workload": POD_NAME,
        "stager": "unused-v141-smoke16-dev-stager",
        "logical_run_id": (
            "e2e-smoke-leaf-dynamic-replay-adaptation-official-gpt56sol-"
            f"{SUFFIX}__leaf-classification__dynamic_hybrid__seed-1"
        ),
    }


def write_dynamic_config(output: Path) -> Path:
    target = output / SYSTEM_DIR / "dynamic_hybrid.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        "\n".join(
            [
                "# v141 changes only Dynamic Replay adaptation, alignment repair,",
                "# balanced cross-role synthesis, and the audited v10-r6 memory binding.",
                "extends: ../systems_v138_smoke_r3/dynamic_hybrid.yaml",
                "",
                "agent:",
                "  draft_role_policy:",
                "    ensure_valid_candidate_per_role: true",
                "    role_balance_min_valid_candidates: 1",
                f"    replay_targets_path: {MEMORY_BUNDLE}/reports/leaf_official_replay_targets_v139.json",
                "    replay_adaptation_as_novel: true",
                "    replay_alignment_repair_enabled: true",
                "    replay_alignment_repair_max_attempts: 1",
                "    cross_role_synthesis_after_balance: true",
                "",
                "external_skill_memory:",
                f"  recipe_sop_path: {MEMORY_BUNDLE}/recipe/recipe_sops.json",
                "  recipe_sop_file_sha256: fcfb8e6220e4548c96e46a178d0b460cd4232e9dee54d84689c1b5ea9fc49004",
                "  recipe_sop_bundle_sha256: 5189a8b7c32670d5ef40f667a38fd5b4b2d9933767a4addc4c074e5630579ae0",
                f"  recipe_evidence_path: {MEMORY_BUNDLE}/recipe/evidence_manifest.json",
                "  recipe_evidence_file_sha256: 130b0bfa7e23d23a457e74bf8b5b243f5f3bbe11cd395ffa5d9e74dd702f8d68",
                "  recipe_evidence_manifest_sha256: cce7ec191e8513246b58c013ab516645ba5cfd2832e53391f173b2b7b415d755",
                f"  recipe_implementation_path: {MEMORY_BUNDLE}/recipe/implementation_capsules.json",
                f"  transition_evidence_capsules_path: {MEMORY_BUNDLE}/recipe/transition_evidence_capsules.json",
                "  transition_evidence_capsules_sha256: 951db74cc7098286fc6395c4942ad3a36765c2afd609cc014543e9a77a1d3d83",
                "",
                "run_identity:",
                "  memory_version: leaf_llm_redistilled_v10_r6_dynamic_replay_adaptation_"
                f"{SUFFIX.replace('-', '_')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def update_memory_manifest(output: Path) -> str:
    path = output / MANIFEST_DIR / "memory_bundles.json"
    payload = v135.read_json(path)
    task = payload["task_bundles"]["leaf-classification"]
    task.update(
        {
            "bundle_id": "end2end-leaf-llm-redistilled-recipe-runforest-v10-r6",
            "bundle_version": "v10-r6-leaf-llm-redistilled-20260819",
            "bundle_root": MEMORY_ROOT,
            "bundle_manifest_sha256": "c46c6fb4e582bf079fd199c8732275293d3f63109f68924a0796b4dc8077e963",
            "bundle_manifest_file_sha256": "9deaf1b1d68ad0152017172566b4381ba836338b87a1286aed07738a61ba0050",
            "current_file_sha256": "8d1304a56d2d9d245738d4c0608770ebd50324182b551a85f74357a55ef00cc4",
            "graph_sha256": "cd84a7e76721139eb1068183c7e92bf2620ee8f2000b7d86e9f52e4f1ef6cd27",
            "index_sha256": "2488813025311b1081daf414b6ff7b5df59247fe9de77868721ce4dbe43c15b2",
            "recipe_sop_bundle_sha256": "5189a8b7c32670d5ef40f667a38fd5b4b2d9933767a4addc4c074e5630579ae0",
            "recipe_evidence_manifest_sha256": "cce7ec191e8513246b58c013ab516645ba5cfd2832e53391f173b2b7b415d755",
            "atomic_claim_bundle_sha256": "ef187196c8b8dd4e60bb4d33a5b7ea1eef65fe485751cfc8f89f62f1e3c11bf3",
            "atomic_debug_authorized_count": 276,
            "formal_clause_file_sha256": "f0074ce169f75210637e7769af88cdf2d95ea804345efdb31eb49000b73f38a1",
            "formal_debug_clause_count": 288,
            "official_ledger_sha256": "ab13a6aa68430aea04123cbd702b68ae9cf31ce83d3658dfbfb7ed13e2170353",
            "certification_level": "gpt56sol_llm_redistilled_audited",
            "memory_scope": "leaf_llm_redistilled_recipe_tactic_repair_plus_latest_runforest",
            "positive_eligible_count": 6,
            "same_task_best_official_metric": 0.00101,
            "same_task_best_validation_protocol": "official_kaggle_scored_test",
            "same_task_history_enabled": True,
        }
    )
    payload.update(
        {
            "source_graph_sha256": task["graph_sha256"],
            "source_index_sha256": task["index_sha256"],
            "atomic_claim_bundle_sha256": task["atomic_claim_bundle_sha256"],
            "atomic_debug_authorized_count": 276,
            "formal_debug_clause_count": 288,
            "ranking_policy": "authority_multigranular_grep_independent_judge_plus_l3_root_cause_v1",
        }
    )
    return v135.write_hashed(path, payload, "manifest_hash")


def rewrite_source_lock(output: Path, run_spec: dict) -> tuple[str, int]:
    manifests = output / MANIFEST_DIR
    excluded = {
        (MANIFEST_DIR / "source_lock.json").as_posix(),
        (MANIFEST_DIR / run_spec["execution_name"]).as_posix(),
        "SOURCE_FILES.sha256",
    }
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        files.append({"path": relative, "sha256": v135.sha256_file(path)})
    source_lock = {
        "schema": "mlevolve_end2end_source_lock_v1",
        "release_id": run_spec["release_id"],
        "git_head": v135.git_head(),
        "git_head_is_not_sufficient_identity": True,
        "complete_runtime_file_hash_lock": True,
        "control_file_exclusions": sorted(excluded),
        "overlay_scope": [path.as_posix() for path in OVERLAY_FILES],
        "files": files,
        "manifest_hash": "",
    }
    digest = v135.write_hashed(
        manifests / "source_lock.json", source_lock, "manifest_hash"
    )
    return digest, len(files)


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
                "mlevolve.ai/purpose": "sixteen-step-three-role-dynamic-smoke",
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
    bindings["memory_bundles_manifest_hash"] = update_memory_manifest(
        output_runtime
    )
    source_lock_hash, source_lock_count = rewrite_source_lock(
        output_runtime, run_spec
    )
    bindings["source_lock_manifest_hash"] = source_lock_hash
    execution = v137.build_execution(run_spec, bindings)
    execution_hash = v135.write_hashed(
        output_runtime / MANIFEST_DIR / run_spec["execution_name"],
        execution,
        "manifest_hash",
    )
    v135.write_json(
        pod_out, build_dev_pod(execution_hash, source_lock_hash)
    )
    return {
        "schema": "mlevolve_leaf_dynamic_replay_adaptation_v141_smoke_build_v1",
        "status": "complete",
        "release_id": run_spec["release_id"],
        "runtime_root": str(output_runtime),
        "cluster_runtime": CLUSTER_RUNTIME,
        "output_root": OUTPUT_ROOT,
        "evaluator_root": EVALUATOR_ROOT,
        "logical_run_id": run_spec["logical_run_id"],
        "pod_name": POD_NAME,
        "source_lock_hash": source_lock_hash,
        "source_lock_file_count": source_lock_count,
        "execution_manifest_hash": execution_hash,
        "memory_bundles_manifest_hash": bindings[
            "memory_bundles_manifest_hash"
        ],
        "dynamic_config_sha256": v135.sha256_file(dynamic_config),
        "agent_steps": 16,
        "dev_memory_gib": DEV_MEMORY_GIB,
        "memory_bundle_root": MEMORY_ROOT,
        "memory_bundle_manifest_sha256": "c46c6fb4e582bf079fd199c8732275293d3f63109f68924a0796b4dc8077e963",
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
    """Advance every mutable identity after a pre-Pod staging failure."""

    if generation < 1:
        raise ValueError("generation must be positive")
    if generation == 1:
        return
    revision = f"-r{generation}"
    manifest_revision = f"_r{generation}"
    global SUFFIX, MANIFEST_DIR, SYSTEM_DIR, CLUSTER_RUNTIME
    global OUTPUT_ROOT, EVALUATOR_ROOT, EXPERIMENT_LABEL, POD_NAME
    global DEV_MEMORY_GIB
    SUFFIX = f"v141-smoke16{revision}"
    MANIFEST_DIR = EXPERIMENT / f"manifests_v141_smoke16{manifest_revision}"
    SYSTEM_DIR = EXPERIMENT / f"systems_v141_smoke16{manifest_revision}"
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
