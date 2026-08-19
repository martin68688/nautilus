#!/usr/bin/env python3
"""Build fresh v143-r2 smoke assets bound to the audited v10-r8 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_leaf_dynamic_replay_adaptation_v141_smoke as v141
import build_leaf_dynamic_replay_adaptation_v143_smoke as v143
import build_leaf_ten_system_gpt_v135_runtime as v135


EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
SUFFIX = "v143-smoke16-r2"
MANIFEST_DIR = EXPERIMENT / "manifests_v143_smoke16_r2"
SYSTEM_DIR = EXPERIMENT / "systems_v143_smoke16_r2"
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v143-smoke16-r2"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v143-smoke16-r2/runs"
EVALUATOR_ROOT = (
    "/workspace/experiment-end2end-leaf-official-evaluator-v143-smoke16-r2"
)
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v143-smoke16-r2"
POD_NAME = "mlevolve-leaf-gpt56sol-v143-smoke16-r2-dev"
MEMORY_ROOT = (
    "/workspace/experiment-end2end-memory-agent-v143-r2/"
    "memory-leaf-llm-redistilled-v10-r8/leaf-classification"
)
MEMORY_BUNDLE = (
    f"{MEMORY_ROOT}/bundles/v10-r8-leaf-llm-redistilled-20260820"
)

MANIFEST_SHA256 = "e2b037f9a3742c13f763764a7f92697131b89e3dcf9ef0ab242dc09b7deca498"
MANIFEST_FILE_SHA256 = "c2aa61226356e6724e5eaf465b1c8133463ff0ef3392f471479028a523611fe0"
CURRENT_FILE_SHA256 = "9cd534afd92229ce5cf577c3329559ad18ff8840c393744e7d58a9ff22547d73"
RECIPE_FILE_SHA256 = "8cde37ef7fc1ba0a2867f264122d1d8e68c5953aca89bfc2dfc12fb242695a81"
RECIPE_BUNDLE_SHA256 = "a452926e8ee2f463d68112a3532d0a623cd3c258c54ed90d58e934b5d0a423fe"
EVIDENCE_FILE_SHA256 = "9e92bace6636b88e97eae5e4da1bd404e50df32aaf2d883ad0d0f8fd9e5ea11a"
EVIDENCE_MANIFEST_SHA256 = "b001e0b340354a0dfb670616b05d5ebd502c11793971f2788c91b1fcd1448ced"
TRANSITION_EVIDENCE_SHA256 = "951db74cc7098286fc6395c4942ad3a36765c2afd609cc014543e9a77a1d3d83"


def configure_builder() -> None:
    v143.configure_v141_builder()
    v141.OVERLAY_FILES = (
        *v141.OVERLAY_FILES,
        Path("mlevolve/agents/code_review_agent.py"),
    )
    v141.SUFFIX = SUFFIX
    v141.MANIFEST_DIR = MANIFEST_DIR
    v141.SYSTEM_DIR = SYSTEM_DIR
    v141.CLUSTER_RUNTIME = CLUSTER_RUNTIME
    v141.OUTPUT_ROOT = OUTPUT_ROOT
    v141.EVALUATOR_ROOT = EVALUATOR_ROOT
    v141.EXPERIMENT_LABEL = EXPERIMENT_LABEL
    v141.POD_NAME = POD_NAME
    v141.DEV_MEMORY_GIB = 32
    v141.GPU_RESOURCE_KEY = "nvidia.com/a100"
    v141.GPU_PRODUCT_CONSTRAINT = None
    v141.GPU_TYPE = "NVIDIA A100 family"
    v141.GPU_COUNT = 1
    v141.MEMORY_ROOT = MEMORY_ROOT
    v141.MEMORY_BUNDLE = MEMORY_BUNDLE
    v141.TEST_FILES = (
        *v141.TEST_FILES,
        Path("tests/test_leaf_v143_replay_target_recipe_projection.py"),
    )
    v141.TEST_SUPPORT_FILES = (
        *v141.TEST_SUPPORT_FILES,
        Path(
            "experiments/end2end_memory_systems_20260804/"
            "publish_leaf_replay_target_projection_v10.py"
        ),
        Path(
            "experiments/end2end_memory_systems_20260804/"
            "build_leaf_dynamic_replay_adaptation_v143_r2_smoke.py"
        ),
        Path(
            "experiments/end2end_memory_systems_20260804/"
            "V143_SMOKE16_R2_PRELAUNCH_FAILURE.json"
        ),
        Path(
            "experiments/end2end_memory_systems_20260804/"
            "V143_SMOKE16_R3_PRELAUNCH_FAILURE.json"
        ),
        Path(
            "experiments/end2end_memory_systems_20260804/"
            "V143_SMOKE16_R4_TERMINATED_FOR_FIX.json"
        ),
    )
    v141.write_dynamic_config = write_dynamic_config
    v141.update_memory_manifest = update_memory_manifest


def write_dynamic_config(output: Path) -> Path:
    target = output / SYSTEM_DIR / "dynamic_hybrid.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        "\n".join(
            [
                "# v143-r2 changes only immutable Replay artifact/Recipe bindings.",
                "# Dynamic search, roles, alignment, and candidate code are unchanged.",
                "extends: ../systems_v142_smoke16/dynamic_hybrid.yaml",
                "",
                "agent:",
                "  draft_role_policy:",
                f"    replay_targets_path: {MEMORY_BUNDLE}/reports/leaf_official_replay_targets_v139.json",
                f"    replay_runs_root: {MEMORY_BUNDLE}/run_artifacts",
                "",
                "external_skill_memory:",
                f"  recipe_sop_path: {MEMORY_BUNDLE}/recipe/recipe_sops.json",
                f"  recipe_sop_file_sha256: {RECIPE_FILE_SHA256}",
                f"  recipe_sop_bundle_sha256: {RECIPE_BUNDLE_SHA256}",
                f"  recipe_evidence_path: {MEMORY_BUNDLE}/recipe/evidence_manifest.json",
                f"  recipe_evidence_file_sha256: {EVIDENCE_FILE_SHA256}",
                f"  recipe_evidence_manifest_sha256: {EVIDENCE_MANIFEST_SHA256}",
                f"  recipe_implementation_path: {MEMORY_BUNDLE}/recipe/implementation_capsules.json",
                f"  transition_evidence_capsules_path: {MEMORY_BUNDLE}/recipe/transition_evidence_capsules.json",
                f"  transition_evidence_capsules_sha256: {TRANSITION_EVIDENCE_SHA256}",
                "",
                "run_identity:",
                "  memory_version: leaf_llm_redistilled_v10_r8_"
                f"replay_recipe_projection_{SUFFIX.replace('-', '_')}",
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
            "bundle_id": "end2end-leaf-llm-redistilled-recipe-runforest-v10-r8",
            "bundle_version": "v10-r8-leaf-llm-redistilled-20260820",
            "bundle_root": MEMORY_ROOT,
            "bundle_manifest_sha256": MANIFEST_SHA256,
            "bundle_manifest_file_sha256": MANIFEST_FILE_SHA256,
            "current_file_sha256": CURRENT_FILE_SHA256,
            "graph_sha256": "cd84a7e76721139eb1068183c7e92bf2620ee8f2000b7d86e9f52e4f1ef6cd27",
            "index_sha256": "2488813025311b1081daf414b6ff7b5df59247fe9de77868721ce4dbe43c15b2",
            "recipe_sop_bundle_sha256": RECIPE_BUNDLE_SHA256,
            "recipe_evidence_manifest_sha256": EVIDENCE_MANIFEST_SHA256,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--pod-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--generation", type=int, default=2)
    args = parser.parse_args()
    configure_identity(args.generation)
    configure_builder()
    receipt = v141.build(args.base_runtime, args.output_runtime, args.pod_out)
    receipt.update(
        {
            "schema": "mlevolve_leaf_dynamic_replay_adaptation_v143_smoke_build_v1",
            "memory_bundle_root": MEMORY_ROOT,
            "memory_bundle_manifest_sha256": MANIFEST_SHA256,
            "reused_existing_dev_pod": True,
            "reused_evaluator_root": EVALUATOR_ROOT,
            "pod_manifest_must_not_be_created": True,
        }
    )
    v135.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def configure_identity(generation: int) -> None:
    """Advance every mutable smoke identity after a prelaunch failure."""

    if generation < 2:
        raise ValueError("v143 replay projection generations start at r2")
    global SUFFIX, MANIFEST_DIR, SYSTEM_DIR, CLUSTER_RUNTIME
    global OUTPUT_ROOT, EVALUATOR_ROOT, EXPERIMENT_LABEL, POD_NAME
    SUFFIX = f"v143-smoke16-r{generation}"
    MANIFEST_DIR = EXPERIMENT / f"manifests_v143_smoke16_r{generation}"
    SYSTEM_DIR = EXPERIMENT / f"systems_v143_smoke16_r{generation}"
    CLUSTER_RUNTIME = f"/workspace/nautilus-exp-end2end-agent-{SUFFIX}"
    OUTPUT_ROOT = f"/workspace/experiment-end2end-memory-agent-{SUFFIX}/runs"
    EVALUATOR_ROOT = (
        f"/workspace/experiment-end2end-leaf-official-evaluator-{SUFFIX}"
    )
    if generation >= 5:
        EVALUATOR_ROOT = (
            "/workspace/experiment-end2end-leaf-official-evaluator-"
            "v143-smoke16-r4"
        )
    EXPERIMENT_LABEL = f"experiment-end2end-memory-agent-{SUFFIX}"
    POD_NAME = f"mlevolve-leaf-gpt56sol-{SUFFIX}-dev"


if __name__ == "__main__":
    raise SystemExit(main())
