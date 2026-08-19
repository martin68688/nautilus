#!/usr/bin/env python3
"""Build an isolated six-hour v143 Dynamic full release from the r6 smoke runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import build_leaf_dynamic_replay_adaptation_v141_smoke as v141
import build_leaf_dynamic_replay_adaptation_v143_r2_smoke as v143r2
import build_leaf_dynamic_retrieval_v137_runtime as v137
import build_leaf_ten_system_gpt_v135_runtime as v135


EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
SUFFIX = "v143-full-r2"
MANIFEST_DIR = EXPERIMENT / "manifests_v143_full_r2"
SYSTEM_DIR = EXPERIMENT / "systems_v143_full_r2"
JOBS_DIR = EXPERIMENT / "jobs_v143_full_r2"
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v143-full-r2"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v143-full-r2/runs"
EVALUATOR_ROOT = (
    "/workspace/experiment-end2end-leaf-official-evaluator-v143-smoke16-r4"
)
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v143-full-r2"
JOB_NAME = "mlevolve-leaf-gpt56sol-v143-full-r2-dynamic-hybrid"


def full_spec() -> dict:
    return {
        "mode": "full",
        "suffix": SUFFIX,
        "kind": "pilot",
        "manifest_dir": MANIFEST_DIR,
        "system_dir": SYSTEM_DIR,
        "execution_name": "leaf_dynamic_full_manifest.json",
        "release_id": (
            "end2end-leaf-dynamic-replay-adaptation-gpt56sol-v143-full-r2"
        ),
        "cluster_runtime": CLUSTER_RUNTIME,
        "output_root": OUTPUT_ROOT,
        "evaluator_root": EVALUATOR_ROOT,
        "experiment_label": EXPERIMENT_LABEL,
        "workload": JOB_NAME,
        "stager": "unused-v143-full-r2-stager",
        "logical_run_id": (
            "e2e-full-leaf-dynamic-replay-adaptation-official-gpt56sol-"
            "v143-full-r2__leaf-classification__dynamic_hybrid__seed-1"
        ),
    }


def configure_builders() -> None:
    v143r2.configure_builder()
    v141.SUFFIX = SUFFIX
    v141.MANIFEST_DIR = MANIFEST_DIR
    v141.SYSTEM_DIR = SYSTEM_DIR
    v141.CLUSTER_RUNTIME = CLUSTER_RUNTIME
    v141.OUTPUT_ROOT = OUTPUT_ROOT
    v141.EVALUATOR_ROOT = EVALUATOR_ROOT
    v141.EXPERIMENT_LABEL = EXPERIMENT_LABEL
    v141.DEV_MEMORY_GIB = 64
    v141.GPU_RESOURCE_KEY = "nvidia.com/a100"
    v141.GPU_PRODUCT_CONSTRAINT = None
    v141.GPU_TYPE = "NVIDIA A100 family"
    v141.GPU_COUNT = 1
    v141.MEMORY_ROOT = v143r2.MEMORY_ROOT
    v141.MEMORY_BUNDLE = v143r2.MEMORY_BUNDLE
    v143r2.MANIFEST_DIR = MANIFEST_DIR
    v141.OVERLAY_FILES = tuple(
        dict.fromkeys(
            (
                *v141.OVERLAY_FILES,
                Path("mlevolve/agents/code_review_agent.py"),
                Path("mlevolve/agents/result_parse_agent.py"),
            )
        )
    )
    v141.TEST_FILES = tuple(
        dict.fromkeys(
            (
                *v141.TEST_FILES,
                Path("tests/test_leaf_v143_replay_target_recipe_projection.py"),
            )
        )
    )
    v141.TEST_SUPPORT_FILES = tuple(
        dict.fromkeys(
            (
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
                    "build_leaf_dynamic_replay_adaptation_v143_full.py"
                ),
            )
        )
    )
    v137.OVERLAY_FILES = v141.OVERLAY_FILES
    v137.TEST_FILES = v141.TEST_FILES
    v137.TEST_SUPPORT_FILES = v141.TEST_SUPPORT_FILES


def write_dynamic_config(output: Path) -> Path:
    target = output / SYSTEM_DIR / "dynamic_hybrid.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        "\n".join(
            [
                "# Full v143 uses the exact r6 Dynamic engine and v10-r8 memory semantics.",
                "# Only runtime/output identity and the six-hour full budget differ.",
                "extends: ../systems_v142_smoke16/dynamic_hybrid.yaml",
                "",
                "agent:",
                "  draft_role_policy:",
                "    replay_targets_path: "
                f"{v143r2.MEMORY_BUNDLE}/reports/leaf_official_replay_targets_v139.json",
                "    replay_runs_root: "
                f"{v143r2.MEMORY_BUNDLE}/run_artifacts",
                "",
                "external_skill_memory:",
                f"  recipe_sop_path: {v143r2.MEMORY_BUNDLE}/recipe/recipe_sops.json",
                f"  recipe_sop_file_sha256: {v143r2.RECIPE_FILE_SHA256}",
                f"  recipe_sop_bundle_sha256: {v143r2.RECIPE_BUNDLE_SHA256}",
                f"  recipe_evidence_path: {v143r2.MEMORY_BUNDLE}/recipe/evidence_manifest.json",
                f"  recipe_evidence_file_sha256: {v143r2.EVIDENCE_FILE_SHA256}",
                "  recipe_evidence_manifest_sha256: "
                f"{v143r2.EVIDENCE_MANIFEST_SHA256}",
                "  recipe_implementation_path: "
                f"{v143r2.MEMORY_BUNDLE}/recipe/implementation_capsules.json",
                "  transition_evidence_capsules_path: "
                f"{v143r2.MEMORY_BUNDLE}/recipe/transition_evidence_capsules.json",
                "  transition_evidence_capsules_sha256: "
                f"{v143r2.TRANSITION_EVIDENCE_SHA256}",
                "",
                "run_identity:",
                "  memory_version: "
                "leaf_llm_redistilled_v10_r8_replay_recipe_projection_v143_full_r2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def update_full_budget(output: Path) -> str:
    path = output / MANIFEST_DIR / "budget.json"
    payload = v135.read_json(path)
    payload["pilot"].update(
        {
            "agent_steps": v135.UNBOUNDED_WITHIN_SIX_HOURS,
            "agent_time_limit_seconds": 21_600,
            "cpu_count": 16,
            "execution_timeout_seconds": 21_600,
            "finalize_reserve_seconds": 900,
            "gpu_count": 1,
            "max_replacement_drafts": v135.UNBOUNDED_WITHIN_SIX_HOURS,
            "memory_gib": 64,
            "parallel_search_num": 1,
        }
    )
    payload["runtime"].update(
        {
            "gpu_resource_key": "nvidia.com/a100",
            "gpu_product_constraint": None,
            "gpu_type": "NVIDIA A100 family",
        }
    )
    return v135.write_hashed(path, payload, "manifest_hash")


def build(base_runtime: Path, output_runtime: Path, jobs_out: Path) -> dict:
    spec = full_spec()
    for path in (output_runtime, jobs_out):
        if path.exists():
            raise FileExistsError(f"fresh full output already exists: {path}")

    shutil.copytree(base_runtime.resolve(strict=True), output_runtime, symlinks=True)
    v137.remove_runtime_caches(output_runtime)
    for relative in (
        *v141.OVERLAY_FILES,
        *v141.TEST_FILES,
        *v141.TEST_SUPPORT_FILES,
    ):
        v135.copy_file(v141.REPO / relative, output_runtime / relative)

    dynamic = write_dynamic_config(output_runtime)
    bindings = v137.build_components(output_runtime, spec)
    bindings["budget_manifest_hash"] = update_full_budget(output_runtime)
    bindings["memory_bundles_manifest_hash"] = v143r2.update_memory_manifest(
        output_runtime
    )
    source_lock_hash, source_lock_count = v141.rewrite_source_lock(
        output_runtime, spec
    )
    bindings["source_lock_manifest_hash"] = source_lock_hash

    execution = v137.build_execution(spec, bindings)
    execution_hash = v135.write_hashed(
        output_runtime / MANIFEST_DIR / spec["execution_name"],
        execution,
        "manifest_hash",
    )

    jobs_out.mkdir(parents=True, exist_ok=False)
    budget = v135.read_json(output_runtime / MANIFEST_DIR / "budget.json")["pilot"]
    job = v137.build_job(spec, execution, budget)
    v135.write_json(jobs_out / f"{JOB_NAME}.yaml", job)

    return {
        "schema": "mlevolve_leaf_dynamic_replay_adaptation_v143_full_build_v1",
        "status": "complete",
        "mode": "full",
        "release_id": spec["release_id"],
        "git_head": v135.git_head(),
        "runtime_root": str(output_runtime),
        "cluster_runtime": CLUSTER_RUNTIME,
        "output_root": OUTPUT_ROOT,
        "evaluator_root": EVALUATOR_ROOT,
        "experiment_label": EXPERIMENT_LABEL,
        "logical_run_id": spec["logical_run_id"],
        "job_name": JOB_NAME,
        "source_lock_hash": source_lock_hash,
        "source_lock_file_count": source_lock_count,
        "execution_manifest_hash": execution_hash,
        "dynamic_config_sha256": v135.sha256_file(dynamic),
        "memory_bundle_root": v143r2.MEMORY_ROOT,
        "memory_bundle_manifest_sha256": v143r2.MANIFEST_SHA256,
        "agent_steps": budget["agent_steps"],
        "agent_time_limit_seconds": budget["agent_time_limit_seconds"],
        "max_replacement_drafts": budget["max_replacement_drafts"],
        "memory_gib": budget["memory_gib"],
        "gpu_count": budget["gpu_count"],
        "llm_model": v135.LLM_MODEL,
        "llm_base_url": v135.LLM_BASE_URL,
        "llm_secret_reference": v135.LLM_SECRET,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--jobs-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    configure_builders()
    receipt = build(args.base_runtime, args.output_runtime, args.jobs_out)
    v135.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
