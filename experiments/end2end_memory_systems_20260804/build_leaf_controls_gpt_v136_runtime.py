#!/usr/bin/env python3
"""Build the fresh v136 rerun packet for four failed v135 legacy controls.

This release changes no MLEvolve code and no control-system behavior.  It
repairs the launch-time input binding by pairing the frozen v23 controls with
their historical v2 Memory Bundle instead of the v8 atomic bundle used by the
latest Dynamic condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import build_leaf_ten_system_gpt_v135_runtime as v135


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
BASE_MANIFESTS = EXPERIMENT / "manifests_v135"
LEGACY_MANIFESTS = EXPERIMENT / "manifests_v23"
TARGET_MANIFESTS = EXPERIMENT / "manifests_v136"
TARGET_SYSTEMS = EXPERIMENT / "systems_v136"
EXECUTION_MANIFEST_NAME = "leaf_control_repair_manifest.json"
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v136"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v136/runs"
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v136"
EVALUATOR_ROOT = "/workspace/experiment-end2end-leaf-official-evaluator-v136"
SYSTEM_IDS = (
    "no_memory",
    "flat_retrieval",
    "sop_only",
    "runforest_only",
)
EXPECTED_LEGACY_MEMORY_MANIFEST_HASH = (
    "c46e1ee5e9c2a59330d1f7f44338f5a08e622ebd1715e814c06704d11ee579c6"
)


def configure_shared_builder() -> None:
    """Point v135 packaging helpers at fresh v136 identities."""

    v135.BASE_MANIFESTS = BASE_MANIFESTS
    v135.TARGET_MANIFESTS = TARGET_MANIFESTS
    v135.TARGET_SYSTEMS = TARGET_SYSTEMS
    v135.EXECUTION_MANIFEST_NAME = EXECUTION_MANIFEST_NAME
    v135.CLUSTER_RUNTIME = CLUSTER_RUNTIME
    v135.OUTPUT_ROOT = OUTPUT_ROOT
    v135.EXPERIMENT_LABEL = EXPERIMENT_LABEL
    v135.SYSTEM_IDS = SYSTEM_IDS
    v135.BATCH_1 = SYSTEM_IDS
    v135.BATCH_2 = ()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): v135.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def build_components(output: Path) -> dict[str, str]:
    """Freeze v135 runtime/evaluator surfaces with the historical v2 bundle."""

    base = output / BASE_MANIFESTS
    legacy = output / LEGACY_MANIFESTS
    manifests = output / TARGET_MANIFESTS
    manifests.mkdir(parents=True, exist_ok=False)

    for name in ("schemas.json", "tasks.json", "leaf_official_replay_targets.json"):
        v135.copy_file(base / name, manifests / name)
    v135.copy_file(legacy / "memory_bundles.json", manifests / "memory_bundles.json")
    memory = v135.read_json(manifests / "memory_bundles.json")
    if memory.get("manifest_hash") != EXPECTED_LEGACY_MEMORY_MANIFEST_HASH:
        raise ValueError("historical v2 Memory Bundle manifest identity changed")
    task_bundle = dict((memory.get("task_bundles") or {}).get("leaf-classification") or {})
    if task_bundle.get("bundle_version") != "v2":
        raise ValueError("v136 controls must bind the historical v2 Memory Bundle")

    budget = v135.read_json(base / "budget.json")
    budget["pilot"].update(
        {
            "agent_steps": v135.UNBOUNDED_WITHIN_SIX_HOURS,
            "agent_time_limit_seconds": 21_600,
            "cpu_count": 16,
            "execution_timeout_seconds": 21_600,
            "finalize_reserve_seconds": 900,
            "gpu_count": 1,
            "initial_drafts": 3,
            "max_replacement_drafts": v135.UNBOUNDED_WITHIN_SIX_HOURS,
            "memory_gib": 64,
            "parallel_search_num": 1,
        }
    )
    budget_hash = v135.write_hashed(manifests / "budget.json", budget, "manifest_hash")

    evaluators = v135.read_json(base / "evaluators.json")
    evaluators["formal_releases_root"] = EVALUATOR_ROOT
    evaluators["tasks"]["leaf-classification"].update(
        {
            "release_root": f"{EVALUATOR_ROOT}/leaf-classification/release",
            "terminal_evaluator_spec": "deferred official Kaggle v1",
        }
    )
    evaluator_hash = v135.write_hashed(
        manifests / "evaluators.json", evaluators, "manifest_hash"
    )

    previous_systems = v135.read_json(base / "systems.json")
    system_rows = {
        str(row["system_id"]): dict(row)
        for row in previous_systems.get("systems") or []
    }
    systems = []
    for system_id in SYSTEM_IDS:
        row = system_rows[system_id]
        config = output / TARGET_SYSTEMS / f"{system_id}.yaml"
        row["config_path"] = f"systems_v136/{system_id}.yaml"
        row["config_sha256"] = v135.sha256_file(config)
        row["description"] = (
            f"Frozen v23 {system_id} control with its historical v2 Memory Bundle"
        )
        systems.append(row)
    systems_manifest = {
        "schema": "mlevolve_end2end_systems_manifest_v1",
        "experimental_axis": (
            "byte-identical latest v135 MLEvolve runtime and frozen v23 control "
            "semantics; launch binding restored from v8 atomic to historical v2 bundle"
        ),
        "system_count": len(systems),
        "systems": systems,
        "manifest_hash": "",
    }
    systems_hash = v135.write_hashed(
        manifests / "systems.json", systems_manifest, "manifest_hash"
    )

    source_revision = v135.git_head()
    source_lock = v135.build_source_lock(
        output, source_revision=source_revision, manifests=TARGET_MANIFESTS
    )
    source_lock["release_id"] = "end2end-leaf-control-repair-gpt56sol-v136"
    source_lock_hash = v135.write_hashed(
        manifests / "source_lock.json", source_lock, "manifest_hash"
    )
    return {
        "budget_manifest_hash": budget_hash,
        "evaluators_manifest_hash": evaluator_hash,
        "memory_bundles_manifest_hash": EXPECTED_LEGACY_MEMORY_MANIFEST_HASH,
        "schemas_manifest_hash": v135.read_json(manifests / "schemas.json")["manifest_hash"],
        "source_lock_manifest_hash": source_lock_hash,
        "systems_manifest_hash": systems_hash,
        "tasks_manifest_hash": v135.read_json(manifests / "tasks.json")["manifest_hash"],
    }


def build_execution(bindings: Mapping[str, str]) -> dict[str, Any]:
    runs = []
    for position, system_id in enumerate(SYSTEM_IDS):
        row = {
            "task_id": "leaf-classification",
            "system_id": system_id,
            "seed": 1,
            "logical_run_id": (
                "e2e-pilot-leaf-control-repair-official-gpt56sol-v136__"
                f"leaf-classification__{system_id}__seed-1"
            ),
            "formal_result_eligible": True,
            "exploratory_pilot": True,
            "launch_position": position,
            "task_launch_position": position,
            "bindings": dict(bindings),
            "row_hash": "",
        }
        row["row_hash"] = v135.payload_hash(row, "row_hash")
        runs.append(row)
    manifest = {
        "schema": "mlevolve_end2end_execution_manifest_v1",
        "release_id": "end2end-leaf-control-repair-official-gpt56sol-v136",
        "kind": "pilot",
        "comparison_baseline_release_id": "end2end-leaf-ten-system-official-gpt56sol-v135",
        "seed": 1,
        "task_ids": ["leaf-classification"],
        "system_ids": list(SYSTEM_IDS),
        "run_count": len(runs),
        "first_parallel_batch": list(SYSTEM_IDS),
        "second_parallel_batch": [],
        "launch_order_randomization": "fresh repair rerun of four retained v135 failures",
        "formal_result_eligible": True,
        "exploratory_pilot": True,
        "statistical_significance_claim_allowed": False,
        "bindings": dict(bindings),
        "runs": runs,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = v135.payload_hash(manifest, "manifest_hash")
    return manifest


def replace_release(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("v135", "v136")
    if isinstance(value, list):
        return [replace_release(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_release(item) for key, item in value.items()}
    return value


def build_job(*, manifest: Mapping[str, Any], index: int, system_id: str) -> dict[str, Any]:
    job = v135.build_job(
        manifest=manifest, index=index, system_id=system_id, batch=1
    )
    job = replace_release(job)
    job["metadata"]["annotations"]["mlevolve.ai/repair-scope"] = (
        "bundle-binding-only-v8-to-historical-v2"
    )
    job["metadata"]["annotations"]["mlevolve.ai/engine-code-change"] = "none"
    return job


def build(
    *, base_runtime: Path, output_runtime: Path, manifests_out: Path, jobs_out: Path
) -> dict[str, Any]:
    configure_shared_builder()
    for path in (output_runtime, manifests_out, jobs_out):
        if path.exists():
            raise FileExistsError(f"fresh v136 output already exists: {path}")
    shutil.copytree(base_runtime.resolve(strict=True), output_runtime, symlinks=True)
    before_engine = tree_hashes(base_runtime.resolve(strict=True) / "mlevolve")
    v135.overlay_latest_release_files(output_runtime)
    after_engine = tree_hashes(output_runtime / "mlevolve")
    if after_engine != before_engine:
        raise ValueError("v136 bundle-binding repair changed MLEvolve code")

    bindings = build_components(output_runtime)
    execution = build_execution(bindings)
    runtime_manifests = output_runtime / TARGET_MANIFESTS
    execution_hash = v135.write_hashed(
        runtime_manifests / EXECUTION_MANIFEST_NAME,
        execution,
        "manifest_hash",
    )
    shutil.copytree(runtime_manifests, manifests_out)

    jobs_out.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(execution["runs"]):
        system_id = str(row["system_id"])
        job = build_job(manifest=execution, index=index, system_id=system_id)
        v135.write_json(
            jobs_out / f"{job['metadata']['name']}.yaml",
            job,
        )

    source_lock = v135.read_json(runtime_manifests / "source_lock.json")
    atomic = output_runtime / "mlevolve/agents/atomic_actuation.py"
    receipt = {
        "schema": "mlevolve_leaf_control_bundle_binding_repair_build_v1",
        "status": "complete",
        "release_version": 136,
        "git_head": source_lock["git_head"],
        "runtime_root": str(output_runtime),
        "runtime_file_count": sum(
            1 for path in output_runtime.rglob("*") if path.is_file()
        ),
        "source_lock_hash": source_lock["manifest_hash"],
        "source_lock_file_count": len(source_lock["files"]),
        "execution_manifest_hash": execution_hash,
        "atomic_actuation_sha256": v135.sha256_file(atomic),
        "mlevolve_tree_file_count": len(after_engine),
        "mlevolve_tree_unchanged_from_v135_runtime": True,
        "system_count": len(SYSTEM_IDS),
        "systems": list(SYSTEM_IDS),
        "agent_time_limit_seconds": 21_600,
        "agent_steps_nonbinding_sentinel": v135.UNBOUNDED_WITHIN_SIX_HOURS,
        "memory_bundle_manifest_hash": EXPECTED_LEGACY_MEMORY_MANIFEST_HASH,
        "memory_bundle_version": "v2",
        "llm_model": v135.LLM_MODEL,
        "llm_base_url": v135.LLM_BASE_URL,
        "llm_secret_reference": v135.LLM_SECRET,
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--manifests-out", required=True, type=Path)
    parser.add_argument("--jobs-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = build(
        base_runtime=args.base_runtime,
        output_runtime=args.output_runtime,
        manifests_out=args.manifests_out,
        jobs_out=args.jobs_out,
    )
    v135.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
