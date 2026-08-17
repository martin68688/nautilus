#!/usr/bin/env python3
"""Build the fresh v127 Leaf Replay GPT runtime from the immutable v122 base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
BASE_MANIFESTS = EXPERIMENT / "manifests_v122"
TARGET_MANIFESTS = EXPERIMENT / "manifests_v127"
EXECUTION_MANIFEST_NAME = "leaf_replay_gpt56sol_smoke_manifest.json"
OVERLAY_FILES = (
    EXPERIMENT / "run_assignment.py",
    EXPERIMENT / "stage_leaf_official_evaluator_v127.py",
    EXPERIMENT / "systems_v123/dynamic_hybrid.yaml",
    EXPERIMENT / "systems_v127/dynamic_hybrid.yaml",
    Path("mlevolve/analysis/adoption_tracker.py"),
    Path("mlevolve/analysis/adoption_verifier_smoke.py"),
    Path("mlevolve/config/__init__.py"),
    Path("mlevolve/config/config.yaml"),
    Path("mlevolve/config/config_leaf_official.yaml"),
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_hashed(path: Path, payload: dict[str, Any], field: str) -> str:
    payload[field] = payload_hash(payload, field)
    write_json(path, payload)
    return str(payload[field])


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def copy_overlay(output: Path) -> None:
    for relative in OVERLAY_FILES:
        source = (REPO / relative).resolve(strict=True)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        shutil.copy2(source, destination)


def build_source_lock(output: Path, head: str) -> dict[str, Any]:
    excluded = {
        (TARGET_MANIFESTS / "source_lock.json").as_posix(),
        (TARGET_MANIFESTS / EXECUTION_MANIFEST_NAME).as_posix(),
        "SOURCE_FILES.sha256",
    }
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {
        "schema": "mlevolve_end2end_source_lock_v1",
        "git_head": head,
        "git_head_is_not_sufficient_identity": True,
        "complete_runtime_file_hash_lock": True,
        "control_file_exclusions": sorted(excluded),
        "files": files,
        "manifest_hash": "",
    }


def build_runtime(base: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"fresh v127 runtime already exists: {output}")
    shutil.copytree(base.resolve(strict=True), output, symlinks=True)
    copy_overlay(output)

    base_manifests = output / BASE_MANIFESTS
    manifests = output / TARGET_MANIFESTS
    manifests.mkdir(parents=True, exist_ok=False)
    for name in (
        "evaluators.json",
        "memory_bundles.json",
        "schemas.json",
        "tasks.json",
        "leaf_official_replay_targets.json",
    ):
        shutil.copy2(base_manifests / name, manifests / name)

    budget = read_json(base_manifests / "budget.json")
    budget["smoke"].update(
        {
            "agent_steps": 16,
            "agent_time_limit_seconds": 18000,
            "cpu_count": 16,
            "execution_timeout_seconds": 3600,
            "finalize_reserve_seconds": 600,
            "gpu_count": 1,
            "initial_drafts": 3,
            "max_replacement_drafts": 0,
            "memory_gib": 32,
            "parallel_search_num": 1,
        }
    )
    budget["runtime"].update(
        {
            "solver_model_id": "gpt-5.6-sol-openai-compatible-solver",
            "solver_model_revision": "sha256:"
            + hashlib.sha256(
                b"https://apizh.net/v1|gpt-5.6-sol|chat-completions"
            ).hexdigest(),
        }
    )
    budget_hash = write_hashed(manifests / "budget.json", budget, "manifest_hash")

    evaluators = read_json(manifests / "evaluators.json")
    evaluators["formal_releases_root"] = (
        "/workspace/experiment-end2end-leaf-official-evaluator-v127"
    )
    evaluators["tasks"]["leaf-classification"].update(
        {
            "release_root": (
                "/workspace/experiment-end2end-leaf-official-evaluator-v127/"
                "leaf-classification/release"
            ),
            "terminal_evaluator_spec": "deferred official Kaggle v1",
        }
    )
    evaluator_hash = write_hashed(
        manifests / "evaluators.json", evaluators, "manifest_hash"
    )

    system_config = output / EXPERIMENT / "systems_v127/dynamic_hybrid.yaml"
    systems = {
        "schema": "mlevolve_end2end_systems_manifest_v1",
        "experimental_axis": (
            "Leaf v122 Replay Research alignment behavior on the full official test "
            "set with every live LLM role routed to GPT-5.6 Sol"
        ),
        "system_count": 1,
        "systems": [
            {
                "system_id": "dynamic_hybrid",
                "kind": "internal_exploratory",
                "label": "S5-v127-leaf-replay-research-official-gpt56sol",
                "description": (
                    "v122 Replay Research plus the v122 alignment gate, v123 full "
                    "official-test output contract, and GPT-5.6 Sol for code, feedback, "
                    "Search/Grep/Judge/Resolver, Strategy, normalization, and verifier roles"
                ),
                "limitation": "single exploratory 16-step A100 online smoke",
                "primary_reference": None,
                "config_path": "systems_v127/dynamic_hybrid.yaml",
                "config_sha256": sha256_file(system_config),
            }
        ],
        "manifest_hash": "",
    }
    systems_hash = write_hashed(
        manifests / "systems.json", systems, "manifest_hash"
    )

    head = git_head()
    source_lock = build_source_lock(output, head)
    source_lock_hash = write_hashed(
        manifests / "source_lock.json", source_lock, "manifest_hash"
    )

    component_hashes = {
        "budget_manifest_hash": budget_hash,
        "evaluators_manifest_hash": evaluator_hash,
        "memory_bundles_manifest_hash": read_json(
            manifests / "memory_bundles.json"
        )["manifest_hash"],
        "schemas_manifest_hash": read_json(manifests / "schemas.json")[
            "manifest_hash"
        ],
        "source_lock_manifest_hash": source_lock_hash,
        "systems_manifest_hash": systems_hash,
        "tasks_manifest_hash": read_json(manifests / "tasks.json")["manifest_hash"],
    }
    logical_run_id = (
        "e2e-smoke-leaf-replay-research-official-gpt56sol-v127__"
        "leaf-classification__dynamic_hybrid__seed-1"
    )
    row = {
        "task_id": "leaf-classification",
        "system_id": "dynamic_hybrid",
        "seed": 1,
        "logical_run_id": logical_run_id,
        "formal_result_eligible": False,
        "exploratory_pilot": True,
        "launch_position": 0,
        "task_launch_position": 0,
        "bindings": component_hashes,
        "row_hash": "",
    }
    row["row_hash"] = payload_hash(row, "row_hash")
    execution = {
        "schema": "mlevolve_end2end_execution_manifest_v1",
        "release_id": "end2end-leaf-replay-research-official-gpt56sol-v127-smoke",
        "kind": "smoke",
        "comparison_baseline_release_id": "end2end-agent-v3",
        "seed": 1,
        "task_ids": ["leaf-classification"],
        "system_ids": ["dynamic_hybrid"],
        "run_count": 1,
        "first_parallel_batch": ["dynamic_hybrid"],
        "launch_order_randomization": "single fresh Dynamic Leaf v127 smoke",
        "formal_result_eligible": False,
        "exploratory_pilot": True,
        "statistical_significance_claim_allowed": False,
        "bindings": component_hashes,
        "runs": [row],
        "manifest_hash": "",
    }
    execution_hash = write_hashed(
        manifests / EXECUTION_MANIFEST_NAME, execution, "manifest_hash"
    )
    return {
        "schema": "mlevolve_leaf_replay_gpt_v127_runtime_build_v1",
        "status": "complete",
        "git_head": head,
        "runtime_root": str(output),
        "execution_manifest": str(manifests / EXECUTION_MANIFEST_NAME),
        "execution_manifest_hash": execution_hash,
        "source_lock_hash": source_lock_hash,
        "logical_run_id": logical_run_id,
        "agent_steps": 16,
        "official_test_mode": True,
        "llm_model": "gpt-5.6-sol",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = build_runtime(args.base_runtime, args.output_runtime)
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
