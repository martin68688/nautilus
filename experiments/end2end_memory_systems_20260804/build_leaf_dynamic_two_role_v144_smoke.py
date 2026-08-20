#!/usr/bin/env python3
"""Build the isolated 16-step Dynamic Replay+Novel v144 Dev-Pod smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_leaf_dynamic_replay_adaptation_v141_smoke as base
import build_leaf_dynamic_replay_adaptation_v143_r2_smoke as v143
import build_leaf_ten_system_gpt_v135_runtime as v135


EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
SUFFIX = "v144-smoke16"
MANIFEST_DIR = EXPERIMENT / "manifests_v144_smoke16"
SYSTEM_DIR = EXPERIMENT / "systems_v144_smoke16"
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v144-smoke16"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v144-smoke16/runs"
EVALUATOR_ROOT = "/workspace/experiment-end2end-leaf-official-evaluator-v144-smoke16"
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v144-smoke16"
POD_NAME = "mlevolve-leaf-gpt56sol-v144-smoke16-dev"
MEMORY_ROOT = v143.MEMORY_ROOT
MEMORY_BUNDLE = v143.MEMORY_BUNDLE


def _dedupe(paths):
    return tuple(dict.fromkeys(paths))


def configure_builder() -> None:
    v143.configure_builder()
    base.SUFFIX = SUFFIX
    base.MANIFEST_DIR = MANIFEST_DIR
    base.SYSTEM_DIR = SYSTEM_DIR
    base.CLUSTER_RUNTIME = CLUSTER_RUNTIME
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.EVALUATOR_ROOT = EVALUATOR_ROOT
    base.EXPERIMENT_LABEL = EXPERIMENT_LABEL
    base.POD_NAME = POD_NAME
    base.DEV_MEMORY_GIB = 32
    base.GPU_RESOURCE_KEY = "nvidia.com/a100"
    base.GPU_PRODUCT_CONSTRAINT = None
    base.GPU_TYPE = "NVIDIA A100 family"
    base.GPU_COUNT = 1
    base.MEMORY_ROOT = MEMORY_ROOT
    base.MEMORY_BUNDLE = MEMORY_BUNDLE
    base.OVERLAY_FILES = _dedupe(
        (
            *base.OVERLAY_FILES,
            Path("mlevolve/run.py"),
            Path("mlevolve/config/config.yaml"),
        )
    )
    base.TEST_FILES = _dedupe(
        (
            *base.TEST_FILES,
            Path("tests/test_two_role_coverage_fusion_v144.py"),
        )
    )
    base.TEST_SUPPORT_FILES = _dedupe(
        (
            *base.TEST_SUPPORT_FILES,
            Path(
                "experiments/end2end_memory_systems_20260804/"
                "build_leaf_dynamic_two_role_v144_smoke.py"
            ),
        )
    )
    base.write_dynamic_config = write_dynamic_config
    base.update_runtime_budget = update_runtime_budget
    base.update_memory_manifest = update_memory_manifest
    base.build_dev_pod = build_dev_pod


def write_dynamic_config(output: Path) -> Path:
    target = output / SYSTEM_DIR / "dynamic_hybrid.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        "\n".join(
            [
                "# v144 removes only the in-run Cold Start control branch.",
                "# Replay and independent Novel must each score before one protected Fusion.",
                "extends: ../systems_v143_full_r2/dynamic_hybrid.yaml",
                "",
                "agent:",
                "  initial_drafts: 2",
                "  branch_fusion_trigger_prob: 1.0",
                "  search:",
                "    num_drafts: 2",
                "  draft_role_policy:",
                "    roles:",
                "      - memory_reproduction",
                "      - novel_exploration",
                "    ensure_valid_candidate_per_role: true",
                "    role_balance_min_valid_candidates: 1",
                "    cross_role_synthesis_after_balance: true",
                "    cross_role_synthesis_on_coverage: true",
                "",
                "run_identity:",
                "  memory_version: leaf_llm_redistilled_v10_r8_"
                "dynamic_replay_novel_coverage_fusion_v144_smoke16",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def update_memory_manifest(output: Path) -> str:
    previous = v143.MANIFEST_DIR
    try:
        v143.MANIFEST_DIR = MANIFEST_DIR
        return v143.update_memory_manifest(output)
    finally:
        v143.MANIFEST_DIR = previous


def update_runtime_budget(output: Path) -> str:
    """Keep the frozen command-line draft count aligned with the two-role policy."""

    path = output / MANIFEST_DIR / "budget.json"
    payload = v135.read_json(path)
    payload["smoke"]["initial_drafts"] = 2
    return v135.write_hashed(path, payload, "manifest_hash")


def configure_generation(generation: int) -> None:
    """Advance all mutable identities after a pre-launch staging failure."""

    if generation < 1:
        raise ValueError("generation must be positive")
    if generation == 1:
        return
    revision = f"-r{generation}"
    manifest_revision = f"_r{generation}"
    global SUFFIX, MANIFEST_DIR, SYSTEM_DIR, CLUSTER_RUNTIME
    global OUTPUT_ROOT, EVALUATOR_ROOT, EXPERIMENT_LABEL, POD_NAME
    SUFFIX = f"v144-smoke16{revision}"
    MANIFEST_DIR = EXPERIMENT / f"manifests_v144_smoke16{manifest_revision}"
    SYSTEM_DIR = EXPERIMENT / f"systems_v144_smoke16{manifest_revision}"
    CLUSTER_RUNTIME = f"/workspace/nautilus-exp-end2end-agent-{SUFFIX}"
    OUTPUT_ROOT = f"/workspace/experiment-end2end-memory-agent-{SUFFIX}/runs"
    EVALUATOR_ROOT = (
        f"/workspace/experiment-end2end-leaf-official-evaluator-{SUFFIX}"
    )
    EXPERIMENT_LABEL = f"experiment-end2end-memory-agent-{SUFFIX}"
    POD_NAME = f"mlevolve-leaf-gpt56sol-{SUFFIX}-dev"


def build_dev_pod(manifest_hash: str, source_lock_hash: str) -> dict:
    original = getattr(base, "_v144_original_build_dev_pod", None)
    if original is None:
        raise RuntimeError("v144 original Pod builder was not preserved")
    pod = original(manifest_hash, source_lock_hash)
    pod["metadata"]["annotations"]["mlevolve.ai/purpose"] = (
        "sixteen-step-two-role-replay-novel-coverage-fusion-smoke"
    )
    return pod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--pod-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--generation", type=int, default=1)
    args = parser.parse_args()

    if not hasattr(base, "_v144_original_build_dev_pod"):
        base._v144_original_build_dev_pod = base.build_dev_pod
    configure_generation(args.generation)
    configure_builder()
    receipt = base.build(args.base_runtime, args.output_runtime, args.pod_out)
    receipt.update(
        {
            "schema": "mlevolve_leaf_dynamic_two_role_v144_smoke_build_v1",
            "draft_roles": ["memory_reproduction", "novel_exploration"],
            "coverage_min_valid_per_role": 1,
            "first_fusion_trigger": "two_role_coverage_milestone_v1",
            "coldstart_baseline_enabled": False,
            "memory_bundle_root": MEMORY_ROOT,
            "memory_bundle_manifest_sha256": v143.MANIFEST_SHA256,
        }
    )
    v135.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
