#!/usr/bin/env python3
"""Build the fresh v146 equal-branch Dynamic 16-step Dev-Pod smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_leaf_dynamic_two_role_v144_smoke as v144
import build_leaf_ten_system_gpt_v135_runtime as v135


EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
SUFFIX = "v146-smoke16"
MANIFEST_DIR = EXPERIMENT / "manifests_v146_smoke16"
SYSTEM_DIR = EXPERIMENT / "systems_v146_smoke16"
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v146-smoke16"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v146-smoke16/runs"
EVALUATOR_ROOT = (
    "/workspace/experiment-end2end-leaf-official-evaluator-v146-smoke16"
)
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v146-smoke16"
POD_NAME = "mlevolve-leaf-gpt56sol-v146-smoke16-dev"
MEMORY_ROOT = v144.MEMORY_ROOT
MEMORY_BUNDLE = v144.MEMORY_BUNDLE


def _dedupe(paths):
    return tuple(dict.fromkeys(paths))


def configure_builder() -> None:
    v144.SUFFIX = SUFFIX
    v144.MANIFEST_DIR = MANIFEST_DIR
    v144.SYSTEM_DIR = SYSTEM_DIR
    v144.CLUSTER_RUNTIME = CLUSTER_RUNTIME
    v144.OUTPUT_ROOT = OUTPUT_ROOT
    v144.EVALUATOR_ROOT = EVALUATOR_ROOT
    v144.EXPERIMENT_LABEL = EXPERIMENT_LABEL
    v144.POD_NAME = POD_NAME
    v144.configure_builder()
    v144.base.OVERLAY_FILES = _dedupe(
        (
            *v144.base.OVERLAY_FILES,
            Path("mlevolve/config/__init__.py"),
            Path("mlevolve/config/config.yaml"),
            Path("mlevolve/engine/agent_search.py"),
            Path("mlevolve/engine/conditions.py"),
            Path("mlevolve/engine/node_selection.py"),
            Path("mlevolve/engine/role_balance.py"),
        )
    )
    v144.base.TEST_FILES = _dedupe(
        (
            *v144.base.TEST_FILES,
            Path("tests/test_branch_fair_scheduler_v146.py"),
            Path("tests/test_two_role_coverage_fusion_v144.py"),
        )
    )
    v144.base.TEST_SUPPORT_FILES = _dedupe(
        (
            *v144.base.TEST_SUPPORT_FILES,
            Path(
                "experiments/end2end_memory_systems_20260804/"
                "build_leaf_dynamic_branch_fair_v146_smoke.py"
            ),
        )
    )
    v144.base.write_dynamic_config = write_dynamic_config
    v144.base.build_dev_pod = build_dev_pod


def write_dynamic_config(output: Path) -> Path:
    target = output / SYSTEM_DIR / "dynamic_hybrid.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        "\n".join(
            [
                "# v146 changes only post-coverage branch resource allocation.",
                "# Replay, independent Novel and first Fusion receive equal attempt depth.",
                "extends: ../systems_v145_smoke16/dynamic_hybrid.yaml",
                "",
                "agent:",
                "  max_fusion_drafts: 1",
                "  fusion_vs_evolution_prob: 0.0",
                "  draft_role_policy:",
                "    equal_branch_allocation_after_coverage: true",
                "    single_coverage_synthesis_only: true",
                "",
                "run_identity:",
                "  memory_version: leaf_llm_redistilled_v10_r8_dynamic_"
                "replay_novel_fusion_equal_branch_v146_smoke16",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def build_dev_pod(manifest_hash: str, source_lock_hash: str) -> dict:
    original = getattr(v144.base, "_v144_original_build_dev_pod", None)
    if original is None:
        raise RuntimeError("v144 original Pod builder was not preserved")
    pod = original(manifest_hash, source_lock_hash)
    pod["metadata"]["annotations"]["mlevolve.ai/purpose"] = (
        "sixteen-step-two-role-one-fusion-equal-branch-smoke"
    )
    return pod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--pod-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    if not hasattr(v144.base, "_v144_original_build_dev_pod"):
        v144.base._v144_original_build_dev_pod = v144.base.build_dev_pod
    configure_builder()
    receipt = v144.base.build(
        args.base_runtime,
        args.output_runtime,
        args.pod_out,
    )
    receipt.update(
        {
            "schema": "mlevolve_leaf_dynamic_branch_fair_v146_smoke_build_v1",
            "draft_roles": ["memory_reproduction", "novel_exploration"],
            "coverage_min_valid_per_role": 1,
            "first_fusion_trigger": "two_role_coverage_milestone_v1",
            "post_coverage_scheduler": "least_cumulative_branch_attempts_v1",
            "fair_branches": ["replay", "novel", "fusion"],
            "failed_debug_inflight_attempts_are_charged": True,
            "within_branch_selection": "legacy_uct_topk",
            "max_fusion_drafts": 1,
            "stagnant_cross_branch_fusion_enabled": False,
            "coldstart_baseline_enabled": False,
            "fresh_dev_pod": True,
            "memory_bundle_root": MEMORY_ROOT,
            "memory_bundle_manifest_sha256": v144.v143.MANIFEST_SHA256,
        }
    )
    v135.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
