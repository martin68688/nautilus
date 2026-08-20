#!/usr/bin/env python3
"""Build the v145 parent-code-aware Fusion smoke for the existing v144 Dev Pod."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_leaf_dynamic_two_role_v144_smoke as v144
import build_leaf_ten_system_gpt_v135_runtime as v135


EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
SUFFIX = "v145-smoke16"
MANIFEST_DIR = EXPERIMENT / "manifests_v145_smoke16"
SYSTEM_DIR = EXPERIMENT / "systems_v145_smoke16"
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v145-smoke16"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v145-smoke16/runs"
EVALUATOR_ROOT = "/workspace/experiment-end2end-leaf-official-evaluator-v145-smoke16"
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v145-smoke16"
EXISTING_POD_NAME = "mlevolve-leaf-gpt56sol-v144-smoke16-r2-dev"
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
    v144.POD_NAME = EXISTING_POD_NAME
    v144.configure_builder()
    v144.base.TEST_SUPPORT_FILES = _dedupe(
        (
            *v144.base.TEST_SUPPORT_FILES,
            Path(
                "experiments/end2end_memory_systems_20260804/"
                "build_leaf_dynamic_parent_code_fusion_v145_smoke.py"
            ),
        )
    )
    v144.base.write_dynamic_config = write_dynamic_config


def write_dynamic_config(output: Path) -> Path:
    target = output / SYSTEM_DIR / "dynamic_hybrid.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        "\n".join(
            [
                "# v145 changes only the protected two-role Fusion generation contract.",
                "# Replay and Novel complete source programs are visible to free Fusion.",
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
                "  memory_version: leaf_llm_redistilled_v10_r8_dynamic_replay_novel_parent_code_fusion_v145_smoke16",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


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
            "schema": "mlevolve_leaf_dynamic_parent_code_fusion_v145_smoke_build_v1",
            "draft_roles": ["memory_reproduction", "novel_exploration"],
            "coverage_min_valid_per_role": 1,
            "first_fusion_trigger": "two_role_coverage_milestone_v1",
            "fusion_parent_code_visible": True,
            "fusion_fresh_rewrite_required": False,
            "fusion_non_degradation_required": False,
            "coldstart_baseline_enabled": False,
            "existing_dev_pod_reused": True,
            "existing_dev_pod_name": EXISTING_POD_NAME,
            "memory_bundle_root": MEMORY_ROOT,
            "memory_bundle_manifest_sha256": v144.v143.MANIFEST_SHA256,
        }
    )
    v135.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
