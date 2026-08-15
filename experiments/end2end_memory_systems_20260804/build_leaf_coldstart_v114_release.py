#!/usr/bin/env python3
"""Build the immutable v114 atomic/repair alias Resolver smoke release."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import prepare_leaf_strategy_active_v74 as builder


ATOMIC_BUNDLE_ID = "end2end-leaf-atomic-recipe-runforest-v8"
ATOMIC_BUNDLE_SHA256 = (
    "fa697bbd5fc47eb728ba13a63d693bc4777b47c6b5c984f653e89041871aa0bb"
)
ATOMIC_BUNDLE_ROOT = (
    "/workspace/experiment-end2end-memory-agent-v89/"
    "memory-leaf-atomic-v8/leaf-classification"
)
TRANSITION_CAPSULE_SOURCE = Path(
    "/private/tmp/leaf-v114-transition-evidence-capsules.json"
)
TRANSITION_CAPSULE_SHA256 = (
    "ae8a06a6a4ef896e09516ad2d953462a601c1ea5871bf9707fdd20ff7dea8559"
)
TRANSITION_CAPSULE_TARGET = Path(
    "experiments/end2end_memory_systems_20260804/transition_evidence_v114/"
    "transition_evidence_capsules.json"
)


_extract_overlay = builder.extract_overlay


def _extract_overlay_with_transition_capsule(overlay: Path, destination: Path) -> None:
    """Embed the independently materialized v114 capsule before source-locking."""

    _extract_overlay(overlay, destination)
    source = TRANSITION_CAPSULE_SOURCE.resolve(strict=True)
    if builder.sha256_file(source) != TRANSITION_CAPSULE_SHA256:
        raise ValueError("v114 transition evidence capsule SHA-256 mismatch")
    target = destination / TRANSITION_CAPSULE_TARGET
    if target.exists():
        raise FileExistsError(f"overlay unexpectedly contains {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _sixteen_step_smoke_budget(budget: dict) -> None:
    profile = dict(budget["smoke"])
    profile.update(
        {
            "agent_steps": 16,
            "agent_time_limit_seconds": 21600,
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
    budget["smoke"] = profile
    budget["manifest_hash"] = builder.payload_hash(budget, "manifest_hash")


def _append_frozen_v114_contract() -> None:
    locked_options = {
        "--release-version": "114",
        "--source-manifest-version": "113",
        "--source-smoke-manifest-name": "leaf_strategy_active_smoke_manifest.json",
        "--release-slug": "leaf-latest-coldstart-three-role-alias-resolver",
        "--experimental-axis": (
            "Leaf deterministic post-Judge Evidence Resolver with exact "
            "atomic-transition and repair-claim source-transition aliases"
        ),
        "--system-description": (
            "v113 Evidence Resolver plus a stage-safe, hash-bound atomic/repair "
            "alias bridge; no Compatibility Preflight"
        ),
        "--required-memory-bundle-id": ATOMIC_BUNDLE_ID,
        "--required-memory-bundle-manifest-sha256": ATOMIC_BUNDLE_SHA256,
        "--required-memory-bundle-root": ATOMIC_BUNDLE_ROOT,
        "--required-formal-debug-clause-count": "296",
    }
    supplied = set(sys.argv[1:]) & set(locked_options)
    if supplied:
        raise ValueError(
            "v114 release contract options are immutable: "
            + ", ".join(sorted(supplied))
        )
    for option, value in locked_options.items():
        sys.argv.extend([option, value])


builder._six_hour_smoke_budget = _sixteen_step_smoke_budget
builder.extract_overlay = _extract_overlay_with_transition_capsule


if __name__ == "__main__":
    _append_frozen_v114_contract()
    raise SystemExit(builder.main())
