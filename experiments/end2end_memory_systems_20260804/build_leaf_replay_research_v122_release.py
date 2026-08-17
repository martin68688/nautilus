#!/usr/bin/env python3
"""Reissue v121 with fail-closed modified-replay metric alignment."""

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
    "/private/tmp/leaf-v116-transition-evidence-capsules.json"
)
TRANSITION_CAPSULE_SHA256 = (
    "ae8a06a6a4ef896e09516ad2d953462a601c1ea5871bf9707fdd20ff7dea8559"
)
TRANSITION_CAPSULE_TARGET = Path(
    "experiments/end2end_memory_systems_20260804/transition_evidence_v122/"
    "transition_evidence_capsules.json"
)


_extract_overlay = builder.extract_overlay


def _extract_overlay_with_transition_capsule(overlay: Path, destination: Path) -> None:
    _extract_overlay(overlay, destination)
    source = TRANSITION_CAPSULE_SOURCE.resolve(strict=True)
    if builder.sha256_file(source) != TRANSITION_CAPSULE_SHA256:
        raise ValueError("v122 transition evidence capsule SHA-256 mismatch")
    target = destination / TRANSITION_CAPSULE_TARGET
    if target.exists():
        raise FileExistsError(f"overlay unexpectedly contains {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _six_hour_replay_research_budget(budget: dict) -> None:
    profile = dict(budget["smoke"])
    profile.update(
        {
            "agent_steps": 80,
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


def _append_frozen_v122_contract() -> None:
    locked_options = {
        "--release-version": "122",
        "--source-manifest-version": "121",
        "--source-smoke-manifest-name": "leaf_strategy_active_smoke_manifest.json",
        "--release-slug": "leaf-replay-research-top5-alignment-gate-pod32",
        "--experimental-axis": (
            "Leaf v121 Replay Research reissued with fail-closed submission metric "
            "alignment for modified replay descendants"
        ),
        "--system-description": (
            "Feature-identical v121 Replay Research with isolated identity and a "
            "rank gate that rejects modified replay submissions lacking an exact "
            "submission-aligned metric marker"
        ),
        "--required-memory-bundle-id": ATOMIC_BUNDLE_ID,
        "--required-memory-bundle-manifest-sha256": ATOMIC_BUNDLE_SHA256,
        "--required-memory-bundle-root": ATOMIC_BUNDLE_ROOT,
        "--required-formal-debug-clause-count": "296",
    }
    supplied = set(sys.argv[1:]) & set(locked_options)
    if supplied:
        raise ValueError(
            "v122 release contract options are immutable: "
            + ", ".join(sorted(supplied))
        )
    for option, value in locked_options.items():
        sys.argv.extend([option, value])


builder._six_hour_smoke_budget = _six_hour_replay_research_budget
builder.extract_overlay = _extract_overlay_with_transition_capsule


if __name__ == "__main__":
    _append_frozen_v122_contract()
    raise SystemExit(builder.main())
