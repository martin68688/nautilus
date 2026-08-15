from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / "end2end_memory_systems_20260804"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from prepare_leaf_strategy_active_v74 import (  # noqa: E402
    validate_required_memory_binding,
)


ATOMIC_BUNDLE_ID = "end2end-leaf-atomic-recipe-runforest-v8"
ATOMIC_BUNDLE_SHA256 = (
    "fa697bbd5fc47eb728ba13a63d693bc4777b47c6b5c984f653e89041871aa0bb"
)
ATOMIC_BUNDLE_ROOT = (
    "/workspace/experiment-end2end-memory-agent-v89/"
    "memory-leaf-atomic-v8/leaf-classification"
)


def _atomic_memory_manifest() -> dict:
    return {
        "claim_level_debug_memory": True,
        "task_bundles": {
            "leaf-classification": {
                "bundle_id": ATOMIC_BUNDLE_ID,
                "bundle_manifest_sha256": ATOMIC_BUNDLE_SHA256,
                "bundle_root": ATOMIC_BUNDLE_ROOT,
                "formal_debug_clause_count": 296,
                "formal_clause_file_sha256": "2" * 64,
                "declared_scope_masks_file_sha256": "1" * 64,
            }
        },
    }


def test_release_accepts_required_atomic_debug_binding() -> None:
    receipt = validate_required_memory_binding(
        _atomic_memory_manifest(),
        required_bundle_id=ATOMIC_BUNDLE_ID,
        required_bundle_manifest_sha256=ATOMIC_BUNDLE_SHA256,
        required_bundle_root=ATOMIC_BUNDLE_ROOT,
        required_formal_debug_clause_count=296,
    )

    assert receipt["status"] == "validated"
    assert receipt["formal_debug_clause_count"] == 296


def test_release_rejects_old_base_even_with_atomic_recipe_overlay() -> None:
    memory = _atomic_memory_manifest()
    task = memory["task_bundles"]["leaf-classification"]
    task.update(
        {
            "bundle_id": "end2end-leaf-official-recipe-runforest-v6",
            "bundle_manifest_sha256": "c" * 64,
            "bundle_root": (
                "/workspace/experiment-end2end-memory-agent-v56/"
                "memory-leaf-official-v6-r6/leaf-classification"
            ),
        }
    )

    with pytest.raises(ValueError, match="memory bundle mismatch"):
        validate_required_memory_binding(
            memory,
            required_bundle_id=ATOMIC_BUNDLE_ID,
            required_bundle_manifest_sha256=ATOMIC_BUNDLE_SHA256,
            required_bundle_root=ATOMIC_BUNDLE_ROOT,
            required_formal_debug_clause_count=296,
        )


@pytest.mark.parametrize(
    "missing_key",
    ["formal_clause_file_sha256", "declared_scope_masks_file_sha256"],
)
def test_release_rejects_missing_formal_authority_artifact(
    missing_key: str,
) -> None:
    memory = _atomic_memory_manifest()
    memory["task_bundles"]["leaf-classification"][missing_key] = ""

    with pytest.raises(ValueError, match="formal Clause and declared-scope"):
        validate_required_memory_binding(
            memory,
            required_bundle_id=ATOMIC_BUNDLE_ID,
            required_bundle_manifest_sha256=ATOMIC_BUNDLE_SHA256,
            required_bundle_root=ATOMIC_BUNDLE_ROOT,
            required_formal_debug_clause_count=296,
        )
