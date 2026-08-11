#!/usr/bin/env python3
"""Build immutable v77 Leaf source bound to the atomic Debug memory release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from prepare_leaf_strategy_active_v73 import (
    extract_overlay,
    freeze,
    make_writable,
    payload_hash,
    read_object,
    sha256_file,
    write_object,
)


RELEASE_VERSION = 77
BUNDLE_ROOT = (
    "/workspace/experiment-end2end-memory-agent-v77/"
    "memory-leaf-atomic-v7/leaf-classification"
)
OFFICIAL_LEDGER_SHA256 = (
    "e15176956e4161e45348ab382438e19ce2bad0cdd98134b54e7a8de0b277dc66"
)


def _update_memory_manifest(
    path: Path,
    *,
    publication: dict[str, Any],
    bundle_manifest: dict[str, Any],
) -> dict[str, Any]:
    payload = read_object(path)
    task = dict(payload["task_bundles"]["leaf-classification"])
    task.update(
        {
            "bundle_id": publication["bundle_id"],
            "bundle_manifest_file_sha256": publication[
                "bundle_manifest_file_sha256"
            ],
            "bundle_manifest_sha256": publication["bundle_manifest_sha256"],
            "bundle_root": BUNDLE_ROOT,
            "bundle_version": publication["bundle_version"],
            "current_file_sha256": publication["current_file_sha256"],
            "formal_child_publication": False,
            "graph_sha256": publication["graph_sha256"],
            "index_sha256": publication["index_sha256"],
            "memory_scope": (
                "leaf_official_recipe_v6_plus_full_transition_atomic_debug_v7"
            ),
            "official_ledger_sha256": OFFICIAL_LEDGER_SHA256,
            "protocol_ref": "leaf-atomic-debug-memory@20260811#v77",
            "recipe_evidence_manifest_sha256": publication[
                "recipe_evidence_manifest_sha256"
            ],
            "recipe_sop_bundle_sha256": publication[
                "recipe_sop_bundle_sha256"
            ],
            "atomic_claim_bundle_sha256": publication[
                "atomic_claim_bundle_sha256"
            ],
            "atomic_debug_authorized_count": int(
                bundle_manifest["atomic_debug_authorized_count"]
            ),
            "authority_policy_version": bundle_manifest[
                "authority_policy_version"
            ],
            "certification_level": bundle_manifest["certification_level"],
        }
    )
    payload.update(
        {
            "source_graph_sha256": publication["graph_sha256"],
            "source_index_sha256": publication["index_sha256"],
            "claim_level_debug_memory": True,
            "atomic_claim_bundle_sha256": publication[
                "atomic_claim_bundle_sha256"
            ],
            "atomic_debug_authorized_count": int(
                bundle_manifest["atomic_debug_authorized_count"]
            ),
            "ranking_policy": "task_first_structured_debug_signature_v3",
            "task_bundles": {"leaf-classification": task},
        }
    )
    payload["manifest_hash"] = payload_hash(payload, "manifest_hash")
    write_object(path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--memory-bundle-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.resolve(strict=True)
    destination = args.destination.resolve()
    overlay = args.overlay.resolve(strict=True)
    memory_root = args.memory_bundle_root.resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable source: {destination}")
    shutil.copytree(base, destination, symlinks=False)
    make_writable(destination)
    extract_overlay(overlay, destination)

    exp_root = destination / "experiments/end2end_memory_systems_20260804"
    source_manifests = exp_root / "manifests_v76"
    manifests = exp_root / "manifests_v77"
    if manifests.exists():
        raise FileExistsError(f"overlay unexpectedly created {manifests}")
    shutil.copytree(source_manifests, manifests)
    make_writable(manifests)

    publication = read_object(memory_root / "PUBLICATION_RECEIPT.json")
    pointer = read_object(memory_root / "CURRENT.json")
    bundle_path = memory_root / str(pointer["bundle_path"])
    bundle_manifest = read_object(bundle_path / "manifest.json")
    if publication["bundle_manifest_sha256"] != bundle_manifest["manifest_sha256"]:
        raise ValueError("memory publication receipt does not match bundle manifest")

    memory_path = manifests / "memory_bundles.json"
    memory = _update_memory_manifest(
        memory_path,
        publication=publication,
        bundle_manifest=bundle_manifest,
    )

    config_path = exp_root / "systems_v77/dynamic_hybrid.yaml"
    systems_path = manifests / "systems.json"
    systems = read_object(systems_path)
    systems["experimental_axis"] = (
        "Leaf full-transition claim-level Debug memory plus task-first structured "
        "causal ranking under the v76 active Strategy/Atomic harness"
    )
    systems["systems"] = [
        {
            "config_path": "systems_v77/dynamic_hybrid.yaml",
            "config_sha256": sha256_file(config_path),
            "description": (
                "Dynamic Router with atomic claim visibility and structured "
                "exception/model/operand/symbol Debug ranking"
            ),
            "kind": "internal_exploratory",
            "label": "S5-v77-leaf-atomic-debug-memory",
            "limitation": "single exploratory six-hour Leaf Smoke",
            "primary_reference": None,
            "system_id": "dynamic_hybrid",
        }
    ]
    systems["system_count"] = 1
    systems["manifest_hash"] = payload_hash(systems, "manifest_hash")
    write_object(systems_path, systems)

    budget_path = manifests / "budget.json"
    budget = read_object(budget_path)
    budget["manifest_hash"] = payload_hash(budget, "manifest_hash")
    write_object(budget_path, budget)

    smoke_path = manifests / "leaf_atomic_memory_smoke_manifest.json"
    source_smoke = read_object(source_manifests / "leaf_strategy_active_smoke_manifest.json")

    # The execution manifest binds the source-lock hash, so including that
    # manifest inside the source lock would create an impossible hash cycle.
    # It remains covered by the outer SOURCE_FILES.sha256 release lock.
    source_lock_path = manifests / "source_lock.json"
    source_lock = read_object(source_lock_path)
    source_lock.update(
        {
            "git_head": args.git_head,
            "git_dirty": False,
            "files": [],
            "complete_runtime_file_hash_lock": True,
            "control_file_exclusions": [
                str(smoke_path.relative_to(destination)),
                str(source_lock_path.relative_to(destination)),
                "SOURCE_FILES.sha256",
            ],
        }
    )
    excluded = {source_lock_path, smoke_path, destination / "SOURCE_FILES.sha256"}
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.is_symlink() or path in excluded:
            continue
        source_lock["files"].append(
            {
                "path": str(path.relative_to(destination)),
                "sha256": sha256_file(path),
            }
        )
    source_lock["manifest_hash"] = payload_hash(source_lock, "manifest_hash")
    write_object(source_lock_path, source_lock)

    smoke = dict(source_smoke)
    logical_id = (
        "e2e-smoke-leaf-atomic-memory-v77__leaf-classification__"
        "dynamic_hybrid__seed-1"
    )
    bindings = dict(smoke["bindings"])
    bindings.update(
        {
            "systems_manifest_hash": systems["manifest_hash"],
            "budget_manifest_hash": budget["manifest_hash"],
            "memory_bundles_manifest_hash": memory["manifest_hash"],
            "source_lock_manifest_hash": source_lock["manifest_hash"],
        }
    )
    row = dict(smoke["runs"][0])
    row.update(
        {
            "logical_run_id": logical_id,
            "launch_position": 0,
            "task_launch_position": 0,
            "formal_result_eligible": False,
            "exploratory_pilot": True,
            "bindings": dict(bindings),
        }
    )
    row["row_hash"] = payload_hash(row, "row_hash")
    smoke.update(
        {
            "release_id": "end2end-leaf-atomic-memory-v77-smoke",
            "kind": "smoke",
            "run_count": 1,
            "runs": [row],
            "system_ids": ["dynamic_hybrid"],
            "task_ids": ["leaf-classification"],
            "first_parallel_batch": ["dynamic_hybrid"],
            "launch_order_randomization": "single Dynamic Leaf atomic-memory Smoke",
            "formal_result_eligible": False,
            "exploratory_pilot": True,
            "bindings": bindings,
        }
    )
    smoke["manifest_hash"] = payload_hash(smoke, "manifest_hash")
    write_object(smoke_path, smoke)

    file_count, source_manifest_sha = freeze(destination)
    receipt = {
        "schema": "mlevolve_leaf_atomic_memory_source_release_v1",
        "release_version": RELEASE_VERSION,
        "git_head": args.git_head,
        "base_source": str(base),
        "frozen_source": str(destination),
        "source_file_count": file_count,
        "source_manifest_sha256": source_manifest_sha,
        "systems_manifest_sha256": systems["manifest_hash"],
        "budget_manifest_sha256": budget["manifest_hash"],
        "memory_bundles_manifest_sha256": memory["manifest_hash"],
        "source_lock_manifest_sha256": source_lock["manifest_hash"],
        "smoke_manifest_sha256": smoke["manifest_hash"],
        "smoke_manifest": str(smoke_path),
        "memory_bundle_manifest_sha256": publication["bundle_manifest_sha256"],
        "atomic_claim_bundle_sha256": publication["atomic_claim_bundle_sha256"],
        "ranking_policy": "task_first_structured_debug_signature_v3",
        "agent_time_limit_seconds": 21600,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
