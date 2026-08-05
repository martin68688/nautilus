#!/usr/bin/env python3
"""Expose the reviewed seed-heldout Base directly for the Leaf experiment.

This intentionally avoids the formal child-publication, domain certification,
and transition-to-SOP proof pipeline.  It copies one already reviewed immutable
Base, writes a normal CURRENT pointer, and emits one combined End2End binding.
The dynamic router—not the bundle builder—then searches target-task history.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mlevolve") not in sys.path:
    sys.path.insert(0, str(ROOT / "mlevolve"))

from authority.memory_snapshot import (  # noqa: E402
    ImmutableBaseBundle,
    make_current_pointer,
    sha256_file,
)


BINDING_SCHEMA = "mlevolve_end2end_direct_memory_binding_v1"
LEAF_TASK_ID = "leaf-classification"


def payload_hash(value: Mapping[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def build(
    *,
    source_bundle: Path,
    frozen_memory_manifest: Path,
    output_root: Path,
    created_at: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Direct memory output already exists: {output_root}")
    source = ImmutableBaseBundle.load(
        source_bundle.resolve(strict=True), verify_artifacts=False
    )
    frozen = json.loads(frozen_memory_manifest.read_text(encoding="utf-8"))
    tasks = copy.deepcopy(frozen.get("task_bundles") or {})
    if set(tasks) != {
        "aerial-cactus-identification",
        "leaf-classification",
        "denoising-dirty-documents",
        "new-york-city-taxi-fare-prediction",
    }:
        raise ValueError("Frozen End2End task bundle inventory is incomplete")

    leaf_root = output_root / LEAF_TASK_ID
    copied = leaf_root / "bundles" / source.bundle_version
    copied.parent.mkdir(parents=True)
    shutil.copytree(source.path, copied)
    current = make_current_pointer(
        bundle_path=f"bundles/{source.bundle_version}",
        manifest=source.manifest,
        parent_bundle=source.manifest.get("parent_bundle"),
        published_at=created_at,
    )
    write_json(leaf_root / "CURRENT.json", current)

    leaf = dict(tasks[LEAF_TASK_ID])
    leaf.update(
        {
            "bundle_root": str(leaf_root),
            "bundle_id": source.bundle_id,
            "bundle_version": source.bundle_version,
            "bundle_manifest_sha256": source.manifest_sha256,
            "bundle_manifest_file_sha256": sha256_file(
                copied / "manifest.json"
            ),
            "current_file_sha256": sha256_file(leaf_root / "CURRENT.json"),
            "graph_sha256": sha256_file(copied / "runforest" / "graph.json"),
            "index_sha256": sha256_file(copied / "runforest" / "index.npz"),
            "memory_scope": "full_reviewed_seed_heldout_base",
            "formal_child_publication": False,
        }
    )
    tasks[LEAF_TASK_ID] = leaf

    binding: dict[str, Any] = {
        "schema": BINDING_SCHEMA,
        "status": "direct_experimental_seed_heldout_base",
        "created_at": created_at,
        "source_bundle_path": str(source.path),
        "source_bundle_id": source.bundle_id,
        "source_bundle_manifest_sha256": source.manifest_sha256,
        "tasks": tasks,
        "binding_sha256": "",
    }
    binding["binding_sha256"] = payload_hash(binding, "binding_sha256")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "MEMORY_BINDING.json", binding)
    return binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--frozen-memory-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    result = build(
        source_bundle=args.source_bundle,
        frozen_memory_manifest=args.frozen_memory_manifest,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
