#!/usr/bin/env python3
"""Package the v7 Leaf atomic release as an immutable MemorySnapshot Bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping


BUNDLE_ID = "end2end-leaf-atomic-recipe-runforest-v7"
BUNDLE_VERSION = "v7-leaf-atomic-20260811"
OFFICIAL_LEDGER_SHA256 = (
    "e15176956e4161e45348ab382438e19ce2bad0cdd98134b54e7a8de0b277dc66"
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def freeze(root: Path) -> None:
    for path in [*root.rglob("*"), root]:
        if path.is_symlink():
            raise ValueError(f"Memory bundle may not contain symlinks: {path}")
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()

    release = args.release_dir.resolve(strict=True)
    root = args.bundle_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to replace memory bundle root: {root}")
    bundle = root / "bundles" / BUNDLE_VERSION
    recipe = bundle / "recipe"
    runforest = bundle / "runforest"
    recipe.mkdir(parents=True)
    runforest.mkdir(parents=True)

    recipe_files = (
        "atomic_claims.json",
        "debug_retrieval_replay_report.json",
        "evidence_manifest.json",
        "implementation_capsules.json",
        "recipe_sops.json",
        "release_report.json",
        "teacher_output_schema.json",
        "teacher_packet.json",
        "teacher_claim_id_aliases_v1.json",
        "teacher_response_gpt56sol.raw.json",
        "teacher_response_gpt56sol.json",
    )
    for name in recipe_files:
        shutil.copy2(release / name, recipe / name)
    for name in ("graph.json", "index.npz", "TASK_AUDIT_REPORT.json"):
        shutil.copy2(release / "runforest" / name, runforest / name)

    recipe_payload = json.loads((recipe / "recipe_sops.json").read_text())
    evidence_payload = json.loads((recipe / "evidence_manifest.json").read_text())
    release_payload = json.loads((recipe / "release_report.json").read_text())
    artifact_hashes = {
        str(path.relative_to(bundle)): sha256_file(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest: dict[str, Any] = {
        "schema": "memory_bundle_manifest_v1",
        "bundle_id": BUNDLE_ID,
        "bundle_version": BUNDLE_VERSION,
        "created_at": args.created_at,
        "parent_bundle": "v6-leaf-official-20260810-r6",
        "target_task_id": "leaf-classification",
        "source_task_ids": ["leaf-classification"],
        "source_graph_manifest_schema": "hyperbolic_run_forest_memory_v1",
        "source_archive_sha256": "",
        "authority_policy_version": "claim_level_debug_operation_visibility_v1",
        "certification_level": "atomic_debug_retrieval_exploratory_smoke",
        "official_ledger_sha256": OFFICIAL_LEDGER_SHA256,
        "graph_hashes": {"runforest": artifact_hashes["runforest/graph.json"]},
        "index_hashes": {"runforest": artifact_hashes["runforest/index.npz"]},
        "build_report": "recipe/release_report.json",
        "recipe_sop_bundle_sha256": str(recipe_payload["bundle_sha256"]),
        "recipe_evidence_manifest_sha256": str(
            evidence_payload["manifest_sha256"]
        ),
        "atomic_claim_bundle_sha256": str(
            release_payload["atomic_claim_bundle_sha256"]
        ),
        "atomic_debug_authorized_count": int(
            release_payload["atomic_debug_authorized_count"]
        ),
        "artifact_hashes": artifact_hashes,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = payload_hash(manifest, "manifest_sha256")
    write_json(bundle / "manifest.json", manifest)

    pointer: dict[str, Any] = {
        "schema": "memory_bundle_current_v1",
        "bundle_id": BUNDLE_ID,
        "bundle_version": BUNDLE_VERSION,
        "bundle_path": f"bundles/{BUNDLE_VERSION}",
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_bundle": manifest["parent_bundle"],
        "published_at": args.created_at,
        "pointer_sha256": "",
    }
    pointer["pointer_sha256"] = payload_hash(pointer, "pointer_sha256")
    write_json(root / "CURRENT.json", pointer)

    receipt = {
        "schema": "mlevolve_leaf_atomic_memory_publication_receipt_v1",
        "bundle_root": str(root),
        "bundle_path": str(bundle),
        "bundle_id": BUNDLE_ID,
        "bundle_version": BUNDLE_VERSION,
        "bundle_manifest_file_sha256": sha256_file(bundle / "manifest.json"),
        "bundle_manifest_sha256": manifest["manifest_sha256"],
        "current_file_sha256": sha256_file(root / "CURRENT.json"),
        "graph_sha256": artifact_hashes["runforest/graph.json"],
        "index_sha256": artifact_hashes["runforest/index.npz"],
        "recipe_sop_file_sha256": artifact_hashes["recipe/recipe_sops.json"],
        "recipe_sop_bundle_sha256": recipe_payload["bundle_sha256"],
        "recipe_evidence_file_sha256": artifact_hashes[
            "recipe/evidence_manifest.json"
        ],
        "recipe_evidence_manifest_sha256": evidence_payload["manifest_sha256"],
        "recipe_implementation_file_sha256": artifact_hashes[
            "recipe/implementation_capsules.json"
        ],
        "atomic_claim_file_sha256": artifact_hashes["recipe/atomic_claims.json"],
        "atomic_claim_bundle_sha256": release_payload[
            "atomic_claim_bundle_sha256"
        ],
    }
    write_json(root / "PUBLICATION_RECEIPT.json", receipt)
    freeze(root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
