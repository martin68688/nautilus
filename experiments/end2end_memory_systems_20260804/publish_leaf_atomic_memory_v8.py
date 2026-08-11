#!/usr/bin/env python3
"""Publish Leaf atomic Recipe memory with formal Debug visibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
MLEVOLVE_ROOT = ROOT / "mlevolve"
if str(MLEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE_ROOT))

from authority.memory_snapshot import MemorySnapshotLoader  # noqa: E402
from authority.recipe_visibility_publication import (  # noqa: E402
    compile_recipe_debug_visibility,
)


BUNDLE_ID = "end2end-leaf-atomic-recipe-runforest-v8"
BUNDLE_VERSION = "v8-leaf-atomic-visibility-20260812"
ACTIVE_PROTOCOL_REF = (
    "mlevolve-default@1#"
    "cdb8439fa96b3add95788d9c1462811e5319ef73bca1cb4cc606ce07cd7a3a29"
)
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


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
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
    parser.add_argument(
        "--active-protocol-ref",
        default=ACTIVE_PROTOCOL_REF,
    )
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
    publication = compile_recipe_debug_visibility(
        recipe_payload,
        active_protocol_ref=str(args.active_protocol_ref),
    )
    published_count = int(publication["report"]["published_clause_count"])
    atomic_authorized_count = int(
        release_payload["atomic_debug_authorized_count"]
    )
    expected_formal_count = int(
        release_payload.get("deterministic_repair_sop_count") or 0
    ) + int(release_payload.get("teacher_generalized_repair_sop_count") or 0)
    if published_count != expected_formal_count:
        raise ValueError(
            "Formal Debug visibility coverage does not match the authorized "
            "claim-backed Recipe repair count: "
            f"{published_count} != {expected_formal_count}"
        )
    write_jsonl(bundle / "sop" / "clauses.jsonl", publication["clauses"])
    write_json(
        bundle
        / "visibility"
        / "precompiled_masks"
        / "declared_scope_masks.json",
        publication["masks"],
    )
    write_json(
        bundle / "reports" / "recipe_visibility_publication.json",
        publication["report"],
    )

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
        "parent_bundle": "v7-leaf-atomic-20260811",
        "target_task_id": "leaf-classification",
        "source_task_ids": ["leaf-classification"],
        "source_graph_manifest_schema": "hyperbolic_run_forest_memory_v1",
        "source_archive_sha256": "",
        "active_protocol_ref": str(args.active_protocol_ref),
        "authority_policy_version": "claim_level_debug_operation_visibility_v2",
        "certification_level": "atomic_debug_visibility_exploratory_smoke",
        "official_ledger_sha256": OFFICIAL_LEDGER_SHA256,
        "graph_hashes": {"runforest": artifact_hashes["runforest/graph.json"]},
        "index_hashes": {"runforest": artifact_hashes["runforest/index.npz"]},
        "build_report": "recipe/release_report.json",
        "visibility_publication_report": (
            "reports/recipe_visibility_publication.json"
        ),
        "recipe_sop_bundle_sha256": str(recipe_payload["bundle_sha256"]),
        "recipe_evidence_manifest_sha256": str(
            evidence_payload["manifest_sha256"]
        ),
        "atomic_claim_bundle_sha256": str(
            release_payload["atomic_claim_bundle_sha256"]
        ),
        "atomic_debug_authorized_count": atomic_authorized_count,
        "formal_debug_clause_count": published_count,
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
        "schema": "mlevolve_leaf_atomic_memory_publication_receipt_v2",
        "bundle_root": str(root),
        "bundle_path": str(bundle),
        "bundle_id": BUNDLE_ID,
        "bundle_version": BUNDLE_VERSION,
        "active_protocol_ref": str(args.active_protocol_ref),
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
        "formal_clause_file_sha256": artifact_hashes["sop/clauses.jsonl"],
        "declared_scope_masks_file_sha256": artifact_hashes[
            "visibility/precompiled_masks/declared_scope_masks.json"
        ],
        "visibility_publication_report_file_sha256": artifact_hashes[
            "reports/recipe_visibility_publication.json"
        ],
        "formal_debug_clause_count": published_count,
    }
    write_json(root / "PUBLICATION_RECEIPT.json", receipt)
    freeze(root)

    base = MemorySnapshotLoader(root).load_base(verify_artifacts=True)
    if base.manifest_sha256 != manifest["manifest_sha256"]:
        raise RuntimeError("Published Bundle failed immutable loader verification")
    if len(base.read_jsonl("sop/clauses.jsonl")) != published_count:
        raise RuntimeError("Published Bundle formal clause count changed after freeze")

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
