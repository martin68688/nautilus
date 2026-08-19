#!/usr/bin/env python3
"""Publish an immutable v10 child with canonical Replay-to-Recipe bindings."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping


TARGET_REPORT = "reports/leaf_official_replay_targets_v139.json"
PROJECTION_REPORT = "reports/replay_target_recipe_projection.json"


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def thaw(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            raise ValueError(f"Source bundle contains a symlink: {path}")
        path.chmod(0o755 if path.is_dir() else 0o644)


def freeze(root: Path) -> None:
    for path in [*root.rglob("*"), root]:
        if path.is_symlink():
            raise ValueError(f"Published bundle contains a symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)


def unique(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def project_replay_target_sop_ids(
    *,
    targets: Mapping[str, Any],
    recipe: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace stale target SOP IDs with the current canonical L1 Recipe IDs.

    Official-support candidate IDs are the immutable join key. Targets for which
    the Teacher intentionally published no L1 Recipe keep their code-bearing
    Replay evidence but receive no fabricated Recipe association.
    """

    projected = json.loads(json.dumps(targets))
    recipe_rows = list(recipe.get("nodes") or [])
    recipe_ids = {
        str(row.get("id") or "")
        for row in recipe_rows
        if row.get("type") == "SOP"
    }
    l1_ids = {
        str(row.get("id") or "")
        for row in recipe_rows
        if row.get("type") == "SOP"
        and row.get("abstraction_level") == "L1_recipe"
    }
    graph_sop_ids = {
        str(row.get("id") or "")
        for row in graph.get("nodes") or []
        if row.get("type") == "SOP"
    }
    available_sop_ids = recipe_ids | graph_sop_ids

    l1_by_candidate: dict[str, list[str]] = defaultdict(list)
    for row in recipe_rows:
        recipe_id = str(row.get("id") or "")
        if recipe_id not in l1_ids:
            continue
        for support in row.get("official_support") or []:
            candidate_id = str(support.get("candidate_id") or "")
            if candidate_id:
                l1_by_candidate[candidate_id].append(recipe_id)
    ambiguous = {
        candidate_id: unique(ids)
        for candidate_id, ids in l1_by_candidate.items()
        if len(unique(ids)) != 1
    }
    if ambiguous:
        raise ValueError(f"Official candidate maps to multiple L1 Recipes: {ambiguous}")

    rows: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()
    for target in projected.get("targets") or []:
        target_id = str(target.get("target_id") or "")
        if not target_id or target_id in seen_target_ids:
            raise ValueError(f"Duplicate or missing Replay target ID: {target_id}")
        seen_target_ids.add(target_id)
        before = unique(target.get("sop_ids") or [])
        canonical = unique(l1_by_candidate.get(target_id, []))
        if canonical:
            after = canonical
            disposition = "canonical_l1_recipe_restored"
        elif set(before).issubset(available_sop_ids):
            after = before
            disposition = "existing_valid_sop_binding_retained"
        else:
            after = []
            disposition = "no_distilled_l1_recipe_for_target"
        target["sop_ids"] = after
        rows.append(
            {
                "target_id": target_id,
                "source_sop_ids": before,
                "projected_sop_ids": after,
                "disposition": disposition,
            }
        )

    invalid = sorted(
        {
            sop_id
            for target in projected.get("targets") or []
            for sop_id in target.get("sop_ids") or []
            if str(sop_id) not in available_sop_ids
        }
    )
    if invalid:
        raise ValueError(f"Projected Replay targets still cite absent SOPs: {invalid}")
    canonical_targets = {
        row["target_id"]
        for row in rows
        if row["disposition"] == "canonical_l1_recipe_restored"
    }
    report = {
        "schema": "leaf_replay_target_recipe_projection_v1",
        "status": "pass",
        "join_key": "official_support.candidate_id == target.target_id",
        "target_count": len(rows),
        "canonical_l1_binding_count": len(canonical_targets),
        "targets_without_distilled_l1_recipe_count": sum(
            row["disposition"] == "no_distilled_l1_recipe_for_target"
            for row in rows
        ),
        "invalid_projected_sop_id_count": 0,
        "rows": rows,
        "report_sha256": "",
    }
    report["report_sha256"] = payload_hash(report, "report_sha256")
    return projected, report


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace bundle root: {output_root}")

    source_current = read_json(source_root / "CURRENT.json")
    source_bundle = (source_root / source_current["bundle_path"]).resolve(strict=True)
    source_manifest = read_json(source_bundle / "manifest.json")
    source_manifest_file_sha256 = sha256_file(source_bundle / "manifest.json")

    output_root.mkdir(parents=True)
    stage = output_root / f".staging-{args.bundle_version}"
    final = output_root / "bundles" / args.bundle_version
    shutil.copytree(source_bundle, stage, copy_function=shutil.copy2)
    thaw(stage)
    (stage / "manifest.json").unlink(missing_ok=False)

    targets_path = stage / TARGET_REPORT
    targets = read_json(targets_path)
    recipe = read_json(stage / "recipe" / "recipe_sops.json")
    graph = read_json(stage / "runforest" / "graph.json")
    source_targets_sha256 = sha256_file(targets_path)
    projected_targets, projection = project_replay_target_sop_ids(
        targets=targets,
        recipe=recipe,
        graph=graph,
    )
    projection.update(
        {
            "created_at": args.created_at,
            "source_bundle_version": source_manifest["bundle_version"],
            "source_target_file_sha256": source_targets_sha256,
        }
    )
    projection["report_sha256"] = payload_hash(projection, "report_sha256")
    write_json(targets_path, projected_targets)
    projection["projected_target_file_sha256"] = sha256_file(targets_path)
    projection["report_sha256"] = payload_hash(projection, "report_sha256")
    write_json(stage / PROJECTION_REPORT, projection)

    release_path = stage / "recipe" / "release_report.json"
    release = read_json(release_path)
    release.update(
        {
            "bundle_version": args.bundle_version,
            "created_at": args.created_at,
            "parent_bundle": source_manifest["bundle_version"],
            "replay_target_recipe_projection": {
                "status": "pass",
                "report": PROJECTION_REPORT,
                "report_sha256": projection["report_sha256"],
                "canonical_l1_binding_count": projection[
                    "canonical_l1_binding_count"
                ],
                "invalid_projected_sop_id_count": 0,
            },
        }
    )
    write_json(release_path, release)

    artifact_hashes = {
        str(path.relative_to(stage)): sha256_file(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = dict(source_manifest)
    manifest.update(
        {
            "bundle_id": args.bundle_id,
            "bundle_version": args.bundle_version,
            "created_at": args.created_at,
            "parent_bundle": source_manifest["bundle_version"],
            "build_report": "recipe/release_report.json",
            "replay_target_recipe_projection_report": PROJECTION_REPORT,
            "replay_target_recipe_projection_sha256": projection[
                "report_sha256"
            ],
            "artifact_hashes": artifact_hashes,
            "manifest_sha256": "",
        }
    )
    manifest["manifest_sha256"] = payload_hash(manifest, "manifest_sha256")
    write_json(stage / "manifest.json", manifest)

    for relative, expected in artifact_hashes.items():
        if sha256_file(stage / relative) != expected:
            raise ValueError(f"Artifact changed during publication: {relative}")
    if payload_hash(manifest, "manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Published manifest canonical hash mismatch")
    if sha256_file(source_bundle / "manifest.json") != source_manifest_file_sha256:
        raise ValueError("Source bundle changed during child publication")

    final.parent.mkdir(parents=True)
    stage.rename(final)
    pointer = {
        "schema": "memory_bundle_current_v1",
        "bundle_id": args.bundle_id,
        "bundle_version": args.bundle_version,
        "bundle_path": f"bundles/{args.bundle_version}",
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_bundle": source_manifest["bundle_version"],
        "published_at": args.created_at,
        "pointer_sha256": "",
    }
    pointer["pointer_sha256"] = payload_hash(pointer, "pointer_sha256")
    write_json(output_root / "CURRENT.json", pointer)
    receipt = {
        "schema": "mlevolve_leaf_replay_target_projection_publication_v1",
        "status": "pass",
        "bundle_root": str(output_root),
        "bundle_path": str(final),
        "bundle_id": args.bundle_id,
        "bundle_version": args.bundle_version,
        "parent_bundle": source_manifest["bundle_version"],
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "bundle_manifest_sha256": manifest["manifest_sha256"],
        "bundle_manifest_file_sha256": sha256_file(final / "manifest.json"),
        "current_file_sha256": sha256_file(output_root / "CURRENT.json"),
        "source_target_file_sha256": source_targets_sha256,
        "projected_target_file_sha256": sha256_file(final / TARGET_REPORT),
        "projection_report_sha256": projection["report_sha256"],
        "canonical_l1_binding_count": projection["canonical_l1_binding_count"],
        "invalid_projected_sop_id_count": 0,
        "source_bundle_unchanged": True,
        "secrets_recorded": False,
    }
    write_json(output_root / "PUBLICATION_RECEIPT.json", receipt)
    freeze(output_root)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    receipt = build(args)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
