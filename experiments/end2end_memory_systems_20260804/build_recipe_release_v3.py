#!/usr/bin/env python3
"""Attach the best retained Leaf terminal run to the frozen Recipe v2 SOP."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


TARGET_SOP_ID = "recipe::leaf-classification::003"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical_hash(value: Mapping[str, Any], excluded_field: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != excluded_field},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(
    base_recipe_path: Path,
    base_evidence_path: Path,
    incremental_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    recipe = copy.deepcopy(_read(base_recipe_path))
    evidence = copy.deepcopy(_read(base_evidence_path))
    incremental = _read(incremental_path)
    records = incremental.get("records") or []
    if len(records) != 1:
        raise ValueError("Recipe v3 requires exactly one retained terminal record")
    record = records[0]
    if (
        record.get("task_id") != "leaf-classification"
        or record.get("system_id") != "dynamic_hybrid"
        or record.get("leakage_audit", {}).get("status") != "clean"
        or record.get("selected_for_terminal") is not True
    ):
        raise ValueError("Retained terminal record is not the clean Leaf Dynamic result")

    node_id = str(record["node_id"])
    leaf_records = evidence["selected_evidence"]["leaf-classification"]
    if any(str(row.get("node_id")) == node_id for row in leaf_records):
        raise ValueError("Retained Leaf terminal record is already attached")
    source_files = dict(record.get("source_files") or {})
    leaf_records.append(
        {
            "audit_status": "clean",
            "code_sha256": str(record["code_sha256"]),
            "code_summary": str(record.get("code_summary") or ""),
            "memory_disposition": "positive_eligible",
            "metric": float(record["metric"]),
            "metric_direction": "minimize",
            "metric_improvement": None,
            "metric_provenance": "sealed_fixed_holdout_terminal_score",
            "node_id": node_id,
            "paper_grade_eligible": True,
            "plan": str(record.get("plan") or ""),
            "rank_eligible": True,
            "run_id": str(record["run_id"]),
            "source_artifact": str(source_files.get("journal") or ""),
            "source_artifact_sha256": str(
                source_files.get("journal_sha256") or ""
            ),
            "source_cohort": "post_freeze_leaf_smoke_20260806",
            "stage": str(record.get("stage") or "draft"),
            "step": record.get("step"),
            "task_domain": "multimodal_multiclass_classification",
            "task_id": "leaf-classification",
        }
    )
    leaf_records.sort(key=lambda row: str(row["node_id"]))
    evidence["selected_counts_by_task"]["leaf-classification"] += 1
    evidence["admitted_counts_by_task"]["leaf-classification"] += 1
    evidence["admitted_node_count"] += 1
    evidence["source_artifacts"].append(
        {
            "cohort": "post_freeze_leaf_smoke_20260806",
            "path": str(incremental_path),
            "record_count": 1,
            "strict_recipe_eligible_count": 1,
        }
    )
    evidence["created_at"] = "2026-08-06T12:00:00+08:00"
    evidence["manifest_sha256"] = ""
    evidence["manifest_sha256"] = _canonical_hash(evidence, "manifest_sha256")

    matching = [node for node in recipe.get("nodes") or [] if node.get("id") == TARGET_SOP_ID]
    if len(matching) != 1:
        raise ValueError(f"Expected one target SOP: {TARGET_SOP_ID}")
    sop = matching[0]
    for field in ("source_node_ids", "clean_supporting_node_ids"):
        values = [str(value) for value in sop.get(field) or []]
        if node_id not in values:
            values.append(node_id)
        sop[field] = values
    recipe["bundle_version"] = "recipe-sop-v3-20260806"
    recipe["created_at"] = "2026-08-06T12:00:00+08:00"
    recipe["evidence_manifest_sha256"] = evidence["manifest_sha256"]
    recipe["bundle_sha256"] = ""
    recipe["bundle_sha256"] = _canonical_hash(recipe, "bundle_sha256")

    report = {
        "schema": "mlevolve_recipe_release_attachment_report_v1",
        "release": recipe["bundle_version"],
        "attached_sop_id": TARGET_SOP_ID,
        "attached_node_id": node_id,
        "terminal_metric": float(record["metric"]),
        "internal_metric": record.get("internal_metric"),
        "code_sha256": str(record["code_sha256"]),
        "teacher_called": False,
        "reason": "attach retained clean terminal implementation to its already-selected Recipe",
    }
    return recipe, evidence, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--base-evidence", type=Path, required=True)
    parser.add_argument("--incremental", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    recipe, evidence, report = build(
        args.base_recipe.resolve(strict=True),
        args.base_evidence.resolve(strict=True),
        args.incremental.resolve(strict=True),
    )
    _write(args.output_dir / "recipe_sops.json", recipe)
    _write(args.output_dir / "evidence_manifest.json", evidence)
    _write(args.output_dir / "release_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
