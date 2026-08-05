#!/usr/bin/env python3
"""Publish one reviewed four-task RunForest as four direct End2End Bundles.

This experiment is not a transfer-heldout benchmark.  Every task receives the
same immutable four-task graph/index and may retrieve its own historical
records.  The source graph already carries the paper-grade admission labels;
publication only wraps the frozen artifacts in the normal Base/CURRENT layout
used by MLEvolve and records the best positive-eligible record for each task.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mlevolve") not in sys.path:
    sys.path.insert(0, str(ROOT / "mlevolve"))

from authority.memory_snapshot import make_current_pointer, sha256_file  # noqa: E402


BINDING_SCHEMA = "mlevolve_end2end_direct_fourtask_memory_binding_v2"
BUNDLE_VERSION = "v2"
TASK_DIRECTIONS = {
    "aerial-cactus-identification": "maximize",
    "leaf-classification": "minimize",
    "denoising-dirty-documents": "minimize",
    "new-york-city-taxi-fare-prediction": "minimize",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_hash(value: Mapping[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_task(value: object) -> str:
    task = str(value or "").strip()
    while task.startswith("full-"):
        task = task[len("full-") :]
    return task


def positive_eligible(node: Mapping[str, Any]) -> bool:
    metric = node.get("metric")
    audit = node.get("leakage_audit")
    return bool(
        node.get("type") == "RunNode"
        and node.get("is_buggy") is False
        and node.get("is_valid") is True
        and isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
        and isinstance(audit, Mapping)
        and audit.get("status") == "clean"
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
        and audit.get("rank_eligible") is True
    )


def _number(value: object, default: float) -> float:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return default


def best_positive_record(
    nodes: list[Mapping[str, Any]], task_id: str
) -> dict[str, Any]:
    direction = TASK_DIRECTIONS[task_id]
    rows: list[tuple[tuple[float, float, float, str], Mapping[str, Any]]] = []
    for node in nodes:
        if canonical_task(node.get("task")) != task_id or not positive_eligible(node):
            continue
        declared_maximize = next(
            (
                node.get(key)
                for key in ("metric_maximize", "maximize")
                if isinstance(node.get(key), bool)
            ),
            None,
        )
        expected_maximize = direction == "maximize"
        if declared_maximize is not expected_maximize:
            raise ValueError(
                f"Positive record has missing/wrong metric direction for {task_id}: "
                f"{node.get('id')}"
            )
        metric = float(node["metric"])
        normalized_metric = metric if direction == "maximize" else -metric
        rows.append(
            (
                (
                    normalized_metric,
                    _number(node.get("metric_improvement"), float("-inf")),
                    _number(node.get("step"), float("-inf")),
                    str(node.get("id") or ""),
                ),
                node,
            )
        )
    if not rows:
        raise ValueError(f"No positive-eligible historical record for {task_id}")
    rows.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            item[0][3],
        )
    )
    node = rows[0][1]
    return {
        "task_id": task_id,
        "node_id": str(node["id"]),
        "run_id": str(node.get("run_id") or ""),
        "run_short_id": str(node.get("run_short_id") or ""),
        "stage": str(node.get("stage") or ""),
        "step": node.get("step"),
        "metric": float(node["metric"]),
        "direction": direction,
        "metric_maximize": direction == "maximize",
        "metric_improvement": node.get("metric_improvement"),
        "admission": "clean_positive_eligible",
    }


def _validate_source(
    *, source_graph: Path, source_index: Path, source_manifest: Path
) -> tuple[dict[str, Any], dict[str, Any], list[Mapping[str, Any]]]:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("schema") != "fourtask_runforest_graph_manifest_v2":
        raise ValueError("Unsupported four-task source manifest")
    if set(map(str, source.get("task_ids") or [])) != set(TASK_DIRECTIONS):
        raise ValueError("Four-task source manifest task inventory mismatch")
    graph_sha = sha256_file(source_graph)
    index_sha = sha256_file(source_index)
    if graph_sha != source.get("graph_sha256"):
        raise ValueError("Four-task source graph hash mismatch")
    if index_sha != source.get("index_sha256"):
        raise ValueError("Four-task source index hash mismatch")
    graph = json.loads(source_graph.read_text(encoding="utf-8"))
    meta = graph.get("meta") if isinstance(graph, Mapping) else None
    if not isinstance(meta, Mapping):
        raise ValueError("Four-task graph has no metadata")
    required = {
        "source_membership_verified": True,
        "leak_verified": True,
        "leak_audited": True,
        "paper_grade": True,
        "positive_admission_enforced": True,
    }
    for field, expected in required.items():
        if meta.get(field) is not expected:
            raise ValueError(f"Four-task graph metadata mismatch: {field}")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Four-task graph has no node inventory")
    return source, graph, nodes


def build(
    *,
    source_graph: Path,
    source_index: Path,
    source_manifest: Path,
    frozen_memory_manifest: Path,
    output_root: Path,
    created_at: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Direct memory output already exists: {output_root}")
    source_graph = source_graph.resolve(strict=True)
    source_index = source_index.resolve(strict=True)
    source_manifest = source_manifest.resolve(strict=True)
    source, graph, nodes = _validate_source(
        source_graph=source_graph,
        source_index=source_index,
        source_manifest=source_manifest,
    )
    frozen = json.loads(frozen_memory_manifest.read_text(encoding="utf-8"))
    tasks = copy.deepcopy(frozen.get("task_bundles") or {})
    if set(tasks) != set(TASK_DIRECTIONS):
        raise ValueError("Frozen End2End task bundle inventory is incomplete")

    graph_sha = str(source["graph_sha256"])
    index_sha = str(source["index_sha256"])
    source_manifest_file_sha = sha256_file(source_manifest)
    best_records = {
        task_id: best_positive_record(nodes, task_id)
        for task_id in TASK_DIRECTIONS
    }
    node_type_counts: dict[str, int] = {}
    for node in nodes:
        node_type = str(node.get("type") or "unknown")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    for task_id in TASK_DIRECTIONS:
        task_root = output_root / task_id
        bundle_root = task_root / "bundles" / BUNDLE_VERSION
        runforest_root = bundle_root / "runforest"
        runforest_root.mkdir(parents=True)
        shutil.copy2(source_graph, runforest_root / "graph.json")
        shutil.copy2(source_index, runforest_root / "index.npz")
        artifact_hashes = {
            "runforest/graph.json": graph_sha,
            "runforest/index.npz": index_sha,
        }
        manifest: dict[str, Any] = {
            "schema": "memory_bundle_manifest_v1",
            "bundle_id": f"end2end-fourtask-direct-{task_id}-v2",
            "bundle_version": BUNDLE_VERSION,
            "parent_bundle": None,
            "authority_policy_version": "experiment_effectiveness_offline_freeze_v1",
            "certification_level": "legacy_uncertified",
            "source_graph_manifest_schema": str(source["schema"]),
            "source_graph_manifest_file_sha256": source_manifest_file_sha,
            "source_archive_sha256": str(source.get("source_archive_sha256") or ""),
            "source_artifact_version": str(source.get("artifact_version") or ""),
            "source_task_ids": sorted(TASK_DIRECTIONS),
            "target_task_id": task_id,
            "graph_hashes": {"runforest": graph_sha},
            "index_hashes": {"runforest": index_sha},
            "build_report": "runforest/graph.json",
            "created_at": created_at,
            "artifact_hashes": artifact_hashes,
            "manifest_sha256": "",
        }
        manifest["manifest_sha256"] = payload_hash(manifest, "manifest_sha256")
        write_json(bundle_root / "manifest.json", manifest)
        current = make_current_pointer(
            bundle_path=f"bundles/{BUNDLE_VERSION}",
            manifest=manifest,
            parent_bundle=None,
            published_at=created_at,
        )
        write_json(task_root / "CURRENT.json", current)
        task = dict(tasks[task_id])
        task.update(
            {
                "bundle_root": str(task_root),
                "bundle_id": str(manifest["bundle_id"]),
                "bundle_version": BUNDLE_VERSION,
                "bundle_manifest_sha256": str(manifest["manifest_sha256"]),
                "bundle_manifest_file_sha256": sha256_file(
                    bundle_root / "manifest.json"
                ),
                "current_file_sha256": sha256_file(task_root / "CURRENT.json"),
                "graph_sha256": graph_sha,
                "index_sha256": index_sha,
                "memory_scope": "full_reviewed_fourtask_with_same_task_history",
                "formal_child_publication": False,
                "same_task_history_enabled": True,
                "same_task_best_node_id": best_records[task_id]["node_id"],
            }
        )
        tasks[task_id] = task

    binding: dict[str, Any] = {
        "schema": BINDING_SCHEMA,
        "status": "direct_experimental_fourtask_same_task_enabled",
        "created_at": created_at,
        "source_graph_path": str(source_graph),
        "source_index_path": str(source_index),
        "source_manifest_path": str(source_manifest),
        "source_graph_sha256": graph_sha,
        "source_index_sha256": index_sha,
        "source_manifest_file_sha256": source_manifest_file_sha,
        "publisher_source_sha256": sha256_file(Path(__file__)),
        "source_archive_sha256": str(source.get("source_archive_sha256") or ""),
        "source_node_count": len(nodes),
        "source_node_type_counts": dict(sorted(node_type_counts.items())),
        "excluded_run_ids": [],
        "same_task_history_policy": (
            "allow target-task history; pin the direction-aware best clean "
            "positive-eligible record for Dynamic retrieval"
        ),
        "same_task_best_records": best_records,
        "tasks": tasks,
        "binding_sha256": "",
    }
    binding["binding_sha256"] = payload_hash(binding, "binding_sha256")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "MEMORY_BINDING.json", binding)
    return binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-graph", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--frozen-memory-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    result = build(
        source_graph=args.source_graph,
        source_index=args.source_index,
        source_manifest=args.source_manifest,
        frozen_memory_manifest=args.frozen_memory_manifest,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
