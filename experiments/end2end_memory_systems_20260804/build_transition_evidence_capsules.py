#!/usr/bin/env python3
"""Materialize strict-clean Debug/Improve parent-child programs for Resolver use.

The first-stage Search/Grep/Judge path intentionally sees only compact graph
metadata.  This builder creates a separate, content-addressed artifact that a
deterministic Evidence Resolver may open *after* the Judge has selected a
candidate.  Full source never participates in retrieval or ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCHEMA = "mlevolve_transition_evidence_capsules_v2"
TASK_ID = "leaf-classification"
EVIDENCE_CLASSES = {
    "strict_debug_observed",
    "official_observed",
    "strict_internal_observed",
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _payload_hash(value: Mapping[str, Any], excluded: str) -> str:
    payload = {key: item for key, item in value.items() if key != excluded}
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _node_metric(node: Mapping[str, Any]) -> float | None:
    metric = node.get("metric")
    return _finite(metric.get("value") if isinstance(metric, Mapping) else metric)


def _strict_clean(node: Mapping[str, Any], *, task_id: str) -> bool:
    audit = node.get("leakage_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    return bool(
        node.get("type") == "RunNode"
        and str(node.get("task") or "") == task_id
        and node.get("is_buggy") is False
        and node.get("is_valid") is True
        and _finite(node.get("metric")) is not None
        and node.get("audit_status") == "clean"
        and node.get("metric_disposition") == "rank_eligible"
        and node.get("memory_disposition") == "positive_eligible"
        and node.get("paper_grade_eligible") is True
        and audit.get("status") == "clean"
        and audit.get("rank_eligible") is True
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
        and node.get("quarantined") is not True
        and node.get("protocol_biased") is not True
    )


def _official_observed(node: Mapping[str, Any]) -> bool:
    audit = node.get("leakage_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    provenance = str(node.get("metric_provenance") or "").lower()
    return bool(
        audit.get("official_score_review") == "pass"
        or audit.get("official_ledger_sha256")
        or "official_kaggle" in provenance
        or "sealed_fixed_holdout_terminal" in provenance
    )


def _attempt_identity(relative_path: str) -> tuple[str, str]:
    parts = PurePosixPath(relative_path).parts
    try:
        runs_index = parts.index("runs")
    except ValueError as exc:
        raise ValueError(f"Journal path is outside a run: {relative_path}") from exc
    if len(parts) <= runs_index + 2:
        raise ValueError(f"Journal path has no logical run/attempt: {relative_path}")
    return str(parts[runs_index + 1]), str(parts[runs_index + 2])


def _journal_entries(
    inventory: Mapping[str, Any], source_root: Path
) -> list[dict[str, Any]]:
    entries = []
    for raw in inventory.get("files") or []:
        if not isinstance(raw, Mapping):
            continue
        relative = str(raw.get("path") or "")
        if PurePosixPath(relative).name != "journal.json":
            continue
        logical_run_id, attempt = _attempt_identity(relative)
        path = (source_root / relative).resolve(strict=True)
        observed_size = path.stat().st_size
        expected_size = int(raw.get("size_bytes") or -1)
        if observed_size != expected_size:
            raise ValueError(
                f"Frozen journal size mismatch: {relative}: "
                f"expected={expected_size} observed={observed_size}"
            )
        observed_sha = _sha256_file(path)
        expected_sha = str(raw.get("sha256") or "")
        if observed_sha != expected_sha:
            raise ValueError(
                f"Frozen journal SHA-256 mismatch: {relative}: "
                f"expected={expected_sha} observed={observed_sha}"
            )
        entries.append(
            {
                "relative_path": relative,
                "path": path,
                "sha256": observed_sha,
                "logical_run_id": logical_run_id,
                "attempt": attempt,
                "base_run_id": f"{logical_run_id}::{attempt}",
            }
        )
    if not entries:
        raise ValueError("Source inventory contains no journal.json entries")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["base_run_id"]].append(entry)
    for base_run_id, rows in grouped.items():
        unique_by_sha = {str(row["sha256"]): row for row in rows}
        unique_rows = sorted(
            unique_by_sha.values(), key=lambda row: str(row["relative_path"])
        )
        canonical = unique_rows[-1]
        for row in unique_rows:
            row["staged_run_id"] = (
                base_run_id
                if row is canonical
                else f"{base_run_id}::source-{str(row['sha256'])[:8]}"
            )
        duplicate_rows = [row for row in rows if row not in unique_rows]
        for row in duplicate_rows:
            source = unique_by_sha[str(row["sha256"])]
            row["staged_run_id"] = source["staged_run_id"]
    return sorted(entries, key=lambda row: str(row["relative_path"]))


def _pair_key(kind: str, before_sha: str, after_sha: str) -> str:
    return _sha256_text(f"{kind}\0{before_sha}\0{after_sha}")


def _representative_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    class_rank = {
        "official_observed": 0,
        "strict_internal_observed": 1,
        "strict_debug_observed": 2,
    }
    return (
        class_rank[str(row["evidence_class"])],
        -float(row.get("metric_improvement") or 0.0),
        str(row["transition_id"]),
    )


def build(
    *,
    graph_path: Path,
    source_inventory_path: Path,
    source_root: Path,
    task_id: str = TASK_ID,
    expected_debug_unique_pairs: int | None = None,
    expected_improve_unique_pairs: int | None = None,
) -> dict[str, Any]:
    graph_path = graph_path.resolve(strict=True)
    source_inventory_path = source_inventory_path.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    graph = _read_object(graph_path)
    inventory = _read_object(source_inventory_path)
    graph_nodes = {
        str(row.get("id") or ""): row
        for row in graph.get("nodes") or []
        if isinstance(row, Mapping) and row.get("id")
    }
    journals = _journal_entries(inventory, source_root)

    code_blobs: dict[str, dict[str, Any]] = {}
    node_bindings: dict[str, dict[str, Any]] = {}
    transition_bindings: list[dict[str, Any]] = []
    observed_parent_child_pairs = 0
    graph_matched_parent_child_pairs = 0

    def bind_node(
        node_id: str,
        code: str,
        *,
        source_journal: str,
        source_journal_sha256: str,
        source_raw_node_id: str,
    ) -> str:
        code_sha = _sha256_text(code)
        graph_node = graph_nodes.get(node_id)
        if not isinstance(graph_node, Mapping) or graph_node.get("type") != "RunNode":
            raise ValueError(f"Resolver endpoint is missing from RunForest: {node_id}")
        graph_sha = str(graph_node.get("code_sha256") or "")
        if graph_sha != code_sha:
            raise ValueError(
                f"Resolver endpoint code identity mismatch: {node_id}: "
                f"graph={graph_sha} journal={code_sha}"
            )
        existing = node_bindings.get(node_id)
        current = {
            "node_id": node_id,
            "code_sha256": code_sha,
            "source_journal": source_journal,
            "source_journal_sha256": source_journal_sha256,
            "source_raw_node_id": source_raw_node_id,
        }
        if existing is not None and existing["code_sha256"] != code_sha:
            raise ValueError(f"Conflicting resolver endpoint code: {node_id}")
        node_bindings.setdefault(node_id, current)
        code_blobs.setdefault(
            code_sha,
            {
                "code_sha256": code_sha,
                "code": code,
            },
        )
        return code_sha

    for journal in journals:
        payload = _read_object(Path(journal["path"]))
        raw_nodes = {
            str(row.get("id") or ""): row
            for row in payload.get("nodes") or []
            if isinstance(row, Mapping) and row.get("id")
        }
        for child_raw_id, parent_raw_id in sorted(
            (payload.get("node2parent") or {}).items()
        ):
            child_raw_id = str(child_raw_id)
            parent_raw_id = str(parent_raw_id)
            child_raw = raw_nodes.get(child_raw_id)
            parent_raw = raw_nodes.get(parent_raw_id)
            if child_raw is None or parent_raw is None:
                continue
            before_code = str(parent_raw.get("code") or "")
            after_code = str(child_raw.get("code") or "")
            if not before_code.strip() or not after_code.strip():
                continue
            observed_parent_child_pairs += 1
            run_id = str(journal["staged_run_id"])
            transition_id = (
                f"run::{run_id}::transition::"
                f"{parent_raw_id[:12]}::{child_raw_id[:12]}"
            )
            transition = graph_nodes.get(transition_id)
            if not isinstance(transition, Mapping) or transition.get("type") != "Transition":
                raise ValueError(
                    f"Frozen journal transition is absent from active graph: {transition_id}"
                )
            graph_matched_parent_child_pairs += 1
            outcome = str(transition.get("outcome") or "")
            if outcome not in {"debug_fixed", "metric_improved"}:
                continue
            child_id = str(transition.get("child_node_id") or "")
            parent_id = str(transition.get("parent_node_id") or "")
            child = graph_nodes.get(child_id, {})
            if not _strict_clean(child, task_id=task_id):
                continue
            evidence_class = (
                "strict_debug_observed"
                if outcome == "debug_fixed"
                else "official_observed"
                if _official_observed(child)
                else "strict_internal_observed"
            )
            before_sha = bind_node(
                parent_id,
                before_code,
                source_journal=str(journal["relative_path"]),
                source_journal_sha256=str(journal["sha256"]),
                source_raw_node_id=parent_raw_id,
            )
            after_sha = bind_node(
                child_id,
                after_code,
                source_journal=str(journal["relative_path"]),
                source_journal_sha256=str(journal["sha256"]),
                source_raw_node_id=child_raw_id,
            )
            transition_bindings.append(
                {
                    "transition_id": transition_id,
                    "task_id": task_id,
                    "outcome": outcome,
                    "evidence_class": evidence_class,
                    "pair_key": _pair_key(outcome, before_sha, after_sha),
                    "parent_node_id": parent_id,
                    "child_node_id": child_id,
                    "before_code_sha256": before_sha,
                    "after_code_sha256": after_sha,
                    "parent_metric": transition.get("parent_metric"),
                    "child_metric": transition.get("child_metric"),
                    "metric_improvement": transition.get("metric_improvement"),
                    "metric_provenance": child.get("metric_provenance"),
                    "stage_pair": transition.get("stage_pair"),
                    "source_journal": str(journal["relative_path"]),
                    "source_journal_sha256": str(journal["sha256"]),
                }
            )

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transition_bindings:
        by_pair[str(row["pair_key"])].append(row)
    pairs = []
    for pair_key, aliases in sorted(by_pair.items()):
        aliases = sorted(aliases, key=_representative_key)
        representative = aliases[0]
        pairs.append(
            {
                "pair_key": pair_key,
                "outcome": representative["outcome"],
                "evidence_class": representative["evidence_class"],
                "representative_transition_id": representative["transition_id"],
                "alias_transition_ids": [row["transition_id"] for row in aliases],
                "before_code_sha256": representative["before_code_sha256"],
                "after_code_sha256": representative["after_code_sha256"],
            }
        )

    debug_pairs = sum(row["outcome"] == "debug_fixed" for row in pairs)
    improve_pairs = sum(row["outcome"] == "metric_improved" for row in pairs)
    if (
        expected_debug_unique_pairs is not None
        and debug_pairs != expected_debug_unique_pairs
    ):
        raise ValueError(
            "Strict-clean Debug unique-pair coverage mismatch: "
            f"expected={expected_debug_unique_pairs} observed={debug_pairs}"
        )
    if (
        expected_improve_unique_pairs is not None
        and improve_pairs != expected_improve_unique_pairs
    ):
        raise ValueError(
            "Strict-clean Improve unique-pair coverage mismatch: "
            f"expected={expected_improve_unique_pairs} observed={improve_pairs}"
        )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": task_id,
        "graph_file_sha256": _sha256_file(graph_path),
        "source_inventory_file_sha256": _sha256_file(source_inventory_path),
        "source_inventory_sha256": inventory.get("inventory_sha256"),
        "journal_count": len(journals),
        "journal_sha256s": sorted(str(row["sha256"]) for row in journals),
        "observed_parent_child_pair_count": observed_parent_child_pairs,
        "graph_matched_parent_child_pair_count": graph_matched_parent_child_pairs,
        "transition_count": len(transition_bindings),
        "unique_pair_count": len(pairs),
        "debug_transition_count": sum(
            row["outcome"] == "debug_fixed" for row in transition_bindings
        ),
        "debug_unique_pair_count": debug_pairs,
        "improve_transition_count": sum(
            row["outcome"] == "metric_improved" for row in transition_bindings
        ),
        "improve_unique_pair_count": improve_pairs,
        "node_binding_count": len(node_bindings),
        "unique_code_count": len(code_blobs),
        "code_bytes": sum(len(str(row["code"])) for row in code_blobs.values()),
        "evidence_class_counts": {
            evidence_class: sum(
                row["evidence_class"] == evidence_class
                for row in transition_bindings
            )
            for evidence_class in sorted(EVIDENCE_CLASSES)
        },
        "code_blobs": [code_blobs[key] for key in sorted(code_blobs)],
        "nodes": [node_bindings[key] for key in sorted(node_bindings)],
        "transitions": sorted(
            transition_bindings, key=lambda row: str(row["transition_id"])
        ),
        "pairs": pairs,
    }
    result["capsule_sha256"] = _payload_hash(result, "capsule_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--expected-debug-unique-pairs", type=int)
    parser.add_argument("--expected-improve-unique-pairs", type=int)
    args = parser.parse_args()
    payload = build(
        graph_path=args.graph,
        source_inventory_path=args.source_inventory,
        source_root=args.source_root,
        task_id=str(args.task_id),
        expected_debug_unique_pairs=args.expected_debug_unique_pairs,
        expected_improve_unique_pairs=args.expected_improve_unique_pairs,
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable capsule: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "capsule_sha256",
                    "journal_count",
                    "transition_count",
                    "unique_pair_count",
                    "debug_transition_count",
                    "debug_unique_pair_count",
                    "improve_transition_count",
                    "improve_unique_pair_count",
                    "unique_code_count",
                    "code_bytes",
                )
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
