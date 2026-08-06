#!/usr/bin/env python3
"""Recover full RunForest code and exact Debug diffs from retained journals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "mlevolve_recipe_implementation_capsules_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _required_inventory(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    required_nodes: dict[str, dict[str, str]] = {}
    required_transitions: dict[str, dict[str, str]] = {}

    for task_id, records in (evidence.get("selected_evidence") or {}).items():
        for record in records or []:
            node_id = str(record.get("node_id") or "")
            code_sha = str(record.get("code_sha256") or "")
            if not node_id or len(code_sha) != 64:
                raise ValueError(f"Selected evidence lacks code identity: {node_id}")
            previous = required_nodes.get(node_id)
            current = {"task_id": str(task_id), "code_sha256": code_sha}
            if previous is not None and previous != current:
                raise ValueError(f"Conflicting selected evidence node: {node_id}")
            required_nodes[node_id] = current

    for task_id, records in (
        evidence.get("selected_repair_evidence") or {}
    ).items():
        for record in records or []:
            transition_id = str(record.get("transition_id") or "")
            parent_id = str(record.get("failure_node_id") or "")
            child_id = str(record.get("successful_node_id") or "")
            parent_sha = str(record.get("failure_node_code_sha256") or "")
            child_sha = str(record.get("successful_node_code_sha256") or "")
            if (
                not transition_id
                or not parent_id
                or not child_id
                or len(parent_sha) != 64
                or len(child_sha) != 64
            ):
                raise ValueError(
                    f"Repair evidence lacks before/after code identity: {transition_id}"
                )
            for node_id, code_sha in (
                (parent_id, parent_sha),
                (child_id, child_sha),
            ):
                current = {"task_id": str(task_id), "code_sha256": code_sha}
                previous = required_nodes.get(node_id)
                if previous is not None and previous != current:
                    raise ValueError(f"Conflicting repair evidence node: {node_id}")
                required_nodes[node_id] = current
            required_transitions[transition_id] = {
                "task_id": str(task_id),
                "parent_node_id": parent_id,
                "child_node_id": child_id,
            }
    return required_nodes, required_transitions


def _scan_journals(
    roots: list[Path], required_hashes: set[str]
) -> tuple[dict[str, dict[str, str]], int, int]:
    found: dict[str, dict[str, str]] = {}
    journals_scanned = 0
    nodes_scanned = 0
    journal_paths = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.resolve(strict=True).rglob("journal.json")
        },
        key=str,
    )
    for journal_path in journal_paths:
        try:
            journal = _read_json(journal_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        journals_scanned += 1
        for node in journal.get("nodes") or []:
            if not isinstance(node, Mapping):
                continue
            code = node.get("code")
            if not isinstance(code, str) or not code.strip():
                continue
            nodes_scanned += 1
            code_sha = _code_sha256(code)
            if code_sha not in required_hashes or code_sha in found:
                continue
            found[code_sha] = {
                "code": code,
                "source_journal": str(journal_path),
                "source_raw_node_id": str(node.get("id") or ""),
            }
        if set(found) == required_hashes:
            break
    return found, journals_scanned, nodes_scanned


def build(
    evidence_path: Path,
    journal_roots: list[Path],
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    evidence = _read_json(evidence_path.resolve(strict=True))
    if evidence.get("schema") != "mlevolve_recipe_distillation_evidence_v1":
        raise ValueError("Unsupported Recipe evidence schema")
    required_nodes, required_transitions = _required_inventory(evidence)
    required_hashes = {
        record["code_sha256"] for record in required_nodes.values()
    }
    found, journals_scanned, nodes_scanned = _scan_journals(
        journal_roots, required_hashes
    )
    missing_hashes = sorted(required_hashes - set(found))
    if missing_hashes and not allow_missing:
        missing_nodes = sorted(
            node_id
            for node_id, record in required_nodes.items()
            if record["code_sha256"] in missing_hashes
        )
        raise ValueError(
            "Retained journals do not cover the frozen Recipe evidence: "
            f"missing_hashes={len(missing_hashes)} missing_nodes={missing_nodes[:12]}"
        )

    nodes = []
    missing_node_ids = sorted(
        node_id
        for node_id, record in required_nodes.items()
        if record["code_sha256"] not in found
    )
    for node_id, record in sorted(required_nodes.items()):
        source = found.get(record["code_sha256"])
        if source is None:
            continue
        nodes.append(
            {
                "node_id": node_id,
                "task_id": record["task_id"],
                "code_sha256": record["code_sha256"],
                "code": source["code"],
                "source_journal": source["source_journal"],
                "source_raw_node_id": source["source_raw_node_id"],
            }
        )
    available_node_ids = {row["node_id"] for row in nodes}
    transitions = [
        {"transition_id": transition_id, **record}
        for transition_id, record in sorted(required_transitions.items())
        if record["parent_node_id"] in available_node_ids
        and record["child_node_id"] in available_node_ids
    ]
    available_transition_ids = {row["transition_id"] for row in transitions}
    missing_transition_ids = sorted(
        set(required_transitions) - available_transition_ids
    )
    return {
        "schema": SCHEMA,
        "coverage_policy": (
            "all selected L1/L2 evidence nodes and all selected L3 before/after nodes"
        ),
        "required_node_ids": sorted(required_nodes),
        "required_transition_ids": sorted(required_transitions),
        "missing_node_ids": missing_node_ids,
        "missing_transition_ids": missing_transition_ids,
        "complete_recipe_coverage": not missing_node_ids,
        "node_count": len(nodes),
        "transition_count": len(transitions),
        "unique_code_count": len(required_hashes),
        "journals_scanned": journals_scanned,
        "journal_nodes_scanned": nodes_scanned,
        "nodes": nodes,
        "transitions": transitions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--journal-root", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Retain explicit missing-node receipts when old journals no longer exist.",
    )
    args = parser.parse_args()
    payload = build(
        args.evidence, args.journal_root, allow_missing=args.allow_missing
    )
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
                    "node_count",
                    "transition_count",
                    "unique_code_count",
                    "journals_scanned",
                    "journal_nodes_scanned",
                    "complete_recipe_coverage",
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
