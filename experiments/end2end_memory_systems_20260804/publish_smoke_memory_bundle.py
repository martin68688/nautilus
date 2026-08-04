#!/usr/bin/env python3
"""Publish the Aerial End2End Smoke same-domain seed-heldout Base Bundle.

The script consumes one reviewed immutable parent Bundle.  For every target it
retains only memory-side runs from other tasks in the same canonical domain,
publishes a task/protocol-scoped METHOD_HYPOTHESIS with real historical
execution evidence, and emits a CURRENT-selected immutable Base Bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
for value in (ROOT / "mlevolve", TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from authority.memory_snapshot import ImmutableBaseBundle  # noqa: E402
from authority.protocol_registry import ProtocolRegistry  # noqa: E402
from publish_tier2_formal_child_bundle import (  # noqa: E402
    publish_formal_child_bundle,
)
from schema import sha256_file  # noqa: E402


SCHEMA = "mlevolve_end2end_smoke_memory_spec_v1"
BINDING_SCHEMA = "mlevolve_end2end_smoke_memory_binding_v1"
DEFAULT_SPEC = ROOT / (
    "experiments/end2end_memory_systems_20260804/"
    "smoke_memory_spec.json"
)


def payload_hash(value: Mapping[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("Unsupported End2End Smoke memory spec")
    expected = payload_hash(value, "spec_sha256")
    if value.get("spec_sha256") != expected:
        raise ValueError("End2End Smoke memory spec hash mismatch")
    tasks = value.get("tasks") or {}
    if set(tasks) != {"aerial-cactus-identification"}:
        raise ValueError("End2End Smoke memory spec requires only Aerial")
    required = int(value.get("required_formal_sop_count") or 0)
    if required != 5:
        raise ValueError("End2End Smoke requires exactly five formal SOPs")
    for task_id, task in tasks.items():
        methods = task.get("methods") or []
        if len(methods) != required:
            raise ValueError(
                f"End2End Smoke task {task_id} does not have five formal methods"
            )
        identities = {
            (
                str(method.get("source_clause_id") or ""),
                str(method.get("source_run_id") or ""),
                str(method.get("source_node_id") or ""),
                str(method.get("method_text") or ""),
            )
            for method in methods
        }
        if len(identities) != required or any(not all(value[:3]) for value in identities):
            raise ValueError(f"End2End Smoke task {task_id} has invalid method specs")
    return value


def build(
    *,
    spec_path: Path,
    parent_bundle_path: Path,
    output_root: Path,
    created_at: str,
) -> dict[str, Any]:
    spec = load_spec(spec_path.resolve(strict=True))
    parent = ImmutableBaseBundle.load(
        parent_bundle_path.resolve(strict=True), verify_artifacts=True
    )
    parent_spec = spec["parent_bundle"]
    if parent.bundle_id != parent_spec["bundle_id"]:
        raise ValueError("End2End Smoke parent Bundle ID mismatch")
    if parent.manifest_sha256 != parent_spec["manifest_sha256"]:
        raise ValueError("End2End Smoke parent Bundle manifest mismatch")
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to reuse production memory root: {output_root}"
        )
    output_root.mkdir(parents=True)

    rows: dict[str, Any] = {}
    for task_id, task in sorted(spec["tasks"].items()):
        protocol_path = (ROOT / str(task["protocol_file"])).resolve(strict=True)
        protocol = ProtocolRegistry(protocol_path.parent).resolve(
            f"{protocol_path.stem.rsplit('-v', 1)[0]}@1"
        )
        if protocol.ref().key() != task["protocol_ref"]:
            raise ValueError(f"Protocol binding drift for {task_id}")
        methods = [dict(value) for value in task["methods"]]
        primary = methods[0]
        task_root = output_root / task_id
        result = publish_formal_child_bundle(
            parent_bundle=parent.path,
            expected_parent_manifest_sha256=parent.manifest_sha256,
            publication_root=task_root,
            bundle_id=str(task["bundle_id"]),
            bundle_version=str(task["bundle_version"]),
            target_task_id=task_id,
            target_task_family=str(task["target_task_family"]),
            target_domain=str(task["target_domain"]),
            split_mode=str(spec["split_mode"]),
            source_clause_id=str(primary["source_clause_id"]),
            source_run_id=str(primary["source_run_id"]),
            source_node_id=str(primary["source_node_id"]),
            protocol_file=protocol_path,
            publication_class=str(spec["publication_class"]),
            agent_seeds=tuple(int(value) for value in spec["agent_seeds"]),
            additional_source_methods=methods[1:],
            created_at=created_at,
        )
        publication = result["publication"]
        current_path = task_root / "CURRENT.json"
        child = ImmutableBaseBundle.load(
            task_root / "bundles" / str(task["bundle_version"]),
            verify_artifacts=True,
        )
        formal_methods = [dict(value) for value in publication["formal_methods"]]
        if len(formal_methods) != int(spec["required_formal_sop_count"]):
            raise ValueError(f"Formal SOP population mismatch for {task_id}")
        graph = child.read_json("runforest/graph.json")
        graph_nodes = {str(node["id"]): node for node in graph.get("nodes") or []}
        graph_edges = [dict(edge) for edge in graph.get("edges") or []]
        for method in formal_methods:
            transition_id = str(method["supporting_transition_ref"])
            sop_id = str(method["formal_sop_id"])
            clause_ids = set(map(str, method["formal_clause_ids"]))
            if graph_nodes.get(transition_id, {}).get("type") != "Transition":
                raise ValueError(f"Formal transition missing for {task_id}: {transition_id}")
            if not any(
                edge.get("src") == transition_id
                and edge.get("dst") == sop_id
                and edge.get("kind") == "navigation_attached_to"
                and bool(clause_ids & set(map(str, edge.get("clause_ids") or [])))
                for edge in graph_edges
            ):
                raise ValueError(f"Formal clause-scoped edge missing for {task_id}")
        rows[task_id] = {
            "bundle_root": str(task_root),
            "bundle_id": child.bundle_id,
            "bundle_version": child.bundle_version,
            "bundle_manifest_sha256": child.manifest_sha256,
            "bundle_manifest_file_sha256": child.manifest_file_sha256,
            "current_file_sha256": sha256_file(current_path),
            "graph_sha256": sha256_file(child.path / "runforest" / "graph.json"),
            "index_sha256": sha256_file(child.path / "runforest" / "index.npz"),
            "protocol_ref": str(publication["formal_protocol_ref"]),
            "formal_clause_id": str(publication["formal_clause_id"]),
            "formal_debug_clause_id": str(publication["formal_debug_clause_id"]),
            "formal_claim_id": str(publication["formal_claim_id"]),
            "formal_receipt_ids": list(publication["formal_receipt_ids"]),
            "formal_method_count": len(formal_methods),
            "formal_methods": formal_methods,
            "source_task_id": str(publication["source_task_id"]),
            "source_task_ids": list(publication["source_task_ids"]),
            "split_mode": str(publication["split_mode"]),
            "certification_level": str(child.manifest["certification_level"]),
        }
        parent.assert_unchanged()

    binding: dict[str, Any] = {
        "schema": BINDING_SCHEMA,
        "status": "published_verified_same_domain_seed_heldout",
        "spec_sha256": str(spec["spec_sha256"]),
        "parent_bundle_id": parent.bundle_id,
        "parent_manifest_sha256": parent.manifest_sha256,
        "created_at": created_at,
        "tasks": rows,
        "binding_sha256": "",
    }
    binding["binding_sha256"] = payload_hash(binding, "binding_sha256")
    binding_path = output_root / "SMOKE_MEMORY_BINDING.json"
    binding_path.write_text(
        json.dumps(binding, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--parent-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    result = build(
        spec_path=args.spec,
        parent_bundle_path=args.parent_bundle,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
