#!/usr/bin/env python3
"""Load a v10 bundle through the production snapshot and hybrid-memory layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from authority.memory_snapshot import MemorySnapshotLoader
from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.bundle_root.resolve(strict=True)
    current = read_json(root / "CURRENT.json")
    bundle = (root / current["bundle_path"]).resolve(strict=True)
    manifest = read_json(bundle / "manifest.json")
    artifacts = manifest["artifact_hashes"]
    protocol_ref = str(manifest["active_protocol_ref"])
    policy_version = str(manifest["authority_policy_version"])

    snapshot = MemorySnapshotLoader(root).load(
        session_overlay_path=args.overlay,
        active_protocol_ref=protocol_ref,
        authority_policy_version=policy_version,
        verify_artifacts=True,
    )
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(bundle / "runforest" / "graph.json"),
        index_path=str(bundle / "runforest" / "index.npz"),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        scoring_mode="poincare",
        retrieval_control="layered_strategy",
        enable_agentic=False,
        top_k=6,
        memory_snapshot=snapshot,
        recipe_sop_path=str(bundle / "recipe" / "recipe_sops.json"),
        recipe_sop_file_sha256=artifacts["recipe/recipe_sops.json"],
        recipe_sop_bundle_sha256=manifest["recipe_sop_bundle_sha256"],
        recipe_evidence_path=str(bundle / "recipe" / "evidence_manifest.json"),
        recipe_evidence_file_sha256=artifacts[
            "recipe/evidence_manifest.json"
        ],
        recipe_evidence_manifest_sha256=manifest[
            "recipe_evidence_manifest_sha256"
        ],
        recipe_implementation_path=str(
            bundle / "recipe" / "implementation_capsules.json"
        ),
        evidence_resolver_enabled=True,
        transition_evidence_capsules_path=str(
            bundle / "recipe" / "transition_evidence_capsules.json"
        ),
        transition_evidence_capsules_sha256=artifacts[
            "recipe/transition_evidence_capsules.json"
        ],
        evidence_resolver_max_pairs=3,
        visibility_mode="shadow",
        visibility_active_protocol=protocol_ref,
        visibility_policy_version=policy_version,
        visibility_task_id="leaf-classification",
        visibility_bundle_version=manifest["bundle_version"],
    )
    counts = {
        "l1": sum(
            layer.nodes[node_id].get("abstraction_level") == "L1_strategy"
            for node_id in layer._recipe_sop_ids
        ),
        "l2": sum(
            layer.nodes[node_id].get("abstraction_level") == "L2_tactic"
            for node_id in layer._recipe_sop_ids
        ),
        "l3": sum(
            layer.nodes[node_id].get("abstraction_level") == "L3_repair"
            for node_id in layer._recipe_sop_ids
        ),
    }
    if counts != {"l1": 8, "l2": 10, "l3": 288}:
        raise ValueError(f"Unexpected normalized Recipe counts: {counts}")
    clause_receipt = layer.base_clause_receipt
    if clause_receipt.get("status") != "loaded":
        raise ValueError(f"Base clauses did not load: {clause_receipt}")
    if clause_receipt.get("clause_count") != 288:
        raise ValueError(f"Unexpected clause count: {clause_receipt}")
    if clause_receipt.get("sop_count") != 288:
        raise ValueError(f"Unexpected formal SOP count: {clause_receipt}")
    evidence_receipt = layer.recipe_evidence_receipt
    if evidence_receipt.get("selected_node_count") != 4:
        raise ValueError(f"Unexpected positive evidence count: {evidence_receipt}")
    if evidence_receipt.get("selected_repair_transition_count") != 6:
        raise ValueError(f"Unexpected repair evidence count: {evidence_receipt}")
    implementation_receipt = layer.recipe_implementation_receipt
    if (
        implementation_receipt.get("required_node_count") != 15
        or implementation_receipt.get("required_transition_count") != 6
        or implementation_receipt.get("node_count") != 15
        or implementation_receipt.get("transition_count") != 6
        or implementation_receipt.get("missing_node_ids")
        or implementation_receipt.get("missing_transition_ids")
        or implementation_receipt.get("complete_recipe_coverage") is not True
    ):
        raise ValueError(
            f"Production implementation capsule load is incomplete: "
            f"{implementation_receipt}"
        )
    resolver_receipt = layer.evidence_resolver_load_receipt
    if resolver_receipt.get("status") != "validated":
        raise ValueError(
            f"Production Evidence Resolver did not validate: {resolver_receipt}"
        )
    expected_new_repairs = {
        f"repair::leaf-classification::v140::{index:03d}"
        for index in range(1, 13)
    }
    if not expected_new_repairs.issubset(layer.nodes):
        raise ValueError("A normalized v140 Repair is absent from runtime memory")

    report = {
        "schema": "leaf_llm_redistilled_memory_v10_runtime_load_smoke_v1",
        "status": "pass",
        "bundle_id": snapshot.base_bundle_id,
        "bundle_version": snapshot.base_bundle.bundle_version,
        "base_manifest_sha256": snapshot.base_bundle.manifest_sha256,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "recipe_sop_receipt": layer.recipe_sop_receipt,
        "recipe_evidence_receipt": evidence_receipt,
        "recipe_implementation_receipt": implementation_receipt,
        "evidence_resolver_load_receipt": resolver_receipt,
        "base_clause_receipt": clause_receipt,
        "normalized_recipe_counts": counts,
        "new_llm_repair_ids_loaded": sorted(expected_new_repairs),
        "overlay_event_count": len(snapshot.session_overlay.events()),
        "bundle_files_unchanged": True,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
