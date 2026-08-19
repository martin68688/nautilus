#!/usr/bin/env python3
"""Independently audit an immutable v10 Leaf LLM-redistilled bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row["id"]) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--parent-bundle", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.bundle_root.resolve(strict=True)
    parent = args.parent_bundle.resolve(strict=True)
    current = read_json(root / "CURRENT.json")
    bundle = (root / current["bundle_path"]).resolve(strict=True)
    manifest = read_json(bundle / "manifest.json")
    recipe = read_json(bundle / "recipe" / "recipe_sops.json")
    evidence = read_json(bundle / "recipe" / "evidence_manifest.json")
    implementation = read_json(
        bundle / "recipe" / "implementation_capsules.json"
    )
    implementation_projection = read_json(
        bundle / "reports" / "recipe_implementation_projection.json"
    )
    replay_targets = read_json(
        bundle / "reports" / "leaf_official_replay_targets_v139.json"
    )
    replay_projection = read_json(
        bundle / "reports" / "replay_target_recipe_projection.json"
    )
    teacher = read_json(
        args.teacher_output / "teacher_response_gpt56sol.json"
    )
    published_teacher = read_json(
        bundle / "recipe" / "teacher_response_gpt56sol.json"
    )
    atomic = read_json(bundle / "recipe" / "atomic_claims.json")
    parent_atomic = read_json(parent / "recipe" / "atomic_claims.json")
    graph = read_json(bundle / "runforest" / "graph.json")
    clauses = read_jsonl(bundle / "sop" / "clauses.jsonl")
    parent_clauses = read_jsonl(parent / "sop" / "clauses.jsonl")

    check(
        payload_hash(current, "pointer_sha256") == current["pointer_sha256"],
        "CURRENT canonical hash mismatch",
    )
    check(
        payload_hash(manifest, "manifest_sha256") == manifest["manifest_sha256"],
        "manifest canonical hash mismatch",
    )
    check(
        current["manifest_sha256"] == manifest["manifest_sha256"],
        "CURRENT does not bind manifest",
    )
    check(
        payload_hash(recipe, "bundle_sha256") == recipe["bundle_sha256"],
        "Recipe canonical hash mismatch",
    )
    check(
        payload_hash(evidence, "manifest_sha256") == evidence["manifest_sha256"],
        "Recipe evidence canonical hash mismatch",
    )
    check(
        recipe["evidence_manifest_sha256"] == evidence["manifest_sha256"],
        "Recipe does not bind evidence manifest",
    )
    check(
        recipe["bundle_sha256"] == manifest["recipe_sop_bundle_sha256"],
        "Manifest does not bind Recipe",
    )

    expected_code_by_node: dict[str, str] = {}

    def require_implementation_node(node_id: object, code_sha256: object) -> None:
        canonical_id = str(node_id or "")
        canonical_sha = str(code_sha256 or "")
        check(
            bool(canonical_id) and len(canonical_sha) == 64,
            "Selected evidence has an incomplete implementation identity",
        )
        previous = expected_code_by_node.setdefault(canonical_id, canonical_sha)
        check(
            previous == canonical_sha,
            f"Selected evidence has conflicting code hashes: {canonical_id}",
        )

    for records in (evidence.get("selected_evidence") or {}).values():
        check(isinstance(records, list), "Positive evidence group is malformed")
        for row in records:
            require_implementation_node(row.get("node_id"), row.get("code_sha256"))

    expected_transition_endpoints: dict[str, tuple[str, str]] = {}
    for records in (evidence.get("selected_repair_evidence") or {}).values():
        check(isinstance(records, list), "Repair evidence group is malformed")
        for row in records:
            transition_id = str(row.get("transition_id") or "")
            parent_id = str(row.get("failure_node_id") or "")
            child_id = str(row.get("successful_node_id") or "")
            check(
                bool(transition_id)
                and transition_id not in expected_transition_endpoints,
                f"Duplicate or missing repair transition: {transition_id}",
            )
            require_implementation_node(
                parent_id, row.get("failure_node_code_sha256")
            )
            require_implementation_node(
                child_id, row.get("successful_node_code_sha256")
            )
            expected_transition_endpoints[transition_id] = (parent_id, child_id)

    required_node_ids = set(expected_code_by_node)
    required_transition_ids = set(expected_transition_endpoints)
    declared_node_ids = set(
        map(str, implementation.get("required_node_ids") or [])
    )
    declared_transition_ids = set(
        map(str, implementation.get("required_transition_ids") or [])
    )
    capsule_nodes = {
        str(row.get("node_id") or ""): row
        for row in implementation.get("nodes") or []
    }
    capsule_transitions = {
        str(row.get("transition_id") or ""): row
        for row in implementation.get("transitions") or []
    }
    check(
        implementation.get("schema")
        == "mlevolve_recipe_implementation_capsules_v1",
        "Unsupported implementation capsule schema",
    )
    check(
        declared_node_ids == required_node_ids,
        "Implementation required-node authority differs from production evidence",
    )
    check(
        set(capsule_nodes) == required_node_ids,
        "Implementation node inventory is not an exact production projection",
    )
    check(
        declared_transition_ids == required_transition_ids,
        "Implementation required-transition authority differs from production evidence",
    )
    check(
        set(capsule_transitions) == required_transition_ids,
        "Implementation transition inventory is not an exact production projection",
    )
    check(
        len(required_node_ids) == 15 and len(required_transition_ids) == 6,
        "Unexpected production implementation authority counts",
    )
    graph_by_id = {str(row["id"]): row for row in graph["nodes"]}
    for node_id, row in capsule_nodes.items():
        code = row.get("code")
        code_sha = str(row.get("code_sha256") or "")
        check(
            isinstance(code, str) and bool(code.strip()),
            f"Implementation capsule has no code: {node_id}",
        )
        check(
            hashlib.sha256(code.encode("utf-8")).hexdigest() == code_sha,
            f"Implementation capsule code hash mismatch: {node_id}",
        )
        check(
            code_sha == expected_code_by_node[node_id],
            f"Implementation capsule differs from selected evidence: {node_id}",
        )
        check(
            str(graph_by_id.get(node_id, {}).get("code_sha256") or "")
            == code_sha,
            f"Implementation capsule differs from RunForest: {node_id}",
        )
        check(
            bool(str(row.get("source_journal") or "")),
            f"Implementation capsule provenance is incomplete: {node_id}",
        )
    for transition_id, row in capsule_transitions.items():
        parent_id, child_id = expected_transition_endpoints[transition_id]
        check(
            str(row.get("parent_node_id") or "") == parent_id
            and str(row.get("child_node_id") or "") == child_id,
            f"Implementation transition endpoints differ: {transition_id}",
        )
    check(
        implementation.get("node_count") == 15
        and implementation.get("transition_count") == 6
        and implementation.get("complete_recipe_coverage") is True
        and not implementation.get("missing_node_ids")
        and not implementation.get("missing_transition_ids"),
        "Implementation capsule coverage receipt is incomplete",
    )
    check(
        implementation_projection.get("status") == "pass"
        and implementation_projection.get("required_node_count") == 15
        and implementation_projection.get("required_transition_count") == 6
        and implementation_projection.get("projected_node_count") == 15
        and implementation_projection.get("projected_transition_count") == 6,
        "Implementation projection report is incomplete",
    )
    check(
        implementation_projection.get("implementation_capsule_file_sha256")
        == sha256_file(bundle / "recipe" / "implementation_capsules.json"),
        "Implementation projection report does not bind the capsule file",
    )

    artifact_hashes = manifest["artifact_hashes"]
    for relative, expected in artifact_hashes.items():
        check(
            sha256_file(bundle / relative) == expected,
            f"Artifact hash mismatch: {relative}",
        )
    actual_artifacts = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    check(
        actual_artifacts == set(artifact_hashes),
        "Artifact inventory differs from manifest",
    )

    check(teacher == published_teacher, "Published Teacher response changed")
    check(
        teacher["teacher_response_sha256"]
        == "98ef9a2746ec8c21221faba388b1557c45138be2b4758d814723682953ba3504",
        "Unexpected Teacher response identity",
    )
    check(
        sha256_file(bundle / "recipe" / "teacher_response_gpt56sol.json")
        == "df8d960a177df133768e643d476b92af540b6e9b52aa36fbdc71787efd6e9143",
        "Unexpected Teacher response file hash",
    )
    check(
        recipe["teacher_response_sha256"] == teacher["teacher_response_sha256"],
        "Recipe does not bind Teacher response",
    )
    check(
        evidence["distillation_teacher_response_sha256"]
        == teacher["teacher_response_sha256"],
        "Evidence does not bind Teacher response",
    )

    recipe_nodes = recipe["nodes"]
    level_counts = Counter(str(row.get("abstraction_level") or "") for row in recipe_nodes)
    expected_level_counts = {
        "L1_recipe": 8,
        "L2_tactic": 10,
        "L3_repair": 288,
    }
    check(dict(level_counts) == expected_level_counts, "Unexpected Recipe level counts")
    check(len(recipe_nodes) == 306, "Unexpected Recipe node count")
    recipe_ids = ids(recipe_nodes)
    check(len(recipe_ids) == len(recipe_nodes), "Duplicate Recipe node ID")
    check(
        ids(teacher["recipes"]).issubset(recipe_ids)
        and ids(teacher["tactics"]).issubset(recipe_ids)
        and ids(teacher["repairs"]).issubset(recipe_ids),
        "Teacher node missing from published Recipe",
    )
    check(
        not any(
            node_id.startswith("recipe::leaf-classification::00")
            or node_id.startswith("tactic::leaf-classification::00")
            or node_id.startswith("repair-teacher::leaf-classification::")
            for node_id in recipe_ids
        ),
        "Withdrawn template strategy node survived publication",
    )

    claim_by_id = {str(row["id"]): row for row in atomic["claims"]}
    check(atomic == parent_atomic, "Atomic claim bundle changed")
    graph_sop_ids = {
        str(row["id"]) for row in graph["nodes"] if row.get("type") == "SOP"
    }
    available_sop_ids = recipe_ids | graph_sop_ids
    replay_target_sop_ids = {
        str(sop_id)
        for target in replay_targets.get("targets") or []
        for sop_id in target.get("sop_ids") or []
    }
    check(
        replay_target_sop_ids.issubset(available_sop_ids),
        "Replay target cites a SOP absent from production memory",
    )
    l1_by_candidate: dict[str, set[str]] = {}
    for row in recipe_nodes:
        if row.get("abstraction_level") != "L1_recipe":
            continue
        for support in row.get("official_support") or []:
            candidate_id = str(support.get("candidate_id") or "")
            if candidate_id:
                l1_by_candidate.setdefault(candidate_id, set()).add(str(row["id"]))
    for target in replay_targets.get("targets") or []:
        target_id = str(target.get("target_id") or "")
        expected = sorted(l1_by_candidate.get(target_id, set()))
        if expected:
            check(
                sorted(map(str, target.get("sop_ids") or [])) == expected,
                f"Replay target lacks its canonical L1 Recipe: {target_id}",
            )
    check(
        replay_projection.get("status") == "pass"
        and replay_projection.get("invalid_projected_sop_id_count") == 0
        and replay_projection.get("canonical_l1_binding_count") == 4
        and payload_hash(replay_projection, "report_sha256")
        == replay_projection.get("report_sha256"),
        "Replay target Recipe projection report is invalid",
    )
    clause_ids = {str(row["clause_id"]) for row in clauses}
    check(len(clauses) == 288 and len(clause_ids) == 288, "Clause count mismatch")
    check(
        all(str(row["sop_id"]) in recipe_ids | graph_sop_ids for row in clauses),
        "Formal clause lacks a container",
    )
    for clause in clauses:
        claim_refs = [
            *(clause.get("claim_refs") or []),
            *(clause.get("additional_source_claim_refs") or []),
        ]
        check(bool(claim_refs), f"Clause lacks claim authority: {clause['clause_id']}")
        check(
            all(str(ref) in claim_by_id for ref in claim_refs),
            f"Clause references an absent claim: {clause['clause_id']}",
        )
        check(
            all(
                claim_by_id[str(ref)].get("claim_status")
                == "authorized_debug_only"
                for ref in claim_refs
            ),
            f"Clause uses non-debug authority: {clause['clause_id']}",
        )

    retained_parent_clauses = {
        str(row["clause_id"]): row
        for row in parent_clauses
        if str(row.get("sop_id") or "").startswith(
            "repair-claim::leaf-classification::"
        )
    }
    retained_published_clauses = {
        str(row["clause_id"]): row
        for row in clauses
        if str(row.get("sop_id") or "").startswith(
            "repair-claim::leaf-classification::"
        )
    }
    check(
        retained_parent_clauses == retained_published_clauses,
        "Retained atomic clauses changed",
    )
    new_repairs = [
        row
        for row in recipe_nodes
        if str(row["id"]).startswith("repair::leaf-classification::v140::")
    ]
    check(len(new_repairs) == 12, "Expected 12 new LLM Repairs")
    for repair in new_repairs:
        refs = [str(ref) for ref in repair.get("source_claim_ids") or []]
        check(bool(refs), f"New Repair lacks claim authority: {repair['id']}")
        check(
            all(
                ref in claim_by_id
                and claim_by_id[ref].get("claim_status") == "authorized_debug_only"
                for ref in refs
            ),
            f"New Repair uses unauthorized claims: {repair['id']}",
        )
        check(
            bool(repair.get("source_atomic_container_ids")),
            f"New Repair lacks source atomic containers: {repair['id']}",
        )

    readonly_failures = []
    symlinks = []
    secret_prefix_hits = []
    for path in [root, *root.rglob("*")]:
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            symlinks.append(str(path))
            continue
        if mode & 0o222:
            readonly_failures.append(str(path))
        if path.is_file():
            data = path.read_bytes()
            if b"agt_codex_" in data:
                secret_prefix_hits.append(str(path))
    check(not symlinks, "Published bundle contains symlinks")
    check(not readonly_failures, "Published bundle is not immutable")
    check(not secret_prefix_hits, "A credential prefix appears in the bundle")

    report = {
        "schema": "leaf_llm_redistilled_memory_v10_independent_audit_v1",
        "status": "pass",
        "bundle_root": str(root),
        "bundle_path": str(bundle),
        "bundle_version": manifest["bundle_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": sha256_file(bundle / "manifest.json"),
        "teacher_response_sha256": teacher["teacher_response_sha256"],
        "teacher_response_file_sha256": sha256_file(
            bundle / "recipe" / "teacher_response_gpt56sol.json"
        ),
        "recipe_counts": expected_level_counts,
        "formal_clause_count": len(clauses),
        "retained_atomic_clause_count": len(retained_published_clauses),
        "new_llm_repair_count": len(new_repairs),
        "implementation_capsule_node_count": len(required_node_ids),
        "implementation_capsule_transition_count": len(
            required_transition_ids
        ),
        "implementation_projection_exact": True,
        "replay_target_recipe_projection_exact": True,
        "replay_target_canonical_l1_binding_count": replay_projection[
            "canonical_l1_binding_count"
        ],
        "artifact_count": len(artifact_hashes),
        "atomic_claim_bundle_unchanged": True,
        "retained_atomic_clauses_unchanged": True,
        "withdrawn_template_strategy_nodes_retained": False,
        "formal_clause_missing_container_count": 0,
        "all_formal_claims_authorized_debug_only": True,
        "all_paths_read_only": True,
        "symlink_count": 0,
        "secret_prefix_hit_count": 0,
        "secrets_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
