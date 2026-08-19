#!/usr/bin/env python3
"""Publish the audited v140 GPT-5.6 Sol output as an immutable v10 bundle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping


BUNDLE_ID = "end2end-leaf-llm-redistilled-recipe-runforest-v10"
BUNDLE_VERSION = "v10-leaf-llm-redistilled-20260819"
TASK_ID = "leaf-classification"

TACTIC_KIND_TO_RUNTIME_TAXONOMY = {
    "promotion_gate": "validation_protocol",
    "variant_control": "validation_protocol",
    "oof_integrity": "validation_protocol",
    "leakage_control": "validation_protocol",
    "ensemble_selection": "training_protocol",
    "calibration_guard": "training_protocol",
    "execution_preflight": "infrastructure",
    "shape_contract": "architecture",
    "submission_gate": "validation_protocol",
    "checkpoint_ensemble": "training_protocol",
}


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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def strict_clean_graph_node(node: Mapping[str, Any] | None) -> bool:
    if not isinstance(node, Mapping):
        return False
    audit = (
        node.get("leakage_audit")
        if isinstance(node.get("leakage_audit"), Mapping)
        else {}
    )
    metric = node.get("metric")
    return bool(
        node.get("type") == "RunNode"
        and node.get("task") == TASK_ID
        and node.get("is_buggy") is False
        and node.get("is_valid") is True
        and isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and audit.get("status") == "clean"
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
        and audit.get("rank_eligible") is True
        and node.get("quarantined") is not True
        and node.get("protocol_biased") is not True
        and len(str(node.get("code_sha256") or "")) == 64
    )


def thaw(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            raise ValueError(f"Bundle scaffold contains a symlink: {path}")
        path.chmod(0o755 if path.is_dir() else 0o644)


def freeze(root: Path) -> None:
    for path in [*root.rglob("*"), root]:
        if path.is_symlink():
            raise ValueError(f"Published bundle contains a symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)


def source_details(
    refs: Iterable[str], source_index: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], list[str]]:
    nodes: list[str] = []
    transitions: list[str] = []
    for ref in refs:
        if ref not in source_index:
            raise ValueError(f"Teacher handle is absent from source index: {ref}")
        row = source_index[ref]
        nodes.extend(map(str, row.get("source_node_ids") or []))
        transitions.extend(map(str, row.get("source_transition_ids") or []))
    return unique(nodes), unique(transitions)


def official_support(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(row["target_id"]),
        "csv_sha256": str(row["official_csv_sha256"]),
        "disposition": "official_positive",
        "kaggle_ref": str(row["official_kaggle_ref"]),
        "official_metric": float(row["official_metric"]),
        "positive_memory_eligible": True,
        "submission_key": (
            f"{row['target_id']}::{row['official_submission_variant']}"
        ),
        "validation_protocol": "official_kaggle_scored_test",
        "variant": str(row["official_submission_variant"]),
    }


def stable_clause_id(sop_id: str, claim_id: str, protocol_ref: str) -> str:
    digest = hashlib.sha256(
        canonical_bytes(
            {
                "active_protocol_ref": protocol_ref,
                "claim_id": claim_id,
                "sop_id": sop_id,
            }
        )
    ).hexdigest()[:24]
    return f"recipe-debug-clause::{digest}"


def project_recipe_implementation_capsules(
    *,
    stage: Path,
    evidence: Mapping[str, Any],
    graph_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Project copied implementation capsules to production evidence authority.

    The production loader derives its required implementation inventory from
    ``selected_evidence`` plus the two code-bearing endpoints of every selected
    repair transition. A redistilled bundle must publish exactly that
    inventory; retaining unrelated parent-bundle capsules makes the loader
    fail closed even when every actually required node is present.
    """

    path = stage / "recipe" / "implementation_capsules.json"
    payload = read_json(path)
    if payload.get("schema") != "mlevolve_recipe_implementation_capsules_v1":
        raise ValueError("Unsupported Recipe implementation capsule schema")

    expected_code_by_node: dict[str, str] = {}

    def require_node(node_id: object, code_sha256: object) -> None:
        canonical_id = str(node_id or "")
        canonical_sha = str(code_sha256 or "")
        if not canonical_id or len(canonical_sha) != 64:
            raise ValueError(
                "Selected Recipe evidence has an incomplete implementation identity"
            )
        previous = expected_code_by_node.setdefault(canonical_id, canonical_sha)
        if previous != canonical_sha:
            raise ValueError(
                f"Conflicting implementation hashes for selected node: {canonical_id}"
            )

    selected = evidence.get("selected_evidence")
    selected_repairs = evidence.get("selected_repair_evidence")
    if not isinstance(selected, Mapping) or not isinstance(
        selected_repairs, Mapping
    ):
        raise ValueError("Recipe evidence projection is malformed")
    for records in selected.values():
        if not isinstance(records, list):
            raise ValueError("Recipe positive evidence group is malformed")
        for row in records:
            if not isinstance(row, Mapping):
                raise ValueError("Recipe positive evidence row is malformed")
            require_node(row.get("node_id"), row.get("code_sha256"))

    expected_transition_endpoints: dict[str, tuple[str, str]] = {}
    for records in selected_repairs.values():
        if not isinstance(records, list):
            raise ValueError("Recipe repair evidence group is malformed")
        for row in records:
            if not isinstance(row, Mapping):
                raise ValueError("Recipe repair evidence row is malformed")
            transition_id = str(row.get("transition_id") or "")
            parent_id = str(row.get("failure_node_id") or "")
            child_id = str(row.get("successful_node_id") or "")
            if not transition_id or transition_id in expected_transition_endpoints:
                raise ValueError(
                    f"Duplicate or missing selected repair transition: {transition_id}"
                )
            require_node(parent_id, row.get("failure_node_code_sha256"))
            require_node(child_id, row.get("successful_node_code_sha256"))
            expected_transition_endpoints[transition_id] = (parent_id, child_id)

    node_rows = payload.get("nodes")
    transition_rows = payload.get("transitions")
    if not isinstance(node_rows, list) or not isinstance(transition_rows, list):
        raise ValueError("Recipe implementation capsule inventory is malformed")
    nodes_by_id: dict[str, Mapping[str, Any]] = {}
    for row in node_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Recipe implementation node is not an object")
        node_id = str(row.get("node_id") or "")
        if not node_id or node_id in nodes_by_id:
            raise ValueError(
                f"Duplicate or missing implementation node id: {node_id}"
            )
        nodes_by_id[node_id] = row
    transitions_by_id: dict[str, Mapping[str, Any]] = {}
    for row in transition_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Recipe implementation transition is not an object")
        transition_id = str(row.get("transition_id") or "")
        if not transition_id or transition_id in transitions_by_id:
            raise ValueError(
                "Duplicate or missing implementation transition id: "
                f"{transition_id}"
            )
        transitions_by_id[transition_id] = row

    required_node_ids = set(expected_code_by_node)
    required_transition_ids = set(expected_transition_endpoints)
    missing_nodes = required_node_ids - set(nodes_by_id)
    missing_transitions = required_transition_ids - set(transitions_by_id)
    if missing_nodes or missing_transitions:
        raise ValueError(
            "Implementation capsules do not cover the selected evidence: "
            f"nodes={sorted(missing_nodes)} transitions={sorted(missing_transitions)}"
        )

    projected_nodes: list[dict[str, Any]] = []
    for node_id in sorted(required_node_ids):
        row = dict(nodes_by_id[node_id])
        code = row.get("code")
        code_sha = str(row.get("code_sha256") or "")
        graph_node = graph_by_id.get(node_id)
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Implementation capsule has no code: {node_id}")
        if hashlib.sha256(code.encode("utf-8")).hexdigest() != code_sha:
            raise ValueError(f"Implementation capsule code hash mismatch: {node_id}")
        if code_sha != expected_code_by_node[node_id]:
            raise ValueError(
                f"Implementation capsule differs from selected evidence: {node_id}"
            )
        if not isinstance(graph_node, Mapping) or str(
            graph_node.get("code_sha256") or ""
        ) != code_sha:
            raise ValueError(
                f"Implementation capsule differs from RunForest: {node_id}"
            )
        if not str(row.get("source_journal") or ""):
            raise ValueError(
                f"Implementation capsule provenance is incomplete: {node_id}"
            )
        projected_nodes.append(row)

    projected_transitions: list[dict[str, Any]] = []
    for transition_id in sorted(required_transition_ids):
        row = dict(transitions_by_id[transition_id])
        expected_parent, expected_child = expected_transition_endpoints[transition_id]
        if (
            str(row.get("parent_node_id") or "") != expected_parent
            or str(row.get("child_node_id") or "") != expected_child
        ):
            raise ValueError(
                f"Implementation transition endpoints differ from evidence: {transition_id}"
            )
        projected_transitions.append(row)

    dropped_node_ids = sorted(set(nodes_by_id) - required_node_ids)
    dropped_transition_ids = sorted(
        set(transitions_by_id) - required_transition_ids
    )
    payload.update(
        {
            "coverage_policy": (
                "exact production selected_evidence nodes plus selected repair "
                "transition endpoints"
            ),
            "required_node_ids": sorted(required_node_ids),
            "required_transition_ids": sorted(required_transition_ids),
            "nodes": projected_nodes,
            "transitions": projected_transitions,
            "node_count": len(projected_nodes),
            "transition_count": len(projected_transitions),
            "unique_code_count": len(
                {str(row["code_sha256"]) for row in projected_nodes}
            ),
            "missing_node_ids": [],
            "missing_transition_ids": [],
            "complete_recipe_coverage": True,
        }
    )
    write_json(path, payload)
    report = {
        "schema": "mlevolve_recipe_implementation_projection_report_v1",
        "status": "pass",
        "policy": payload["coverage_policy"],
        "required_node_count": len(required_node_ids),
        "required_transition_count": len(required_transition_ids),
        "projected_node_count": len(projected_nodes),
        "projected_transition_count": len(projected_transitions),
        "dropped_parent_node_count": len(dropped_node_ids),
        "dropped_parent_transition_count": len(dropped_transition_ids),
        "dropped_parent_node_ids": dropped_node_ids,
        "dropped_parent_transition_ids": dropped_transition_ids,
        "implementation_capsule_file_sha256": sha256_file(path),
    }
    write_json(stage / "reports" / "recipe_implementation_projection.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    bundle_id = str(args.bundle_id)
    bundle_version = str(args.bundle_version)
    base = args.base_bundle.resolve(strict=True)
    teacher_input = args.teacher_input.resolve(strict=True)
    teacher_output = args.teacher_output.resolve(strict=True)
    root = args.bundle_root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to replace bundle root: {root}")
    root.mkdir(parents=True)
    stage = root / f".staging-{bundle_version}"
    final = root / "bundles" / bundle_version
    shutil.copytree(base, stage, copy_function=shutil.copy2)
    thaw(stage)
    (stage / "manifest.json").unlink(missing_ok=False)

    parent_manifest = read_json(base / "manifest.json")
    parent_recipe = read_json(base / "recipe" / "recipe_sops.json")
    parent_evidence = read_json(base / "recipe" / "evidence_manifest.json")
    teacher_packet = read_json(teacher_input / "teacher_packet.json")
    strategy_packet = read_json(teacher_input / "strategy_teacher_packet.json")
    teacher = read_json(teacher_output / "teacher_response_gpt56sol.json")
    teacher_report = read_json(teacher_output / "RUN_REPORT.json")
    if payload_hash(teacher_packet, "packet_sha256") != teacher_packet.get(
        "packet_sha256"
    ):
        raise ValueError("Teacher packet hash mismatch")
    if payload_hash(teacher, "teacher_response_sha256") != teacher.get(
        "teacher_response_sha256"
    ):
        raise ValueError("Teacher response hash mismatch")
    if teacher_report.get("teacher_response_sha256") != teacher.get(
        "teacher_response_sha256"
    ):
        raise ValueError("Teacher run report does not bind the response")

    source_index = teacher_packet["source_index"]
    official_by_ref = {
        row["ref"]: row for row in teacher_packet["official_candidates"]
    }
    graph = read_json(stage / "runforest" / "graph.json")
    graph_by_id = {str(row["id"]): row for row in graph["nodes"]}
    graph_node_ids = {str(row["id"]) for row in graph["nodes"]}
    graph_transition_ids = {
        str(row["id"])
        for row in graph["nodes"]
        if row.get("type") == "Transition"
    }

    parent_clauses = read_jsonl(stage / "sop" / "clauses.jsonl")
    retained_atomic_ids = {
        str(row["sop_id"])
        for row in parent_clauses
        if str(row.get("sop_id") or "").startswith(
            f"repair-claim::{TASK_ID}::"
        )
    }
    retained_atomic = [
        dict(row)
        for row in parent_recipe["nodes"]
        if str(row.get("id") or "") in retained_atomic_ids
    ]
    if len(retained_atomic) != 276 or len(retained_atomic_ids) != 276:
        raise ValueError("Expected exactly 276 atomic Repair containers")

    recipes = []
    for row in teacher["recipes"]:
        refs = list(row["evidence_refs"])
        nodes, transitions = source_details(refs, source_index)
        supports = sorted(
            (official_support(official_by_ref[ref]) for ref in refs),
            key=lambda value: (value["official_metric"], value["candidate_id"]),
        )
        best = supports[0]
        recipes.append(
            {
                "id": row["id"],
                "type": "SOP",
                "abstraction_level": "L1_recipe",
                "task_id": TASK_ID,
                "task_domain": "multimodal_multiclass_classification",
                "title": row["title"],
                "method_family": row["method_family"],
                "sop_kind": "model_strategy_recipe",
                "when_to_use": row["when_to_use"],
                "pipeline": row["pipeline"],
                "teacher_distilled_recipe": row["distilled_recipe"],
                "teacher_boundary": row["boundary"],
                "recipe_complete": True,
                "decision_stages": ["draft"],
                "validation_protocol": "official_kaggle_scored_test",
                "official_metric": best["official_metric"],
                "official_csv_sha256": best["csv_sha256"],
                "official_kaggle_ref": best["kaggle_ref"],
                "official_submission_variant": best["variant"],
                "official_support": supports,
                "source_node_ids": nodes,
                "clean_supporting_node_ids": [
                    node_id
                    for node_id in nodes
                    if strict_clean_graph_node(graph_by_id.get(node_id))
                ],
                "source_transition_ids": transitions,
                "teacher_evidence_refs": refs,
                "source_admission": "gpt56sol_llm_redistilled_official_v10",
            }
        )

    tactics = []
    for row in teacher["tactics"]:
        teacher_sop_kind = str(row["sop_kind"])
        if teacher_sop_kind not in TACTIC_KIND_TO_RUNTIME_TAXONOMY:
            raise ValueError(
                f"Teacher tactic kind lacks a runtime taxonomy mapping: {teacher_sop_kind}"
            )
        refs = list(row["evidence_refs"])
        boundary_refs = list(row["boundary_refs"])
        nodes, transitions = source_details(refs, source_index)
        boundary_nodes, boundary_transitions = source_details(
            boundary_refs, source_index
        )
        support_rows = sorted(
            (
                official_support(official_by_ref[ref])
                for ref in refs
                if ref in official_by_ref
            ),
            key=lambda value: (value["official_metric"], value["candidate_id"]),
        )
        tactic = {
            "id": row["id"],
            "type": "SOP",
            "abstraction_level": "L2_tactic",
            "task_id": TASK_ID,
            "task_domain": "multimodal_multiclass_classification",
            "title": row["title"],
            "sop_kind": TACTIC_KIND_TO_RUNTIME_TAXONOMY[teacher_sop_kind],
            "teacher_sop_kind": teacher_sop_kind,
            "host_taxonomy_projection": {
                "schema": "teacher_tactic_runtime_taxonomy_projection_v1",
                "teacher_kind": teacher_sop_kind,
                "runtime_kind": TACTIC_KIND_TO_RUNTIME_TAXONOMY[
                    teacher_sop_kind
                ],
                "teacher_content_changed": False,
            },
            "instruction": row["instruction"],
            "when_to_use": row["when_to_use"],
            "parent_method_families": row["parent_method_families"],
            "teacher_boundary": row["boundary"],
            "decision_stages": ["draft", "improve", "model_design"],
            "source_node_ids": nodes,
            "clean_supporting_node_ids": [
                node_id
                for node_id in nodes
                if strict_clean_graph_node(graph_by_id.get(node_id))
            ],
            "source_transition_ids": transitions,
            "boundary_source_node_ids": boundary_nodes,
            "boundary_source_transition_ids": boundary_transitions,
            "teacher_evidence_refs": refs,
            "teacher_boundary_refs": boundary_refs,
            "source_admission": "gpt56sol_llm_redistilled_tactic_v10",
        }
        if support_rows:
            best = support_rows[0]
            tactic.update(
                {
                    "validation_protocol": "official_kaggle_scored_test",
                    "official_metric": best["official_metric"],
                    "official_csv_sha256": best["csv_sha256"],
                    "official_kaggle_ref": best["kaggle_ref"],
                    "official_submission_variant": best["variant"],
                    "official_support": support_rows,
                }
            )
        tactics.append(tactic)

    atomic_claims = read_json(stage / "recipe" / "atomic_claims.json")
    claims_by_transition: dict[str, list[str]] = defaultdict(list)
    claim_by_id = {}
    for claim in atomic_claims["claims"]:
        claim_by_id[str(claim["id"])] = claim
        if claim.get("claim_status") == "authorized_debug_only":
            claims_by_transition[str(claim.get("source_transition_id") or "")].append(
                str(claim["id"])
            )
    clauses_by_sop = {
        str(row["sop_id"]): row
        for row in parent_clauses
        if str(row.get("sop_id") or "") in retained_atomic_ids
    }
    if set(clauses_by_sop) != retained_atomic_ids:
        raise ValueError("Retained atomic containers do not map one-to-one to clauses")

    repairs = []
    for row in teacher["repairs"]:
        refs = list(row["evidence_refs"])
        nodes, transitions = source_details(refs, source_index)
        direct_claim_ids = unique(
            claim_id
            for transition in transitions
            for claim_id in claims_by_transition.get(transition, [])
        )
        source_atomic_container_ids = unique(
            source_index[ref].get("canonical_id")
            for ref in refs
            if str(source_index[ref].get("canonical_id") or "")
            in retained_atomic_ids
        )
        container_claim_ids = unique(
            claim_id
            for container_id in source_atomic_container_ids
            for claim_id in [
                *(clauses_by_sop[container_id].get("claim_refs") or []),
                *(clauses_by_sop[container_id].get(
                    "additional_source_claim_refs"
                ) or []),
            ]
            if str(claim_id) in claim_by_id
            and claim_by_id[str(claim_id)].get("claim_status")
            == "authorized_debug_only"
        )
        claim_ids = unique([*direct_claim_ids, *container_claim_ids])
        if not claim_ids:
            raise ValueError(f"LLM Repair lacks an authorized atomic claim: {row['id']}")
        run_ids = unique(
            claim_by_id[claim_id].get("source_run_id") for claim_id in claim_ids
        )
        failure_nodes = unique(
            claim_by_id[claim_id].get("source_parent_node_id")
            for claim_id in claim_ids
        )
        successful_nodes = unique(
            claim_by_id[claim_id].get("source_child_node_id")
            for claim_id in claim_ids
        )
        repairs.append(
            {
                "id": row["id"],
                "type": "SOP",
                "abstraction_level": "L3_repair",
                "task_id": TASK_ID,
                "task_domain": "multimodal_multiclass_classification",
                "task_family": "structured_leaf_multimodal_classification",
                "task_type": "multiclass_classification",
                "title": row["title"],
                "method_family": row["method_family"],
                "sop_kind": "debug_fix",
                "when_to_use": row["when_to_use"],
                "failure_signature": {
                    "id": f"{row['id']}::failure-signature",
                    "pattern": row["failure_signature"],
                    "root_cause": row["failure_signature"],
                },
                "repair_action": {
                    "summary": row["repair_action"],
                    "steps": [row["repair_action"]],
                },
                "teacher_failure_signature_summary": row["failure_signature"],
                "teacher_repair_action_summary": row["repair_action"],
                "decision_stages": ["debug"],
                "runtime_stage": "debug",
                "runtime_stages": ["debug"],
                "evidence_status": "gpt56sol_consolidated_observed_debug_fixed",
                "source_node_ids": nodes,
                "failure_node_ids": failure_nodes,
                "successful_node_ids": successful_nodes,
                "source_transition_ids": transitions,
                "supporting_transition_ids": transitions,
                "source_atomic_container_ids": source_atomic_container_ids,
                "source_claim_ids": claim_ids,
                "distinct_run_ids": run_ids,
                "distinct_run_count": len(run_ids),
                "successful_repair_count": len(transitions),
                "confidence_prior": min(0.95, 0.68 + 0.03 * len(transitions)),
                "teacher_evidence_refs": refs,
                "source_admission": "gpt56sol_llm_redistilled_repair_v10",
                "infrastructure_failure": False,
            }
        )

    new_nodes = sorted(
        [*retained_atomic, *recipes, *tactics, *repairs], key=lambda row: row["id"]
    )
    if len({row["id"] for row in new_nodes}) != len(new_nodes):
        raise ValueError("Published Recipe IDs are not unique")
    recipe_payload = {
        "schema": "mlevolve_recipe_sop_bundle_v1",
        "bundle_version": bundle_version,
        "created_at": args.created_at,
        "evidence_manifest_sha256": "",
        "official_ledger_sha256": parent_manifest["official_ledger_sha256"],
        "teacher_response_sha256": teacher["teacher_response_sha256"],
        "teacher": teacher["teacher"],
        "routing_contract": parent_recipe.get("routing_contract") or {},
        "llm_redistillation": {
            "model": "gpt-5.6-sol",
            "base_url": "https://apizh.net/v1",
            "temperature": 0.0,
            "parent_packet_sha256": teacher_packet["packet_sha256"],
            "strategy_projection_sha256": strategy_packet["projection_sha256"],
            "teacher_response_sha256": teacher["teacher_response_sha256"],
            "retained_atomic_container_count": len(retained_atomic),
            "withdrawn_template_strategy_nodes_retained": False,
        },
        "nodes": new_nodes,
        "bundle_sha256": "",
    }

    evidence = dict(parent_evidence)
    selected_evidence = [
        dict(row)
        for row in (
            parent_evidence.get("selected_evidence", {}).get(TASK_ID, [])
        )
        if row.get("audit_status") == "clean"
        and row.get("memory_disposition") == "positive_eligible"
        and row.get("paper_grade_eligible") is True
        and row.get("rank_eligible") is True
        and len(str(row.get("code_sha256") or "")) == 64
        and strict_clean_graph_node(graph_by_id.get(str(row.get("node_id") or "")))
        and graph_by_id[str(row["node_id"])].get("metric") == row.get("metric")
        and graph_by_id[str(row["node_id"])].get("code_sha256")
        == row.get("code_sha256")
    ]
    selected_repair_evidence = [
        dict(row)
        for row in (
            parent_evidence.get("selected_repair_evidence", {}).get(
                TASK_ID, []
            )
        )
        if row.get("audit_status") == "clean"
        and row.get("memory_disposition") == "positive_eligible"
        and row.get("paper_grade_eligible") is True
        and row.get("rank_eligible") is True
        and row.get("task_id") == TASK_ID
        and "debug" in str(row.get("stage_pair") or "")
        and str(row.get("failure_node_id") or "")
        and str(row.get("successful_node_id") or "")
        and str(row.get("failure_text") or "")
        and str(row.get("repair_action_text") or "")
        and isinstance(row.get("successful_metric"), (int, float))
        and not isinstance(row.get("successful_metric"), bool)
    ]
    if len(selected_evidence) != 4 or len(selected_repair_evidence) != 6:
        raise ValueError(
            "Expected the strict-clean v8 evidence projection (4 positive, 6 repair)"
        )
    evidence.update(
        {
            "created_at": args.created_at,
            "evidence_version": bundle_version,
            "distillation_teacher_response_sha256": teacher[
                "teacher_response_sha256"
            ],
            "teacher_packet_sha256": teacher_packet["packet_sha256"],
            "strategy_projection_sha256": strategy_packet["projection_sha256"],
            "llm_distilled_counts": {"recipes": 8, "tactics": 10, "repairs": 12},
            "selected_evidence": {TASK_ID: selected_evidence},
            "selected_counts_by_task": {TASK_ID: len(selected_evidence)},
            "selected_repair_evidence": {
                TASK_ID: selected_repair_evidence
            },
            "selected_repair_counts_by_task": {
                TASK_ID: len(selected_repair_evidence)
            },
            "llm_redistillation_evidence_projection": {
                "policy": (
                    "Retain only records that already satisfy the production "
                    "strict-clean Recipe evidence schema and exactly match the "
                    "immutable RunForest node. New official outcomes remain in "
                    "the official ledger/support fields; new repair transitions "
                    "remain in RunForest and atomic claim authority."
                ),
                "positive_records_retained": len(selected_evidence),
                "repair_records_retained": len(selected_repair_evidence),
                "incomplete_mechanical_records_retained": 0,
            },
            "manifest_sha256": "",
        }
    )
    evidence["manifest_sha256"] = payload_hash(evidence, "manifest_sha256")
    recipe_payload["evidence_manifest_sha256"] = evidence["manifest_sha256"]
    recipe_payload["bundle_sha256"] = payload_hash(
        recipe_payload, "bundle_sha256"
    )
    write_json(stage / "recipe" / "recipe_sops.json", recipe_payload)
    write_json(stage / "recipe" / "evidence_manifest.json", evidence)
    write_json(stage / "recipe" / "teacher_packet.json", teacher_packet)
    write_json(stage / "recipe" / "teacher_response_gpt56sol.json", teacher)
    write_json(stage / "recipe" / "teacher_response_gpt56sol.raw.json", teacher)

    implementation_projection = project_recipe_implementation_capsules(
        stage=stage,
        evidence=evidence,
        graph_by_id=graph_by_id,
    )
    if (
        implementation_projection["required_node_count"] != 15
        or implementation_projection["required_transition_count"] != 6
        or implementation_projection["projected_node_count"] != 15
        or implementation_projection["projected_transition_count"] != 6
    ):
        raise ValueError(
            "Unexpected v10 production implementation capsule projection"
        )

    teacher_evidence = stage / "recipe" / "teacher_evidence"
    teacher_evidence.mkdir(exist_ok=True)
    for path in sorted(teacher_input.glob("*.json")):
        shutil.copy2(path, teacher_evidence / f"input__{path.name}")
    for path in sorted(teacher_output.glob("*.json")):
        shutil.copy2(path, teacher_evidence / f"output__{path.name}")

    retained_ids = {row["id"] for row in retained_atomic}
    old_clauses = [
        row for row in parent_clauses if row["sop_id"] in retained_ids
    ]
    if len(old_clauses) != 276:
        raise ValueError("Atomic formal clauses are not exactly preserved")
    protocol_ref = str(parent_manifest["active_protocol_ref"])
    new_clauses = []
    for row in repairs:
        primary_claim = row["source_claim_ids"][0]
        clause_id = stable_clause_id(row["id"], primary_claim, protocol_ref)
        failure = row["teacher_failure_signature_summary"]
        repair = row["teacher_repair_action_summary"]
        text = (
            f"{row['title']}\nFailure: {failure}\nRepair: {repair}\n"
            f"Use when: {row['when_to_use']}"
        )
        new_clauses.append(
            {
                "schema": "sop_clause_v1",
                "clause_id": clause_id,
                "sop_id": row["id"],
                "text": text,
                "retrieval_text": text,
                "claim_refs": [primary_claim],
                "claim_types": ["debug_repair"],
                "source_artifact_refs": [row["source_node_ids"][0]],
                "source_transition_refs": row["source_transition_ids"],
                "source_run_ids": row["distinct_run_ids"],
                "source_task_ids": [TASK_ID],
                "source_task_families": [row["task_family"]],
                "source_domains": ["multimodal"],
                "transfer_scope": "",
                "protocol_scope": [protocol_ref],
                "task_scope": {"task_ids": [TASK_ID]},
                "permitted_operations": ["debug_hypothesis"],
                "permitted_generation_stages": ["debug"],
                "permitted_governance_stages": ["retrieval"],
                "publication_class": "diagnostic",
                "authority_decision_refs": [],
                "receipt_refs": [],
                "derivation_refs": unique(
                    [
                        *row["source_claim_ids"],
                        *row["source_transition_ids"],
                        *row["source_atomic_container_ids"],
                    ]
                ),
                "additional_source_claim_refs": row["source_claim_ids"][1:],
                "applies_when": [row["when_to_use"]],
                "prevents": [failure],
                "protocol_agnostic": False,
                "legacy_status": "native_recipe_l3_debug_v1",
                "publication_origin": "gpt56sol_llm_redistilled_v10",
            }
        )
    clauses = sorted(
        [*old_clauses, *new_clauses],
        key=lambda row: (row["sop_id"], row["clause_id"]),
    )
    write_jsonl(stage / "sop" / "clauses.jsonl", clauses)
    mask_key = "|".join([protocol_ref, "debug_hypothesis", "debug", "retrieval"])
    masks = {
        "schema": "declared_scope_visibility_masks_v1",
        "semantics": (
            "Declared-scope prefilter only; runtime Authority evaluation and "
            "the task-local atomic repair hard gate still apply."
        ),
        "active_protocol_refs": [protocol_ref],
        "masks": {mask_key: sorted(row["clause_id"] for row in clauses)},
    }
    write_json(
        stage / "visibility" / "precompiled_masks" / "declared_scope_masks.json",
        masks,
    )
    visibility_report = {
        "schema": "recipe_visibility_publication_v1",
        "source_recipe_schema": recipe_payload["schema"],
        "source_recipe_bundle_sha256": recipe_payload["bundle_sha256"],
        "active_protocol_ref": protocol_ref,
        "recipe_node_count": len(new_nodes),
        "l3_recipe_count": len(retained_atomic) + len(repairs),
        "published_clause_count": len(clauses),
        "published_sop_count": len({row["sop_id"] for row in clauses}),
        "retained_atomic_clause_count": len(old_clauses),
        "llm_redistilled_clause_count": len(new_clauses),
        "skipped_count": 0,
        "mask_keys": [mask_key],
        "cross_task_transfer_enabled": False,
        "score_or_metric_claims_published": False,
        "replay_claims_published": False,
    }
    write_json(stage / "reports" / "recipe_visibility_publication.json", visibility_report)

    release = {
        "schema": "mlevolve_leaf_llm_redistilled_memory_release_v1",
        "bundle_version": bundle_version,
        "created_at": args.created_at,
        "teacher_model": "gpt-5.6-sol",
        "teacher_base_url": "https://apizh.net/v1",
        "temperature": 0.0,
        "parent_packet_sha256": teacher_packet["packet_sha256"],
        "teacher_response_sha256": teacher["teacher_response_sha256"],
        "recipe_sop_bundle_sha256": recipe_payload["bundle_sha256"],
        "recipe_evidence_manifest_sha256": evidence["manifest_sha256"],
        "atomic_claim_bundle_sha256": atomic_claims["bundle_sha256"],
        "counts": {
            "retained_atomic_repair_containers": len(retained_atomic),
            "llm_recipes": len(recipes),
            "llm_tactics": len(tactics),
            "llm_repairs": len(repairs),
            "formal_clauses": len(clauses),
            "implementation_capsule_nodes": implementation_projection[
                "projected_node_count"
            ],
            "implementation_capsule_transitions": implementation_projection[
                "projected_transition_count"
            ],
        },
        "withdrawn_template_strategy_nodes_retained": False,
        "secrets_recorded": False,
    }
    write_json(stage / "recipe" / "release_report.json", release)

    recipe_ids = {row["id"] for row in new_nodes}
    graph_sop_ids = {
        row["id"] for row in graph["nodes"] if row.get("type") == "SOP"
    }
    missing_containers = sorted(
        {row["sop_id"] for row in clauses} - recipe_ids - graph_sop_ids
    )
    if missing_containers:
        raise ValueError(f"Formal clauses lack containers: {missing_containers[:5]}")
    if not all(
        set(row.get("source_node_ids") or []).issubset(graph_node_ids)
        for row in [*recipes, *tactics, *repairs]
    ):
        raise ValueError("LLM SOP references a node absent from RunForest")
    if not all(
        set(row.get("source_transition_ids") or []).issubset(graph_transition_ids)
        for row in [*recipes, *tactics, *repairs]
    ):
        raise ValueError("LLM SOP references a transition absent from RunForest")

    artifact_hashes = {
        str(path.relative_to(stage)): sha256_file(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "bundle_id": bundle_id,
            "bundle_version": bundle_version,
            "created_at": args.created_at,
            "parent_bundle": parent_manifest["bundle_version"],
            "certification_level": "gpt56sol_llm_redistilled_audited",
            "authority_policy_version": "claim_level_debug_operation_visibility_v2",
            "build_report": "recipe/release_report.json",
            "visibility_publication_report": "reports/recipe_visibility_publication.json",
            "recipe_sop_bundle_sha256": recipe_payload["bundle_sha256"],
            "recipe_evidence_manifest_sha256": evidence["manifest_sha256"],
            "atomic_claim_bundle_sha256": atomic_claims["bundle_sha256"],
            "atomic_debug_authorized_count": int(
                atomic_claims["claim_status_counts"]["authorized_debug_only"]
            ),
            "formal_debug_clause_count": len(clauses),
            "artifact_hashes": artifact_hashes,
            "graph_hashes": {
                "runforest": artifact_hashes["runforest/graph.json"]
            },
            "index_hashes": {
                "runforest": artifact_hashes["runforest/index.npz"]
            },
            "manifest_sha256": "",
        }
    )
    manifest["manifest_sha256"] = payload_hash(manifest, "manifest_sha256")
    write_json(stage / "manifest.json", manifest)

    for relative, expected in manifest["artifact_hashes"].items():
        if sha256_file(stage / relative) != expected:
            raise ValueError(f"Artifact changed during publication: {relative}")
    if payload_hash(manifest, "manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Published manifest canonical hash mismatch")

    final.parent.mkdir(parents=True)
    stage.rename(final)
    pointer = {
        "schema": "memory_bundle_current_v1",
        "bundle_id": bundle_id,
        "bundle_version": bundle_version,
        "bundle_path": f"bundles/{bundle_version}",
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_bundle": manifest["parent_bundle"],
        "published_at": args.created_at,
        "pointer_sha256": "",
    }
    pointer["pointer_sha256"] = payload_hash(pointer, "pointer_sha256")
    write_json(root / "CURRENT.json", pointer)
    receipt = {
        "schema": "mlevolve_leaf_llm_redistilled_memory_publication_receipt_v1",
        "bundle_root": str(root),
        "bundle_path": str(final),
        "bundle_id": bundle_id,
        "bundle_version": bundle_version,
        "parent_bundle": manifest["parent_bundle"],
        "bundle_manifest_sha256": manifest["manifest_sha256"],
        "bundle_manifest_file_sha256": sha256_file(final / "manifest.json"),
        "current_file_sha256": sha256_file(root / "CURRENT.json"),
        "teacher_response_sha256": teacher["teacher_response_sha256"],
        "teacher_response_file_sha256": sha256_file(
            final / "recipe" / "teacher_response_gpt56sol.json"
        ),
        "recipe_sop_bundle_sha256": recipe_payload["bundle_sha256"],
        "recipe_sop_file_sha256": sha256_file(
            final / "recipe" / "recipe_sops.json"
        ),
        "formal_clause_file_sha256": sha256_file(
            final / "sop" / "clauses.jsonl"
        ),
        "counts": release["counts"],
        "all_opaque_refs_restored": True,
        "formal_clause_missing_container_count": 0,
        "secrets_recorded": False,
    }
    write_json(root / "PUBLICATION_RECEIPT.json", receipt)
    freeze(root)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--teacher-input", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--bundle-id", default=BUNDLE_ID)
    parser.add_argument("--bundle-version", default=BUNDLE_VERSION)
    args = parser.parse_args()
    receipt = build(args)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
