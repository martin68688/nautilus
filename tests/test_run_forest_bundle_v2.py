from __future__ import annotations

import json

import numpy as np

from tests.memory_bundle_helpers import (
    REPO,
    prepare_audit_and_splits,
    prepare_corpus,
    prepare_runforest_and_bundle,
    prepare_sops,
)


def test_runforest_v2_uses_manifest_sidecars_clauses_and_split_only(tmp_path) -> None:
    from schema import sha256_file

    legacy_graph = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
    legacy_hash = sha256_file(legacy_graph)
    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    sops = prepare_sops(tmp_path, corpus, audit_splits)
    result = prepare_runforest_and_bundle(
        tmp_path, corpus, audit_splits, sops, split_name="task-heldout"
    )
    report = result["runforest_report"]

    assert report["source_run_count"] == 9
    assert report["heldout_run_count"] == 3
    assert report["spooky_source_run_count"] == 0
    assert report["all_code_nodes_have_sidecars"] is True
    assert report["all_clause_sources_resolve"] is True
    assert report["heldout_run_refs_in_graph"] == []
    assert report["authorized_edge_count"] == 0
    assert report["navigation_edge_count"] == 9
    assert report["edge_kind_counts"].get("distills_to", 0) == 0
    assert report["included_clause_count"] == 27
    assert len(report["excluded_clauses"]) == 9
    assert sha256_file(legacy_graph) == legacy_hash

    graph = json.loads(
        (result["runforest_dir"] / "graph.json").read_text(encoding="utf-8")
    )
    clause_ids = {
        node["clause_id"] for node in graph["nodes"] if node["type"] == "SOPClause"
    }
    clauses = [
        node for node in graph["nodes"] if node["type"] == "SOPClause"
    ]
    index = np.load(
        result["runforest_dir"] / "clause_index.npz", allow_pickle=True
    )
    assert set(index["clause_ids"].tolist()) == clause_ids
    assert index["embeddings"].shape == (27, 64)

    mask_payload = json.loads(
        (
            result["runforest_dir"]
            / "visibility"
            / "precompiled_masks"
            / "declared_scope_masks.json"
        ).read_text(encoding="utf-8")
    )
    masks = mask_payload["masks"]
    assert all(len(key.split("|")) == 4 for key in masks)
    for clause in clauses:
        for protocol in clause["protocol_scope"]:
            for operation in clause["permitted_operations"]:
                for generation_stage in clause["permitted_generation_stages"]:
                    for governance_stage in clause["permitted_governance_stages"]:
                        key = "|".join(
                            [
                                protocol,
                                operation,
                                generation_stage,
                                governance_stage,
                            ]
                        )
                        assert clause["clause_id"] in masks[key]
                        assert governance_stage != generation_stage
                        duplicated_generation_key = "|".join(
                            [
                                protocol,
                                operation,
                                generation_stage,
                                generation_stage,
                            ]
                        )
                        assert duplicated_generation_key not in masks


def test_seed_heldout_keeps_source_seed_clauses_despite_shared_tasks(tmp_path) -> None:
    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    sops = prepare_sops(tmp_path, corpus, audit_splits)
    result = prepare_runforest_and_bundle(
        tmp_path,
        corpus,
        audit_splits,
        sops,
        split_name="seed-heldout",
    )
    report = result["runforest_report"]

    assert report["source_run_count"] == 8
    assert report["heldout_run_count"] == 4
    assert report["included_clause_count"] == 24
    assert report["excluded_clause_reason_counts"] == {
        "source_ref_outside_split": 12
    }
    assert report["task_scope_exclusion_enabled"] is False
    assert not any(
        row["reason"] == "heldout_task_scope"
        for row in report["excluded_clauses"]
    )
    assert result["bundle_result"]["validation"]["valid"] is True


def test_same_domain_task_heldout_propagates_auditable_lineage(tmp_path) -> None:
    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    sops = prepare_sops(tmp_path, corpus, audit_splits)
    result = prepare_runforest_and_bundle(
        tmp_path,
        corpus,
        audit_splits,
        sops,
        split_name="same-domain-task-heldout",
    )
    report = result["runforest_report"]

    assert report["source_run_count"] == 3
    assert report["heldout_run_count"] == 3
    assert report["source_task_ids"] == ["task-b"]
    assert report["heldout_task_ids"] == ["task-a"]
    assert report["source_task_families"] == ["family-a"]
    assert report["source_domains"] == ["family_a"]
    assert report["target_task_id"] == "task-a"
    assert report["target_domain"] == "family_a"
    assert report["same_domain_task_heldout"] is True
    assert report["all_included_clauses_have_domain_lineage"] is True
    assert report["cross_domain_included_clause_ids"] == []
    assert report["included_clause_count"] == 9
    assert result["bundle_result"]["validation"]["valid"] is True

    graph = json.loads(
        (result["runforest_dir"] / "graph.json").read_text(encoding="utf-8")
    )
    assert graph["meta"]["domain_scope_required"] is True
    assert graph["meta"]["transfer_design"] == (
        "same_domain_different_task_task_heldout"
    )
    clauses = [
        node for node in graph["nodes"] if node["type"] == "SOPClause"
    ]
    containers = [node for node in graph["nodes"] if node["type"] == "SOP"]
    assert clauses and containers
    assert all(clause["source_task_ids"] == ["task-b"] for clause in clauses)
    assert all(clause["source_domains"] == ["family_a"] for clause in clauses)
    assert all(clause["transfer_scope"] == "same_domain" for clause in clauses)
    assert all(container["domain_scope_complete"] is True for container in containers)
    assert all(container["source_domains"] == ["family_a"] for container in containers)
