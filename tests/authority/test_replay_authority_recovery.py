from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from authority.authority_engine import AuthorityEngine
from authority.certified_bundle import CertifiedBundlePublisher
from authority.bundle_authority import load_snapshot_authority
from authority.clean_replay import ReplayAuthorityRecovery
from authority.collectors import TrustedCollectorHost
from authority.evidence_graph import EvidenceGraph
from authority.memory_snapshot import (
    MemorySnapshotLoader,
    SessionOverlay,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from authority.models import (
    AuthorityRequest,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    GenerationStage,
    GovernanceStage,
    Operation,
    TaskContext,
    SOPClauseV1,
)
from authority.replay_clause_publication import ReplayClausePublication
from authority.replay_certifier import ProtocolRepairSurface, verify_protocol_only_patch
from authority.protocol_registry import ProtocolRegistry
from tests.authority.clean_replay_helpers import (
    PROTOCOL_REPAIR_CODE,
    SOURCE_CODE,
    build_registry,
    historical_score_claim,
    trusted_replay_receipts,
)
from tests.test_memory_snapshot_overlay import build_tiny_bundle, write_current


def _request(claim, ref, operation):
    return AuthorityRequest(
        artifact_id=claim.subject_artifact_id,
        claim_id=claim.claim_id,
        operation=operation,
        decision_stage=(
            DecisionStage.MEMORY_WRITEBACK
            if operation == Operation.PROMOTE
            else DecisionStage.BRANCH_SELECTION
        ),
        active_protocol=ref,
        task_context=TaskContext(task_id="task-a", task_family="tabular"),
        requesting_component="tests.clean_replay",
        generation_stage=GenerationStage.IMPROVE,
        governance_stage=(
            GovernanceStage.MEMORY_WRITEBACK
            if operation == Operation.PROMOTE
            else GovernanceStage.BRANCH_SELECTION
        ),
    )


def _claim_row(claim):
    return {
        "claim_id": claim.claim_id,
        "claim_type": claim.claim_type.value,
        "subject_artifact_id": claim.subject_artifact_id,
        "task_scope": claim.task_scope,
        "method_fingerprint": claim.method_fingerprint,
        "protocol_ref": {
            "protocol_id": claim.protocol_ref.protocol_id,
            "version": claim.protocol_ref.version,
            "canonical_hash": claim.protocol_ref.canonical_hash,
        },
        "statement": claim.statement,
        "parent_claims": claim.parent_claims,
        "source_artifact_refs": claim.source_artifact_refs,
        "evidence_refs": claim.evidence_refs,
        "boundary": claim.boundary,
        "legacy_status": claim.legacy_status,
    }


def _prepare_tiny_replay_clause_bundle(parent, manifest, protocol_key) -> None:
    clause_rows = [
        json.loads(line)
        for line in (parent / "sop" / "clauses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    containers = [
        {
            "sop_id": row["sop_id"],
            "title": row["text"],
            "task_id": (row.get("task_scope") or {}).get("task_ids", ["task-a"])[0],
            "clause_ids": [row["clause_id"]],
            "source_run_ids": ["run-a"],
            "source_task_ids": ["task-a"],
            "source_task_families": ["tabular"],
            "source_domains": ["tabular"],
            "transfer_scopes": ["same_domain"],
            "domain_scope_complete": True,
        }
        for row in clause_rows
    ]
    write_json_atomic(
        parent / "sop" / "containers.json",
        {"schema": "bundle_sop_containers_v1", "containers": containers},
    )
    write_json_atomic(
        parent / "sop" / "graph.json",
        {
            "schema": "bundle_sop_graph_v1",
            "containers": sorted(row["sop_id"] for row in containers),
            "clauses": sorted(row["clause_id"] for row in clause_rows),
            "edges": [
                {
                    "src": row["sop_id"],
                    "dst": row["clause_id"],
                    "kind": "contains_clause",
                }
                for row in clause_rows
            ],
        },
    )
    nodes = [
        {
            "id": "historical-artifact",
            "type": "RunNode",
            "run_id": "run-a",
            "task": "task-a",
        },
        *[
            {
                "id": row["sop_id"],
                "sop_id": row["sop_id"],
                "type": "SOP",
                "title": row["title"],
                "task": "task-a",
                "clause_ids": row["clause_ids"],
                "source_run_ids": row["source_run_ids"],
                "source_task_ids": row["source_task_ids"],
                "source_task_families": row["source_task_families"],
                "source_domains": row["source_domains"],
                "transfer_scopes": row["transfer_scopes"],
                "domain_scope_complete": True,
            }
            for row in containers
        ],
        *[
            {**row, "id": row["clause_id"], "type": "SOPClause"}
            for row in clause_rows
        ],
    ]
    edges = [
        {
            "src": row["sop_id"],
            "dst": row["clause_id"],
            "kind": "contains_clause",
        }
        for row in clause_rows
    ]
    write_json_atomic(
        parent / "runforest" / "graph.json",
        {"meta": {"bundle_id": manifest["bundle_id"]}, "nodes": nodes, "edges": edges},
    )
    node_ids = np.asarray([row["id"] for row in nodes], dtype=object)
    node_types = np.asarray([row["type"] for row in nodes], dtype=object)
    np.savez_compressed(
        parent / "runforest" / "index.npz",
        node_ids=node_ids,
        node_types=node_types,
        poincare=np.zeros((len(nodes), 2), dtype=np.float32),
        flat_twin=np.zeros((len(nodes), 2), dtype=np.float32),
        euclidean=np.zeros((len(nodes), 64), dtype=np.float32),
    )
    np.savez_compressed(
        parent / "runforest" / "clause_index.npz",
        clause_ids=np.asarray([row["clause_id"] for row in clause_rows], dtype=object),
        embeddings=np.zeros((len(clause_rows), 64), dtype=np.float32),
    )
    (parent / "visibility" / "clause_metadata.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in clause_rows),
        encoding="utf-8",
    )
    (parent / "authority").mkdir(exist_ok=True)
    (parent / "authority" / "derivations.jsonl").write_text("", encoding="utf-8")
    write_json_atomic(
        parent / "runforest" / "build_report.json",
        {
            "schema": "run_forest_builder_report_v2",
            "bundle_id": manifest["bundle_id"],
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_type_counts": {
                "RunNode": 1,
                "SOP": len(containers),
                "SOPClause": len(clause_rows),
            },
            "edge_kind_counts": {"contains_clause": len(edges)},
            "included_clause_count": len(clause_rows),
            "graph_sha256": sha256_file(parent / "runforest" / "graph.json"),
            "index_sha256": sha256_file(parent / "runforest" / "index.npz"),
            "clause_index_sha256": sha256_file(
                parent / "runforest" / "clause_index.npz"
            ),
            "declared_scope_masks_sha256": sha256_file(
                parent
                / "visibility"
                / "precompiled_masks"
                / "declared_scope_masks.json"
            ),
        },
    )
    write_json_atomic(
        parent / "reports" / "build_report.json",
        {
            "schema": "memory_bundle_build_report_v1",
            "bundle_id": manifest["bundle_id"],
            "bundle_version": manifest["bundle_version"],
            "clause_count": len(clause_rows),
            "container_count": len(containers),
        },
    )
    manifest["build_report"] = "reports/build_report.json"
    for relative in (
        "sop/containers.json",
        "sop/graph.json",
        "runforest/graph.json",
        "runforest/index.npz",
        "runforest/clause_index.npz",
        "runforest/build_report.json",
        "visibility/clause_metadata.jsonl",
        "authority/derivations.jsonl",
        "reports/build_report.json",
    ):
        manifest["artifact_hashes"][relative] = sha256_file(parent / relative)
    manifest["graph_hashes"] = {
        "runforest": manifest["artifact_hashes"]["runforest/graph.json"]
    }
    manifest["index_hashes"] = {
        "runforest": manifest["artifact_hashes"]["runforest/index.npz"],
        "clauses": manifest["artifact_hashes"]["runforest/clause_index.npz"],
    }
    manifest["lineage_hash"] = sha256_json(
        {
            "clauses": manifest["artifact_hashes"]["sop/clauses.jsonl"],
            "containers": manifest["artifact_hashes"]["sop/containers.json"],
            "derivations": manifest["artifact_hashes"]["authority/derivations.jsonl"],
            "sop_graph": manifest["artifact_hashes"]["sop/graph.json"],
        }
    )
    manifest["protocol_registry_hash"] = sha256_json({"active": protocol_key})


def test_clean_replay_recovers_only_new_claim_and_publishes_scoped_bundle(tmp_path) -> None:
    registry, ref = build_registry()
    verification = verify_protocol_only_patch(
        SOURCE_CODE,
        PROTOCOL_REPAIR_CODE,
        ProtocolRepairSurface.from_protocol_spec(registry.resolve(ref)),
        source_artifact_id="historical-artifact",
        replay_artifact_id="clean-replay-artifact",
    )
    graph = EvidenceGraph()
    original = historical_score_claim(
        "claim::historical",
        "historical-artifact",
        ref,
        verification.source_method_fingerprint,
    )
    unreplayed = historical_score_claim(
        "claim::unreplayed",
        "unreplayed-artifact",
        ref,
        "9" * 64,
    )
    graph.add_claim(original)
    graph.add_claim(unreplayed)
    original_before = copy.deepcopy(original)
    receipts = trusted_replay_receipts(
        TrustedCollectorHost("recovery-host"),
        artifact_id="clean-replay-artifact",
        protocol_ref=ref,
        method_fingerprint=verification.replay_method_fingerprint,
        code_sha256=verification.replay_code_sha256,
    )
    registration = ReplayAuthorityRecovery(graph, registry).register(
        original_claim_id=original.claim_id,
        verification=verification,
        receipts=receipts,
        protocol_ref=ref,
        statement="Use the preserved TF-IDF and logistic-regression method.",
        claim_type=ClaimType.METHOD_HYPOTHESIS,
    )
    replay_claim = graph.claims[registration.replay_claim_id]
    engine = AuthorityEngine(registry, graph=graph)

    assert (
        engine.authorize(
            _request(replay_claim, ref, Operation.GENERATE_CANDIDATE)
        ).outcome
        == DecisionOutcome.ALLOW
    )
    assert (
        engine.authorize(_request(replay_claim, ref, Operation.RANK)).outcome
        != DecisionOutcome.ALLOW
    )
    assert (
        engine.authorize(_request(replay_claim, ref, Operation.PROMOTE)).outcome
        != DecisionOutcome.ALLOW
    )
    for historical in (original, unreplayed):
        assert engine.authorize(_request(historical, ref, Operation.RANK)).outcome != DecisionOutcome.ALLOW
        assert engine.authorize(_request(historical, ref, Operation.PROMOTE)).outcome != DecisionOutcome.ALLOW
    assert graph.claims[original.claim_id] == original_before
    assert graph.claim_paths[original.claim_id] == []

    parent, parent_manifest = build_tiny_bundle(tmp_path)
    _prepare_tiny_replay_clause_bundle(parent, parent_manifest, ref.key())
    authority_dir = parent / "authority"
    authority_dir.mkdir(exist_ok=True)
    claims_path = authority_dir / "claims.jsonl"
    claims_path.write_text(
        "".join(
            json.dumps(_claim_row(claim), sort_keys=True) + "\n"
            for claim in (original, unreplayed)
        ),
        encoding="utf-8",
    )
    write_json_atomic(
        parent / "corpus" / "manifest.json",
        {"schema": "synthetic_nested_manifest_v1", "bundle": "parent"},
    )
    parent_manifest["artifact_hashes"]["authority/claims.jsonl"] = sha256_file(
        claims_path
    )
    parent_manifest["artifact_hashes"]["corpus/manifest.json"] = sha256_file(
        parent / "corpus" / "manifest.json"
    )
    parent_manifest["manifest_sha256"] = sha256_json(
        {
            key: value
            for key, value in parent_manifest.items()
            if key != "manifest_sha256"
        }
    )
    write_json_atomic(parent / "manifest.json", parent_manifest)
    write_current(tmp_path, parent, parent_manifest)
    parent_tree_hash = {
        str(path.relative_to(parent)): sha256_file(path)
        for path in sorted(parent.rglob("*"))
        if path.is_file()
    }
    overlay = SessionOverlay(tmp_path / "session-overlay")
    replay_clause = SOPClauseV1(
        clause_id="clause::clean-replay",
        sop_id="sop::clean-replay",
        text="Use the preserved TF-IDF and logistic-regression method.",
        retrieval_text="TF-IDF bigrams with logistic regression.",
        claim_refs=(registration.replay_claim_id,),
        claim_types=(ClaimType.METHOD_HYPOTHESIS.value,),
        source_artifact_refs=(original.subject_artifact_id,),
        source_run_ids=("run-a",),
        source_task_ids=("task-a",),
        source_task_families=("tabular",),
        source_domains=("tabular",),
        transfer_scope="same_domain",
        protocol_scope=(ref.key(),),
        task_scope={"task_ids": ["task-a"], "task_families": ["tabular"]},
        permitted_operations=(Operation.GENERATE_CANDIDATE.value,),
        permitted_generation_stages=("improve",),
        permitted_governance_stages=("retrieval",),
        publication_class="certified",
        receipt_refs=registration.receipt_ids,
        contract_spec={
            "replay_artifact_id": registration.replay_artifact_id,
            "registration_hash": registration.registration_hash,
            "verification_report_hash": registration.verification_report_hash,
            "predecessor_claim_id": registration.original_claim_id,
            "predecessor_clause_id": "clause-a",
            "historical_metric_used_as_evidence": False,
        },
        legacy_status="clean_replay_certified_v1",
    )
    result = CertifiedBundlePublisher(tmp_path, graph, registry).publish(
        new_version="v2",
        overlay=overlay,
        registrations=[registration],
        verifications={verification.report_hash: verification},
        expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
        replay_clause_publications=[
            ReplayClausePublication(
                clause=replay_clause,
                title="Certified Clean Replay TF-IDF Method",
                source_clause_id="clause-a",
                registration_hash=registration.registration_hash,
                verification_report_hash=verification.report_hash,
            )
        ],
        formal_bundle_validator=lambda candidate: {
            "valid": (candidate / "manifest.json").is_file(),
            "validator": "synthetic-formal-validator-v1",
        },
    )

    current = json.loads((tmp_path / "CURRENT.json").read_text(encoding="utf-8"))
    assert current["bundle_version"] == "v2"
    certified = tmp_path / current["bundle_path"]
    manifest = json.loads((certified / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["certification_level"] == "certified"
    assert manifest["parent_bundle"] == parent_manifest["bundle_id"]
    assert manifest["artifact_hashes"]["corpus/manifest.json"] == sha256_file(
        certified / "corpus" / "manifest.json"
    )
    assert result.certification_report["blanket_clause_upgrade"] is False
    assert result.certification_report["replay_clause_ids"] == [
        replay_clause.clause_id
    ]
    assert result.certification_report["old_claim_mutation_count"] == 0
    assert result.publication.pipeline_reports["bundle_validation"][
        "formal_validation"
    ] == {
        "valid": True,
        "validator": "synthetic-formal-validator-v1",
    }
    assert "claim::unreplayed" in result.certification_report["unreplayed_score_claim_ids"]
    replay_claim_rows = [
        json.loads(line)
        for line in (certified / "authority" / "claims.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["claim_id"] for row in replay_claim_rows} == {
        original.claim_id,
        unreplayed.claim_id,
        registration.replay_claim_id,
    }
    published_clauses = [
        json.loads(line)
        for line in (certified / "sop" / "clauses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    published_by_id = {row["clause_id"]: row for row in published_clauses}
    assert published_by_id["clause-a"]["publication_class"] == "candidate"
    assert published_by_id[replay_clause.clause_id]["claim_refs"] == [
        registration.replay_claim_id
    ]
    assert set(
        published_by_id[replay_clause.clause_id]["receipt_refs"]
    ) == set(registration.receipt_ids)
    runforest_index = np.load(certified / "runforest" / "index.npz", allow_pickle=True)
    assert replay_clause.sop_id in set(runforest_index["node_ids"].tolist())
    assert replay_clause.clause_id in set(runforest_index["node_ids"].tolist())
    clause_index = np.load(
        certified / "runforest" / "clause_index.npz", allow_pickle=True
    )
    assert replay_clause.clause_id in set(clause_index["clause_ids"].tolist())
    masks = json.loads(
        (
            certified
            / "visibility"
            / "precompiled_masks"
            / "declared_scope_masks.json"
        ).read_text(encoding="utf-8")
    )["masks"]
    mask_key = "|".join(
        [ref.key(), Operation.GENERATE_CANDIDATE.value, "improve", "retrieval"]
    )
    assert replay_clause.clause_id in masks[mask_key]
    assert manifest["graph_hashes"]["runforest"] == sha256_file(
        certified / "runforest" / "graph.json"
    )
    assert manifest["index_hashes"]["runforest"] == sha256_file(
        certified / "runforest" / "index.npz"
    )
    assert manifest["index_hashes"]["clauses"] == sha256_file(
        certified / "runforest" / "clause_index.npz"
    )
    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "next-session-overlay",
        active_protocol_ref=ref.key(),
        authority_policy_version="authority_v1",
    )
    reloaded_engine = AuthorityEngine(registry)
    load_report = load_snapshot_authority(reloaded_engine, snapshot)
    assert registration.replay_claim_id in load_report["claim_ids"]
    reloaded_claim = reloaded_engine.graph.claims[registration.replay_claim_id]
    assert (
        reloaded_engine.authorize(
            _request(reloaded_claim, ref, Operation.GENERATE_CANDIDATE)
        ).outcome
        == DecisionOutcome.ALLOW
    )
    for historical_id in (original.claim_id, unreplayed.claim_id):
        historical = reloaded_engine.graph.claims[historical_id]
        assert (
            reloaded_engine.authorize(
                _request(historical, ref, Operation.RANK)
            ).outcome
            != DecisionOutcome.ALLOW
        )
    assert {
        str(path.relative_to(parent)): sha256_file(path)
        for path in sorted(parent.rglob("*"))
        if path.is_file()
    } == parent_tree_hash


def test_historical_receipts_cannot_be_reused_for_replay_artifact() -> None:
    registry, ref = build_registry()
    verification = verify_protocol_only_patch(
        SOURCE_CODE,
        PROTOCOL_REPAIR_CODE,
        ProtocolRepairSurface.from_protocol_spec(registry.resolve(ref)),
        source_artifact_id="historical-artifact",
        replay_artifact_id="clean-replay-artifact",
    )
    graph = EvidenceGraph()
    original = historical_score_claim(
        "claim::historical",
        "historical-artifact",
        ref,
        verification.source_method_fingerprint,
    )
    graph.add_claim(original)
    historical_receipts = trusted_replay_receipts(
        TrustedCollectorHost("wrong-artifact-host"),
        artifact_id="historical-artifact",
        protocol_ref=ref,
        method_fingerprint=verification.replay_method_fingerprint,
        code_sha256=verification.replay_code_sha256,
    )

    with pytest.raises(ValueError, match="Historical/source Receipt"):
        ReplayAuthorityRecovery(graph, registry).register(
            original_claim_id=original.claim_id,
            verification=verification,
            receipts=historical_receipts,
            protocol_ref=ref,
            statement="must not be accepted",
        )

    assert set(graph.claims) == {original.claim_id}
    assert graph.paths == {}


def test_v2_replay_creates_new_protocol_claim_without_rewriting_v1() -> None:
    registry = ProtocolRegistry("mlevolve/config/protocols")
    v1 = registry.get("mlevolve-default", "1").ref()
    v2_spec = registry.get("mlevolve-default", "2")
    v2 = v2_spec.ref()
    verification = verify_protocol_only_patch(
        SOURCE_CODE,
        PROTOCOL_REPAIR_CODE,
        ProtocolRepairSurface.from_protocol_spec(v2_spec),
        source_artifact_id="historical-v1-artifact",
        replay_artifact_id="clean-v2-artifact",
    )
    graph = EvidenceGraph()
    original = historical_score_claim(
        "claim::historical-v1",
        "historical-v1-artifact",
        v1,
        verification.source_method_fingerprint,
    )
    graph.add_claim(original)
    receipts = trusted_replay_receipts(
        TrustedCollectorHost("v2-recovery-host"),
        artifact_id="clean-v2-artifact",
        protocol_ref=v2,
        method_fingerprint=verification.replay_method_fingerprint,
        code_sha256=verification.replay_code_sha256,
    )
    registration = ReplayAuthorityRecovery(graph, registry).register(
        original_claim_id=original.claim_id,
        verification=verification,
        receipts=receipts,
        protocol_ref=v2,
        statement="A new v2 replay score supports the preserved method.",
    )

    replay_claim = graph.claims[registration.replay_claim_id]
    assert graph.claims[original.claim_id].protocol_ref == v1
    assert replay_claim.protocol_ref == v2
    assert replay_claim.boundary["predecessor_protocol_ref"] == v1.key()
    assert graph.claim_paths[original.claim_id] == []
    assert (
        AuthorityEngine(registry, graph=graph)
        .authorize(_request(replay_claim, v2, Operation.RANK))
        .outcome
        == DecisionOutcome.ALLOW
    )
