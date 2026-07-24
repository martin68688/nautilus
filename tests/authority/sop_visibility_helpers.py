from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from authority.authority_engine import AuthorityEngine
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    Claim,
    ClaimType,
    GenerationStage,
    GovernanceStage,
    Operation,
    ProtocolRef,
    ProtocolSpec,
    TaskContext,
    VisibilityRequest,
)
from authority.protocol_registry import ProtocolRegistry


MIXED_SOP_ID = "sop::mixed"
REPAIR_CLAUSE_ID = "clause::repair"
AUDIT_CLAUSE_ID = "clause::audit"
SCORE_CLAUSE_ID = "clause::score"
REPAIR_TEXT = "Align OOF predictions by sample_id before scoring."
AUDIT_TEXT = "Warning: historical selection read test labels."
FORBIDDEN_SCORE_TEXT = "Historical contaminated score 0.92."


def build_mixed_authority() -> tuple[AuthorityEngine, ProtocolRef]:
    registry = ProtocolRegistry()
    spec = registry.register(
        ProtocolSpec(
            protocol_id="visibility-test",
            version="1",
            task_profile={"family": "tabular"},
            data_split_policy={"kind": "grouped"},
            preprocessing_policy={"fit": "fold_train"},
            evaluator_spec={"name": "macro_f1"},
            metric_spec={"maximize": True},
            selection_policy={"freeze": True},
            seed_policy={},
            holdout_policy={"terminal_only": True},
            promotion_policy={},
            compatibility_rules={},
        )
    )
    ref = spec.ref()
    claims = [
        Claim(
            claim_id="claim::repair",
            claim_type=ClaimType.DEBUG_REPAIR,
            subject_artifact_id="artifact::mixed",
            task_scope={"task_id": "task-1"},
            method_fingerprint="method-v1",
            protocol_ref=ref,
            statement=REPAIR_TEXT,
        ),
        Claim(
            claim_id="claim::audit",
            claim_type=ClaimType.AUDIT_FINDING,
            subject_artifact_id="artifact::mixed",
            task_scope={"task_id": "task-1"},
            method_fingerprint="method-v1",
            protocol_ref=ref,
            statement=AUDIT_TEXT,
        ),
        Claim(
            claim_id="claim::score",
            claim_type=ClaimType.SCORE,
            subject_artifact_id="artifact::mixed",
            task_scope={"task_id": "task-1"},
            method_fingerprint="method-v1",
            protocol_ref=ref,
            statement=FORBIDDEN_SCORE_TEXT,
        ),
    ]
    graph = EvidenceGraph()
    for claim in claims:
        graph.add_claim(claim)
        # Empty paths satisfy receipt-free navigation operations while leaving
        # score/ranking obligations unsatisfied.
        graph.add_path(EvidencePath(f"path::{claim.claim_id}", claim.claim_id, []))
    return AuthorityEngine(registry, graph=graph), ref


def mixed_sop_node(ref: ProtocolRef) -> dict:
    common = {
        "sop_id": MIXED_SOP_ID,
        "source_artifact_refs": ["artifact::mixed"],
        "protocol_scope": [ref.key()],
        "task_scope": {"task_ids": ["task-1"], "task_families": ["tabular"]},
        "permitted_governance_stages": [GovernanceStage.RETRIEVAL.value],
        "legacy_status": "native_v1",
    }
    return {
        "id": MIXED_SOP_ID,
        "type": "SOP",
        "title": f"Mixed historical SOP: {FORBIDDEN_SCORE_TEXT}",
        "action": f"{REPAIR_TEXT} {AUDIT_TEXT} {FORBIDDEN_SCORE_TEXT}",
        "text": f"{REPAIR_TEXT}\n{AUDIT_TEXT}\n{FORBIDDEN_SCORE_TEXT}",
        "abstraction_level": "L3_repair",
        "sop_kind": "debug_fix",
        "method_family": "general",
        "task_families": ["general", "tabular"],
        "decision_stages": ["debug", "improve"],
        "compute_profile": "cpu_light",
        "clauses": [
            {
                **common,
                "clause_id": REPAIR_CLAUSE_ID,
                "text": REPAIR_TEXT,
                "retrieval_text": "OOF sample_id alignment repair",
                "claim_refs": ["claim::repair"],
                "claim_types": [ClaimType.DEBUG_REPAIR.value],
                "permitted_operations": [
                    Operation.INSPECT.value,
                    Operation.DEBUG_HYPOTHESIS.value,
                    Operation.REPAIR_SEED.value,
                ],
                "permitted_generation_stages": [GenerationStage.DEBUG.value],
                "publication_class": "diagnostic",
            },
            {
                **common,
                "clause_id": AUDIT_CLAUSE_ID,
                "text": AUDIT_TEXT,
                "retrieval_text": "test-label leakage warning",
                "claim_refs": ["claim::audit"],
                "claim_types": [ClaimType.AUDIT_FINDING.value],
                "permitted_operations": [
                    Operation.INSPECT.value,
                    Operation.DEBUG_HYPOTHESIS.value,
                    Operation.DISTILL_DIAGNOSTIC.value,
                ],
                "permitted_generation_stages": [GenerationStage.DEBUG.value],
                "publication_class": "diagnostic",
            },
            {
                **common,
                "clause_id": SCORE_CLAUSE_ID,
                "text": FORBIDDEN_SCORE_TEXT,
                "retrieval_text": FORBIDDEN_SCORE_TEXT,
                "claim_refs": ["claim::score"],
                "claim_types": [ClaimType.SCORE.value],
                "permitted_operations": [
                    Operation.INSPECT.value,
                    Operation.RANK.value,
                    Operation.SELECT.value,
                    Operation.PROMOTE.value,
                ],
                "permitted_generation_stages": [GenerationStage.IMPROVE.value],
                "publication_class": "certified",
            },
        ],
    }


def mixed_nodes(ref: ProtocolRef) -> dict[str, dict]:
    node = mixed_sop_node(ref)
    return {MIXED_SOP_ID: node}


def visibility_request(
    ref: ProtocolRef,
    operation: Operation,
    *,
    generation_stage: GenerationStage,
    governance_stage: GovernanceStage = GovernanceStage.RETRIEVAL,
    token_budget: int = 4096,
    task_id: str = "task-1",
    task_family: str = "tabular",
    policy_version: str = "authority_v1",
) -> VisibilityRequest:
    return VisibilityRequest(
        operation=operation,
        generation_stage=generation_stage,
        governance_stage=governance_stage,
        active_protocol=ref,
        task_context=TaskContext(task_id=task_id, task_family=task_family),
        memory_bundle_version="bundle-v1",
        token_budget=token_budget,
        requesting_component="tests.sop_visibility",
        authority_policy_version=policy_version,
    )


def decision_snapshot(
    ref: ProtocolRef,
    *,
    outcome: str = "allow",
    claim_id: str = "claim::score",
    claim_type: str = ClaimType.SCORE.value,
    artifact_id: str = "artifact::mixed",
    operation: Operation = Operation.RANK,
    generation_stage: GenerationStage = GenerationStage.IMPROVE,
    governance_stage: GovernanceStage = GovernanceStage.RETRIEVAL,
    policy_version: str = "authority_v1",
    task_id: str = "task-1",
) -> dict:
    return {
        "decision_id": "decision::snapshot",
        "outcome": outcome,
        "policy_version": policy_version,
        "claim_id": claim_id,
        "artifact_id": artifact_id,
        "operation": operation.value,
        "decision_stage": governance_stage.value,
        "generation_stage": generation_stage.value,
        "governance_stage": governance_stage.value,
        "permitted_scope": {
            "claim_types": [claim_type],
            "operations": [operation.value],
            "stages": [governance_stage.value],
            "protocol_hashes": [ref.canonical_hash],
            "task_ids": [task_id],
            "generation_stages": [generation_stage.value],
            "governance_stages": [governance_stage.value],
        },
        "satisfied_paths": ["path::snapshot"],
        "missing_obligations": [],
        "blocking_receipts": [],
        "required_action": None,
    }


def write_stage_fixture(
    tmp_path: Path,
    ref: ProtocolRef,
    *,
    edge_kind: str = "navigation_attached_to",
    edge_outcome: str = "quarantine",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    clean_audit = {
        "schema": "mlevolve_leakage_audit_v2",
        "status": "clean",
        "memory_disposition": "positive_eligible",
        "paper_grade_eligible": True,
        "rank_eligible": True,
        "issues": [],
    }
    sop = mixed_sop_node(ref)
    nodes = [
        {"id": "run::1", "type": "Run", "run_id": "run-1"},
        {
            "id": "node::parent",
            "type": "RunNode",
            "run_id": "run-1",
            "run_short_id": "run-1",
            "task": "task-1",
            "stage": "debug",
            "step": 0,
            "text": "OOF alignment failure",
            "metric": 0.4,
            "metric_improvement": 0.0,
            "is_buggy": False,
            "is_valid": True,
            "leakage_audit": clean_audit,
        },
        {
            "id": "node::child",
            "type": "RunNode",
            "run_id": "run-1",
            "run_short_id": "run-1",
            "task": "task-1",
            "stage": "improve",
            "step": 1,
            "parent_id": "node::parent",
            "local_best_node_id": "node::child",
            "text": "OOF alignment repaired without forbidden score text",
            "metric": 0.5,
            "metric_improvement": 0.1,
            "is_buggy": False,
            "is_valid": True,
            "leakage_audit": clean_audit,
        },
        {
            "id": "transition::1",
            "type": "Transition",
            "run_id": "run-1",
            "run_short_id": "run-1",
            "task": "task-1",
            "parent_node_id": "node::parent",
            "child_node_id": "node::child",
            "stage_pair": "debug->improve",
            "outcome": "metric_improved",
            "metric_improvement": 0.1,
            "text": "align OOF predictions by sample_id",
            # These deprecated/forged fields must never create an edge.
            "attached_sop_ids": [MIXED_SOP_ID],
            "navigation_sop_ids": [MIXED_SOP_ID],
            "attachment_quality": [
                {"sop_id": MIXED_SOP_ID, "quality": "evidence_turn_match", "score": 1.0}
            ],
        },
        sop,
    ]
    edges = [
        {"src": "node::parent", "dst": "node::child", "kind": "parent_of"},
        {"src": "node::parent", "dst": "transition::1", "kind": "has_transition"},
        {"src": "transition::1", "dst": "node::child", "kind": "transition_to"},
        {
            "src": "transition::1",
            "dst": MIXED_SOP_ID,
            "kind": edge_kind,
            "authority_outcome": edge_outcome,
            "quality": "evidence_turn_match",
            "score": 1.0,
        },
    ]
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.npz"
    graph_path.write_text(
        json.dumps(
            {
                "meta": {
                    "schema": "hyperbolic_run_forest_memory_v1",
                    "bundle_version": "bundle-v1",
                    "leak_verified": True,
                    "paper_grade": True,
                    "leak_audited": True,
                    "positive_admission_enforced": True,
                },
                "nodes": nodes,
                "edges": edges,
            }
        ),
        encoding="utf-8",
    )
    node_ids = np.asarray([node["id"] for node in nodes], dtype=object)
    coords = np.zeros((len(nodes), 2), dtype=np.float32)
    for index in range(len(nodes)):
        coords[index] = [0.01 * index, 0.005 * index]
    np.savez(
        index_path,
        node_ids=node_ids,
        poincare=coords,
        flat_twin=coords.copy(),
        euclidean=np.zeros((len(nodes), 8), dtype=np.float32),
    )
    return graph_path, index_path


def make_stage_layer(
    tmp_path: Path,
    engine: AuthorityEngine,
    ref: ProtocolRef,
    *,
    edge_kind: str = "navigation_attached_to",
    edge_outcome: str = "quarantine",
):
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    graph_path, index_path = write_stage_fixture(
        tmp_path,
        ref,
        edge_kind=edge_kind,
        edge_outcome=edge_outcome,
    )
    return StageAwareHybridMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        enable_agentic=False,
        top_k=6,
        visibility_mode="enforce",
        visibility_authority_engine=engine,
        visibility_active_protocol=ref,
        visibility_policy_version=engine.policy_version,
        visibility_task_id="task-1",
        visibility_bundle_version="bundle-v1",
        visibility_token_budget=4096,
    )
