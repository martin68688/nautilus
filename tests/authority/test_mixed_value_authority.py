from __future__ import annotations

import hashlib
from types import SimpleNamespace

from authority.authority_engine import AuthorityEngine
from authority.claim_decomposer import decompose_node_claims
from authority.collectors import TrustedCollectorHost
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    Operation,
    ProtocolSpec,
    TaskContext,
)
from authority.adapters.mlevolve.receipt_bridge import receipts_for_node
from authority.protocol_registry import ProtocolRegistry


def _registry():
    registry = ProtocolRegistry()
    spec = registry.register(
        ProtocolSpec(
            protocol_id="test",
            version="1",
            task_profile={},
            data_split_policy={},
            preprocessing_policy={},
            evaluator_spec={},
            metric_spec={},
            selection_policy={},
            seed_policy={},
            holdout_policy={},
            promotion_policy={},
            compatibility_rules={},
        )
    )
    return registry, spec.ref()


def _mixed_node():
    code = "oof = pred.set_index(sample_id).reindex(train_ids)\nbest = score(test_labels)"
    return SimpleNamespace(
        id="mixed",
        stage="debug",
        code=code,
        plan="Repair OOF sample_id alignment",
        analysis="",
        metric=SimpleNamespace(value=0.92, maximize=True),
        exec_time=1.0,
        is_buggy=False,
        is_valid=True,
        method_fingerprint=hashlib.sha256(code.encode()).hexdigest(),
        code_sha256_expected=hashlib.sha256(code.encode()).hexdigest(),
        leakage_audit={
            "schema": "mlevolve_leakage_audit_v2",
            "detector_status": "complete",
            "status": "blocked",
            "metric_disposition": "reject",
            "paper_grade_eligible": False,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "issues": [{
                "issue_code": "TEST_LABEL_MODEL_SELECTION",
                "category": "target_leakage",
            }],
        },
        protocol_repair={},
        derived_from_refs=[],
        claim_refs=[],
    )


def _request(ref, claim, operation, stage):
    return AuthorityRequest(
        artifact_id=claim.subject_artifact_id,
        claim_id=claim.claim_id,
        operation=operation,
        decision_stage=stage,
        active_protocol=ref,
        task_context=TaskContext("task"),
        requesting_component="test",
    )


def test_mixed_value_repair_survives_while_polluted_score_is_blocked() -> None:
    registry, ref = _registry()
    node = _mixed_node()
    decomposition = decompose_node_claims(node, ref, "task")
    receipts = receipts_for_node(
        node,
        ref,
        "run",
        collector_host=TrustedCollectorHost("test-host"),
    )
    graph = EvidenceGraph()
    for claim in decomposition.claims:
        graph.add_claim(claim)
        for receipt in receipts:
            graph.add_receipt(receipt)
        graph.add_path(EvidencePath(f"path:{claim.claim_id}", claim.claim_id, [
            receipt.receipt_id for receipt in receipts
        ]))
    engine = AuthorityEngine(registry, graph=graph)

    repair = decomposition.claims_of_type(ClaimType.DEBUG_REPAIR)[0]
    score = decomposition.claims_of_type(ClaimType.SCORE)[0]
    audit = decomposition.claims_of_type(ClaimType.AUDIT_FINDING)[0]

    repair_decision = engine.authorize(
        _request(ref, repair, Operation.REPAIR_SEED, DecisionStage.REPLAY)
    )
    score_decision = engine.authorize(
        _request(ref, score, Operation.RANK, DecisionStage.BRANCH_SELECTION)
    )
    audit_decision = engine.authorize(
        _request(ref, audit, Operation.INSPECT, DecisionStage.DEBUG)
    )

    assert repair_decision.outcome == DecisionOutcome.ALLOW
    assert score_decision.outcome == DecisionOutcome.DENY
    assert score_decision.blocking_receipts
    assert audit_decision.outcome == DecisionOutcome.ALLOW


def test_execution_receipt_cannot_upgrade_score_or_pairwise_superiority() -> None:
    registry, ref = _registry()
    node = _mixed_node()
    receipts = receipts_for_node(
        node,
        ref,
        "run",
        collector_host=TrustedCollectorHost("test-host"),
    )
    code_execution_only = [
        receipt for receipt in receipts if receipt.receipt_type.value == "code_execution"
    ]
    pairwise = Claim(
        claim_id="pairwise",
        claim_type=ClaimType.PAIRWISE_SUPERIORITY,
        subject_artifact_id="mixed",
        task_scope={"task_id": "task"},
        method_fingerprint=node.method_fingerprint,
        protocol_ref=ref,
        statement="A is better than B",
    )
    graph = EvidenceGraph()
    graph.add_claim(pairwise)
    for receipt in code_execution_only:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath("pairwise-path", "pairwise", [
        receipt.receipt_id for receipt in code_execution_only
    ]))
    decision = AuthorityEngine(registry, graph=graph).authorize(
        _request(ref, pairwise, Operation.RANK, DecisionStage.BRANCH_SELECTION)
    )
    assert decision.outcome == DecisionOutcome.REQUIRE_REPLAY
    assert decision.reason_codes == ["missing_evidence"]
    assert any(
        obligation.startswith(("receipt:", "trusted_receipt:", "count:"))
        for obligation in decision.missing_obligations
    )

    executed = Claim(
        claim_id="executed",
        claim_type=ClaimType.EXECUTED,
        subject_artifact_id="mixed",
        task_scope={"task_id": "task"},
        method_fingerprint=node.method_fingerprint,
        protocol_ref=ref,
        statement="code executed",
    )
    graph.add_claim(executed)
    graph.add_path(EvidencePath("executed-path", "executed", [
        receipt.receipt_id for receipt in code_execution_only
    ]))
    incompatible = AuthorityEngine(registry, graph=graph).authorize(
        _request(ref, executed, Operation.RANK, DecisionStage.BRANCH_SELECTION)
    )
    assert "claim_operation_compatibility" in incompatible.missing_obligations
