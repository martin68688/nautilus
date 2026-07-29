from __future__ import annotations

from dataclasses import replace

import pytest

from authority.authority_engine import AuthorityEngine
from authority.collectors import (
    HostObservation,
    MethodIdentityCollector,
    TrustedCollectorHost,
    UntrustedObservationError,
)
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    Operation,
    ProtocolRef,
    ProtocolSpec,
    ReceiptType,
    TaskContext,
)
from authority.protocol_registry import ProtocolRegistry
from authority.receipt_collectors import make_receipt


DIGEST = "a" * 64


def _registry():
    registry = ProtocolRegistry()
    spec = registry.register(ProtocolSpec(protocol_id="test", version="1"))
    return registry, spec.ref()


def test_agent_verified_payload_remains_legacy_and_cannot_satisfy_rank() -> None:
    registry, ref = _registry()
    claim = Claim(
        claim_id="score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task"},
        method_fingerprint=DIGEST,
        protocol_ref=ref,
        statement="score",
    )
    receipts = [
        make_receipt(
            receipt_type,
            "artifact",
            "run",
            ref,
            "agent.self_report",
            {"verified": True, "ok": True},
        )
        for receipt_type in (
            ReceiptType.METHOD_IDENTITY,
            ReceiptType.CODE_EXECUTION,
            ReceiptType.SPLIT_LINEAGE,
            ReceiptType.FIT_SCOPE,
            ReceiptType.PREDICTION_SCOPE,
            ReceiptType.EVALUATOR,
            ReceiptType.SELECTION_FREEZE,
        )
    ]
    assert {receipt.trust_status for receipt in receipts} == {"legacy_static_only"}
    graph = EvidenceGraph()
    graph.add_claim(claim)
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath("path", "score", [receipt.receipt_id for receipt in receipts]))
    decision = AuthorityEngine(registry, graph=graph).authorize(
        AuthorityRequest(
            artifact_id="artifact",
            claim_id="score",
            operation=Operation.RANK,
            decision_stage=DecisionStage.BRANCH_SELECTION,
            active_protocol=ref,
            task_context=TaskContext("task"),
            requesting_component="test",
        )
    )
    assert decision.outcome == DecisionOutcome.QUARANTINE
    assert decision.reason_codes == ["untrusted_evidence"]
    assert all(item.startswith("trusted_receipt:") for item in decision.missing_obligations)


def test_forged_host_observation_cannot_reach_collector() -> None:
    host = TrustedCollectorHost("real-host")
    forged = HostObservation(
        observation_id="forged",
        receipt_type=ReceiptType.METHOD_IDENTITY,
        artifact_id="artifact",
        run_id="run",
        protocol_ref=ProtocolRef("test", "1", DIGEST),
        source="agent",
        payload={"method_fingerprint": DIGEST, "code_sha256": DIGEST, "verified": True},
        payload_hash=DIGEST,
        observed_at="now",
        _capability=object(),
    )
    with pytest.raises(UntrustedObservationError, match="not minted"):
        MethodIdentityCollector(host).collect(forged)
