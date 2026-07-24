from __future__ import annotations

import copy

from authority.actuation import (
    ActuationLevel,
    ActuationTracker,
    ExperienceContractCompiler,
)
from authority.authority_engine import AuthorityEngine
from authority.collectors import (
    CodeExecutionCollector,
    EvaluatorIntegrityCollector,
    FitScopeCollector,
    MethodIdentityCollector,
    PredictionScopeCollector,
    SelectionFreezeCollector,
    SplitLineageCollector,
    TrustedCollectorHost,
)
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    GenerationStage,
    GovernanceStage,
    Operation,
    ProtocolRef,
    SOPClauseV1,
    TaskContext,
    VisibilityRequest,
)
from authority.protocol_registry import ProtocolRegistry


DIGEST = "a" * 64
REGISTRY = ProtocolRegistry("mlevolve/config/protocols")
ACTIVE = REGISTRY.get("mlevolve-default", "1").ref()


def _contract():
    clause = SOPClauseV1(
        clause_id="clause-actuation",
        sop_id="sop-actuation",
        text="Apply a protocol-clean feature repair.",
        retrieval_text="Apply a protocol-clean feature repair.",
        claim_refs=("claim-score",),
        claim_types=(ClaimType.SCORE.value,),
        source_artifact_refs=("artifact",),
        source_transition_refs=(
            "run::source-image-run::transition::transition-a",
        ),
        source_run_ids=("source-image-run",),
        source_task_ids=("source-image-task",),
        source_task_families=("General Image",),
        source_domains=("image",),
        transfer_scope="same_domain",
        task_scope={"task_ids": ["source-image-task"]},
        publication_class="certified",
        applies_when=("the feature path is active",),
        prevents=("invalid feature values",),
    )
    request = VisibilityRequest(
        operation=Operation.GENERATE_CANDIDATE,
        generation_stage=GenerationStage.IMPROVE,
        governance_stage=GovernanceStage.RETRIEVAL,
        active_protocol=ACTIVE,
        task_context=TaskContext(
            task_id="task-a",
            task_family="image_binary_classification",
        ),
        memory_bundle_version="bundle-v1",
        token_budget=1000,
        requesting_component="tests",
    )
    return ExperienceContractCompiler().compile(clause, request)


def _observations(contract):
    preconditions = {
        predicate.name: predicate.expected for predicate in contract.preconditions
    }
    static = {
        predicate.name: predicate.expected
        for predicate in (
            contract.must_preserve
            + contract.must_change
            + contract.must_not_use
        )
    }
    runtime = {
        predicate.name: predicate.expected
        for predicate in contract.expected_runtime_observations
    }
    return preconditions, static, runtime


def test_l0_through_l3_are_sequential_and_host_verified() -> None:
    host = TrustedCollectorHost("actuation-host")
    tracker = ActuationTracker(
        collector_host=host,
        protocol_ref=ACTIVE,
        run_id="run-a",
    )
    contract = _contract()
    tracker.record_exposure(
        artifact_id="artifact",
        contracts=[contract],
        request_id="request-a",
    )
    exposed = tracker.report(
        artifact_id="artifact", contract_id=contract.contract_id
    )
    assert exposed.highest_level == ActuationLevel.EXPOSED
    assert exposed.promotion_eligible is False
    assert exposed.clause_id == "clause-actuation"
    assert exposed.sop_id == "sop-actuation"
    assert exposed.source_run_ids == ["source-image-run"]
    assert exposed.source_task_ids == ["source-image-task"]
    assert exposed.source_task_families == ["General Image"]
    assert exposed.source_domains == ["image"]
    assert exposed.transfer_scope == "same_domain"
    assert exposed.target_scope == {
        "task_id": "task-a",
        "task_family": "image_binary_classification",
        "domain": "image",
    }
    assert exposed.reached(ActuationLevel.EXPOSED)
    assert not exposed.reached(ActuationLevel.CLAIMED_ADOPTION)

    tracker.record_claimed_adoption(
        artifact_id="artifact", contract_id=contract.contract_id
    )
    preconditions, static, runtime = _observations(contract)
    static_receipt = tracker.record_static_observation(
        artifact_id="artifact",
        contract_id=contract.contract_id,
        preconditions=preconditions,
        observations=static,
    )
    assert static_receipt is not None
    static_report = tracker.report(
        artifact_id="artifact", contract_id=contract.contract_id
    )
    assert static_report.highest_level == ActuationLevel.STATIC_CONFORMANT
    assert static_report.promotion_eligible is False

    runtime_receipt = tracker.record_runtime_observation(
        artifact_id="artifact",
        contract_id=contract.contract_id,
        observations=runtime,
    )
    assert runtime_receipt is not None
    runtime_report = tracker.report(
        artifact_id="artifact", contract_id=contract.contract_id
    )
    assert runtime_report.highest_level == ActuationLevel.RUNTIME_CONFORMANT
    assert runtime_report.promotion_eligible is True
    assert len(runtime_report.levels) == 6


def test_exposure_ledger_contains_clause_source_and_domain_provenance() -> None:
    class RecordingLedger:
        def __init__(self) -> None:
            self.events = []

        def append(self, event_type, payload) -> None:
            self.events.append((event_type, payload))

    ledger = RecordingLedger()
    tracker = ActuationTracker(
        collector_host=TrustedCollectorHost("ledger-host"),
        protocol_ref=ACTIVE,
        run_id="run-ledger",
        ledger=ledger,
    )
    contract = _contract()
    tracker.record_exposure(
        artifact_id="artifact",
        contracts=[contract],
        request_id="request-ledger",
        prompt_sha256="f" * 64,
    )
    event_type, event = ledger.events[-1]
    assert event_type == "experience_exposed"
    assert event["clause_id"] == "clause-actuation"
    assert event["sop_id"] == "sop-actuation"
    assert event["source_refs"] == [
        "artifact",
        "run::source-image-run::transition::transition-a",
    ]
    assert event["source_run_ids"] == ["source-image-run"]
    assert event["source_task_ids"] == ["source-image-task"]
    assert event["source_task_families"] == ["General Image"]
    assert event["source_domains"] == ["image"]
    assert event["transfer_scope"] == "same_domain"
    assert event["target_scope"] == {
        "task_id": "task-a",
        "task_family": "image_binary_classification",
        "domain": "image",
    }
    assert event["schema"] == "experience_exposure_event_v2"
    assert event["operation"] == Operation.GENERATE_CANDIDATE.value
    assert event["generation_stage"] == GenerationStage.IMPROVE.value
    assert event["governance_stage"] == GovernanceStage.RETRIEVAL.value
    assert event["publication_class"] == "certified"
    assert event["minimum_writeback_level"] == int(
        ActuationLevel.RUNTIME_CONFORMANT
    )
    assert event["policy_version"] == "authority_v1"
    assert event["compiler_version"] == "experience_contract_compiler_v1"


def _base_score_receipts(host: TrustedCollectorHost):
    specifications = (
        (MethodIdentityCollector, {"method_fingerprint": DIGEST, "code_sha256": DIGEST}),
        (CodeExecutionCollector, {"exit_status": 0, "executed_path": "artifact", "run_hash": DIGEST}),
        (SplitLineageCollector, {"partition_hashes": {"train": DIGEST}, "overlap_count": 0}),
        (FitScopeCollector, {"fit_scope_hashes": {"model": DIGEST}, "holdout_fit_count": 0}),
        (PredictionScopeCollector, {"prediction_scope_hashes": {"valid": DIGEST}, "forbidden_overlap_count": 0}),
        (EvaluatorIntegrityCollector, {"evaluator_hash": DIGEST, "inputs_hash": DIGEST, "metric_direction": "maximize", "tampered": False}),
        (SelectionFreezeCollector, {"candidate_set_hash": DIGEST, "frozen_before_holdout": True}),
    )
    return [
        host.collect(
            collector,
            artifact_id="artifact",
            run_id="run-a",
            protocol_ref=ACTIVE,
            source="tests.host",
            payload=payload,
        )
        for collector, payload in specifications
    ]


def _promotion_request(
    operation: Operation = Operation.PROMOTE,
    *,
    claim_id: str = "claim-score",
) -> AuthorityRequest:
    return AuthorityRequest(
        artifact_id="artifact",
        claim_id=claim_id,
        operation=operation,
        decision_stage=DecisionStage.MEMORY_WRITEBACK,
        active_protocol=ACTIVE,
        task_context=TaskContext(task_id="task-a"),
        requesting_component="tests",
        generation_stage=GenerationStage.IMPROVE,
        governance_stage=GovernanceStage.MEMORY_WRITEBACK,
    )


def test_protocol_legal_result_can_enter_memory_without_experience_actuation() -> None:
    host = TrustedCollectorHost("result-promotion-host")
    claim = Claim(
        claim_id="claim-score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task-a"},
        method_fingerprint=DIGEST,
        protocol_ref=ACTIVE,
        statement="Artifact reports a protocol-legal score.",
    )
    graph = EvidenceGraph()
    graph.add_claim(claim)
    receipts = _base_score_receipts(host)
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id="path-score",
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipts],
        )
    )

    decision = AuthorityEngine(REGISTRY, graph=graph).authorize(
        _promotion_request(Operation.PROMOTE_RESULT)
    )

    assert decision.outcome == DecisionOutcome.ALLOW
    assert "receipt:static_actuation" not in decision.missing_obligations
    assert "receipt:runtime_actuation" not in decision.missing_obligations


def test_exposure_or_claim_without_l3_cannot_promote() -> None:
    host = TrustedCollectorHost("promotion-host")
    contract = _contract()
    claim = Claim(
        claim_id="claim-score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task-a"},
        method_fingerprint=DIGEST,
        protocol_ref=ACTIVE,
        statement="Artifact reports a protocol-legal score.",
        boundary={"experience_contract_hash": contract.contract_hash},
    )
    graph = EvidenceGraph()
    graph.add_claim(claim)
    base_receipts = _base_score_receipts(host)
    for receipt in base_receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id="path-score",
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in base_receipts],
        )
    )
    engine = AuthorityEngine(REGISTRY, graph=graph)
    denied = engine.authorize(_promotion_request())
    assert denied.outcome != DecisionOutcome.ALLOW
    assert "receipt:static_actuation" in denied.missing_obligations
    assert "receipt:runtime_actuation" in denied.missing_obligations

    tracker = ActuationTracker(
        collector_host=host,
        protocol_ref=ACTIVE,
        run_id="run-a",
    )
    tracker.record_exposure(
        artifact_id="artifact", contracts=[contract], request_id="request-a"
    )
    tracker.record_claimed_adoption(
        artifact_id="artifact", contract_id=contract.contract_id
    )
    preconditions, static, runtime = _observations(contract)
    tracker.record_static_observation(
        artifact_id="artifact",
        contract_id=contract.contract_id,
        preconditions=preconditions,
        observations=static,
    )
    tracker.record_runtime_observation(
        artifact_id="artifact",
        contract_id=contract.contract_id,
        observations=runtime,
    )
    all_receipts = [*base_receipts, *tracker.receipts_for_artifact("artifact")]
    for receipt in all_receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id="path-score",
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in all_receipts],
        )
    )
    allowed = engine.authorize(_promotion_request())
    assert allowed.outcome == DecisionOutcome.ALLOW


def test_adoption_edge_requires_l3_even_when_result_itself_is_promotable() -> None:
    host = TrustedCollectorHost("adoption-publication-host")
    contract = _contract()
    method_claim = Claim(
        claim_id="claim-method",
        claim_type=ClaimType.EXPERIENCE_ADOPTION,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task-a"},
        method_fingerprint=DIGEST,
        protocol_ref=ACTIVE,
        statement="Artifact realizes an exposed method.",
        boundary={"experience_contract_hash": contract.contract_hash},
    )
    graph = EvidenceGraph()
    graph.add_claim(method_claim)
    base_receipts = _base_score_receipts(host)
    for receipt in base_receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id="path-method",
            claim_id=method_claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in base_receipts],
        )
    )
    engine = AuthorityEngine(REGISTRY, graph=graph)

    denied = engine.authorize(
        _promotion_request(
            Operation.PUBLISH_ADOPTION,
            claim_id=method_claim.claim_id,
        )
    )
    assert denied.outcome != DecisionOutcome.ALLOW
    assert {
        "receipt:static_actuation",
        "receipt:runtime_actuation",
    } <= set(denied.missing_obligations)

    tracker = ActuationTracker(
        collector_host=host,
        protocol_ref=ACTIVE,
        run_id="run-a",
    )
    tracker.record_exposure(
        artifact_id="artifact", contracts=[contract], request_id="request-a"
    )
    tracker.record_claimed_adoption(
        artifact_id="artifact", contract_id=contract.contract_id
    )
    preconditions, static, runtime = _observations(contract)
    tracker.record_static_observation(
        artifact_id="artifact",
        contract_id=contract.contract_id,
        preconditions=preconditions,
        observations=static,
    )
    tracker.record_runtime_observation(
        artifact_id="artifact",
        contract_id=contract.contract_id,
        observations=runtime,
    )
    all_receipts = [
        *base_receipts,
        *tracker.receipts_for_artifact("artifact"),
    ]
    for receipt in all_receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id="path-method",
            claim_id=method_claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in all_receipts],
        )
    )

    allowed = engine.authorize(
        _promotion_request(
            Operation.PUBLISH_ADOPTION,
            claim_id=method_claim.claim_id,
        )
    )
    assert allowed.outcome == DecisionOutcome.ALLOW


def test_actuation_receipts_cannot_cross_authorize_another_contract() -> None:
    host = TrustedCollectorHost("contract-binding-host")
    contract_a = _contract()
    contract_b = copy.deepcopy(contract_a)
    contract_b.clause_id = "clause-other-contract"
    contract_b.finalize()
    tracker = ActuationTracker(
        collector_host=host,
        protocol_ref=ACTIVE,
        run_id="run-a",
    )
    tracker.record_exposure(
        artifact_id="artifact",
        contracts=[contract_a, contract_b],
        request_id="request-contract-binding",
    )
    tracker.record_claimed_adoption(
        artifact_id="artifact", contract_id=contract_a.contract_id
    )
    preconditions, static, runtime = _observations(contract_a)
    tracker.record_static_observation(
        artifact_id="artifact",
        contract_id=contract_a.contract_id,
        preconditions=preconditions,
        observations=static,
    )
    tracker.record_runtime_observation(
        artifact_id="artifact",
        contract_id=contract_a.contract_id,
        observations=runtime,
    )
    claim_b = Claim(
        claim_id="claim-adoption-b",
        claim_type=ClaimType.EXPERIENCE_ADOPTION,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task-a"},
        method_fingerprint=DIGEST,
        protocol_ref=ACTIVE,
        statement="Artifact adopts contract B.",
        boundary={"experience_contract_hash": contract_b.contract_hash},
    )
    graph = EvidenceGraph()
    graph.add_claim(claim_b)
    receipts = [
        *_base_score_receipts(host),
        *tracker.receipts_for_artifact("artifact"),
    ]
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id="path-adoption-b",
            claim_id=claim_b.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipts],
        )
    )

    denied = AuthorityEngine(REGISTRY, graph=graph).authorize(
        _promotion_request(
            Operation.PUBLISH_ADOPTION,
            claim_id=claim_b.claim_id,
        )
    )

    assert denied.outcome != DecisionOutcome.ALLOW
    assert {
        "payload:static_actuation",
        "payload:runtime_actuation",
    } <= set(denied.missing_obligations)
