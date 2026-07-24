from __future__ import annotations

from authority.actuation import ActuationTracker
from authority.authority_engine import AuthorityEngine
from authority.collectors import DerivationCollector, TrustedCollectorHost
from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    GenerationStage,
    GovernanceStage,
    Operation,
    TaskContext,
)
from authority.positive_distillation import (
    PositiveDistillationKind,
    authorize_positive_distillation,
)
from authority.protocol_registry import ProtocolRegistry
from tests.authority.test_actuation_pipeline import (
    ACTIVE,
    DIGEST,
    _base_score_receipts,
    _contract,
    _observations,
)


def _derivation_receipt(host: TrustedCollectorHost, claim_id: str):
    return host.collect(
        DerivationCollector,
        artifact_id="artifact",
        run_id="run-a",
        protocol_ref=ACTIVE,
        source="tests.sleep_time_binder",
        payload={
            "parent_claim_refs": [claim_id],
            "mapping_hash": "b" * 64,
            "scope_widened": False,
        },
    )


def test_positive_result_distillation_uses_target_evidence_not_actuation() -> None:
    host = TrustedCollectorHost("positive-result-host")
    claim = Claim(
        claim_id="claim-result-score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task-a"},
        method_fingerprint=DIGEST,
        protocol_ref=ACTIVE,
        statement="The current target node has a protocol-legal score.",
    )
    receipts = [
        *_base_score_receipts(host),
        _derivation_receipt(host, claim.claim_id),
    ]
    engine = AuthorityEngine(ProtocolRegistry("mlevolve/config/protocols"))

    result = authorize_positive_distillation(
        engine,
        kind=PositiveDistillationKind.RESULT,
        claim=claim,
        receipts=receipts,
        active_protocol=ACTIVE,
        task_context=TaskContext(task_id="task-a"),
        text="Reuse the independently validated target-node method.",
        sop_id="sop-positive-result",
        source_run_ids=["run-a"],
        source_task_ids=["task-a"],
        source_domains=["image"],
    )

    assert result.decision.outcome == DecisionOutcome.ALLOW
    assert result.clause is not None
    assert result.derived_claim is not None
    assert result.evidence_path is not None
    assert result.clause.publication_class == "positive_result"
    assert result.clause.claim_types == (ClaimType.METHOD_HYPOTHESIS.value,)
    assert result.clause.claim_refs == (result.derived_claim.claim_id,)
    assert result.derived_claim.parent_claims == [claim.claim_id]
    generated = engine.authorize(
        AuthorityRequest(
            artifact_id="artifact",
            claim_id=result.derived_claim.claim_id,
            operation=Operation.GENERATE_CANDIDATE,
            decision_stage=DecisionStage.BRANCH_SELECTION,
            active_protocol=ACTIVE,
            task_context=TaskContext(task_id="task-a"),
            requesting_component="tests.positive_result_generation",
            generation_stage=GenerationStage.IMPROVE,
            governance_stage=GovernanceStage.RETRIEVAL,
        )
    )
    assert generated.outcome == DecisionOutcome.ALLOW
    assert not {
        "receipt:static_actuation",
        "receipt:runtime_actuation",
    } & set(result.decision.missing_obligations)


def test_positive_adopted_distillation_requires_contract_bound_l3() -> None:
    host = TrustedCollectorHost("positive-adopted-host")
    contract = _contract()
    claim = Claim(
        claim_id="claim-adoption",
        claim_type=ClaimType.EXPERIENCE_ADOPTION,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task-a"},
        method_fingerprint=DIGEST,
        protocol_ref=ACTIVE,
        statement="The target artifact adopted the source experience.",
        boundary={
            "experience_contract_hash": contract.contract_hash,
            "required_actuation_level": 3,
        },
    )
    base = [
        *_base_score_receipts(host),
        _derivation_receipt(host, claim.claim_id),
    ]
    engine = AuthorityEngine(ProtocolRegistry("mlevolve/config/protocols"))
    denied = authorize_positive_distillation(
        engine,
        kind=PositiveDistillationKind.ADOPTED,
        claim=claim,
        receipts=base,
        active_protocol=ACTIVE,
        task_context=TaskContext(task_id="task-a"),
        text="Reuse the verified adopted experience.",
        sop_id="sop-positive-adopted",
    )
    assert denied.decision.outcome != DecisionOutcome.ALLOW
    assert denied.clause is None
    assert {
        "receipt:static_actuation",
        "receipt:runtime_actuation",
    } <= set(denied.decision.missing_obligations)

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
    allowed = authorize_positive_distillation(
        engine,
        kind=PositiveDistillationKind.ADOPTED,
        claim=claim,
        receipts=[*base, *tracker.receipts_for_artifact("artifact")],
        active_protocol=ACTIVE,
        task_context=TaskContext(task_id="task-a"),
        text="Reuse the verified adopted experience.",
        sop_id="sop-positive-adopted",
    )
    assert allowed.decision.outcome == DecisionOutcome.ALLOW
    assert allowed.clause is not None
    assert allowed.derived_claim is not None
    assert allowed.clause.publication_class == "positive_adopted"
    assert allowed.clause.claim_types == (
        ClaimType.METHOD_HYPOTHESIS.value,
    )
    assert allowed.derived_claim.parent_claims == [claim.claim_id]


def test_legacy_positive_distillation_is_always_ambiguous_fail_closed() -> None:
    host = TrustedCollectorHost("legacy-positive-host")
    claim = Claim(
        claim_id="claim-legacy-score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task-a"},
        method_fingerprint=DIGEST,
        protocol_ref=ACTIVE,
        statement="A legacy ambiguous positive Claim.",
    )
    receipts = [
        *_base_score_receipts(host),
        _derivation_receipt(host, claim.claim_id),
    ]
    engine = AuthorityEngine(ProtocolRegistry("mlevolve/config/protocols"))
    engine.graph.add_claim(claim)
    for receipt in receipts:
        engine.graph.add_receipt(receipt)
    from authority.evidence_graph import EvidencePath

    engine.graph.add_path(
        EvidencePath(
            "legacy-positive-path",
            claim.claim_id,
            [receipt.receipt_id for receipt in receipts],
        )
    )

    decision = engine.authorize(
        AuthorityRequest(
            artifact_id="artifact",
            claim_id=claim.claim_id,
            operation=Operation.DISTILL_POSITIVE,
            decision_stage=DecisionStage.DISTILLATION,
            active_protocol=ACTIVE,
            task_context=TaskContext(task_id="task-a"),
            requesting_component="tests.legacy_positive",
            generation_stage=GenerationStage.EVOLUTION,
            governance_stage=GovernanceStage.DISTILLATION,
        )
    )
    assert decision.outcome != DecisionOutcome.ALLOW
    assert (
        "explicit_positive_distillation_semantics"
        in decision.missing_obligations
    )
