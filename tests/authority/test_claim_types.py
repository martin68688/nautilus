from __future__ import annotations

from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionStage,
    Operation,
    ProtocolSpec,
    ReceiptType,
    TaskContext,
)
from authority.protocol_compiler import ProtocolCompiler
from authority.protocol_registry import ProtocolRegistry
from authority.stage_ontology import GenerationStage, GovernanceStage


def _registry():
    registry = ProtocolRegistry()
    spec = registry.register(
        ProtocolSpec(
            protocol_id="test",
            version="1",
            task_profile={"family": "classification"},
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
    return registry, spec.ref()


def _request(protocol_ref, operation: Operation) -> AuthorityRequest:
    return AuthorityRequest(
        artifact_id="artifact",
        claim_id="method",
        operation=operation,
        decision_stage=DecisionStage.DISTILLATION,
        active_protocol=protocol_ref,
        task_context=TaskContext("task"),
        requesting_component="test",
        generation_stage=GenerationStage.EVOLUTION,
        governance_stage=GovernanceStage.DISTILLATION,
    )


def test_claim_and_distillation_operation_taxonomy_is_complete() -> None:
    assert {item.value for item in ClaimType}.issuperset(
        {
            "method_hypothesis",
            "debug_repair",
            "audit_finding",
            "experience_adoption",
        }
    )
    assert {item.value for item in Operation}.issuperset(
        {
            "distill_diagnostic",
            "distill_candidate",
            "distill_positive",
            "distill_positive_result",
            "distill_positive_adopted",
            "promote_result",
            "publish_adoption",
            "publish_causal",
        }
    )


def test_result_adoption_and_causal_publication_have_distinct_obligations() -> None:
    registry, protocol_ref = _registry()
    compiler = ProtocolCompiler(registry)
    score = Claim(
        claim_id="score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task"},
        method_fingerprint="method-hash",
        protocol_ref=protocol_ref,
        statement="The current artifact reports a protocol-legal score.",
    )
    adoption_claim = Claim(
        claim_id="method",
        claim_type=ClaimType.EXPERIENCE_ADOPTION,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task"},
        method_fingerprint="method-hash",
        protocol_ref=protocol_ref,
        statement="The current artifact realizes an admitted method.",
    )
    causal = Claim(
        claim_id="causal",
        claim_type=ClaimType.CAUSAL_ATTRIBUTION,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task"},
        method_fingerprint="method-hash",
        protocol_ref=protocol_ref,
        statement="The admitted experience changed the resulting program.",
    )

    result = compiler.compile(
        score, _request(protocol_ref, Operation.PROMOTE_RESULT)
    )
    adoption = compiler.compile(
        adoption_claim, _request(protocol_ref, Operation.PUBLISH_ADOPTION)
    )
    causal_publication = compiler.compile(
        causal, _request(protocol_ref, Operation.PUBLISH_CAUSAL)
    )

    assert ReceiptType.STATIC_ACTUATION not in result.required_receipts
    assert ReceiptType.RUNTIME_ACTUATION not in result.required_receipts
    assert {
        ReceiptType.CODE_EXECUTION,
        ReceiptType.SPLIT_LINEAGE,
        ReceiptType.FIT_SCOPE,
        ReceiptType.PREDICTION_SCOPE,
        ReceiptType.EVALUATOR,
        ReceiptType.SELECTION_FREEZE,
    }.issubset(result.required_receipts)
    assert {
        ReceiptType.STATIC_ACTUATION,
        ReceiptType.RUNTIME_ACTUATION,
    }.issubset(adoption.required_receipts)
    assert ReceiptType.COUNTERFACTUAL_ACTUATION not in adoption.required_receipts
    assert {
        ReceiptType.STATIC_ACTUATION,
        ReceiptType.RUNTIME_ACTUATION,
        ReceiptType.COUNTERFACTUAL_ACTUATION,
    }.issubset(causal_publication.required_receipts)


def test_legacy_distill_maps_to_positive_policy_fail_closed() -> None:
    _registry_instance, protocol_ref = _registry()
    request = _request(protocol_ref, Operation.DISTILL)
    assert request.operation == Operation.DISTILL_POSITIVE
    assert request.legacy_operation == Operation.DISTILL.value


def test_distillation_classes_compile_different_evidence_obligations() -> None:
    registry, protocol_ref = _registry()
    compiler = ProtocolCompiler(registry)
    claim = Claim(
        claim_id="method",
        claim_type=ClaimType.METHOD_HYPOTHESIS,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task"},
        method_fingerprint="method-hash",
        protocol_ref=protocol_ref,
        statement="A provisional method hypothesis",
    )

    diagnostic = compiler.compile(claim, _request(protocol_ref, Operation.DISTILL_DIAGNOSTIC))
    candidate = compiler.compile(claim, _request(protocol_ref, Operation.DISTILL_CANDIDATE))
    score_claim = Claim(
        **{
            **claim.__dict__,
            "claim_id": "score",
            "claim_type": ClaimType.SCORE,
            "statement": "The current result has a protocol-legal score.",
        }
    )
    positive_result = compiler.compile(
        score_claim,
        _request(protocol_ref, Operation.DISTILL_POSITIVE_RESULT),
    )
    adoption_claim = Claim(
        **{
            **claim.__dict__,
            "claim_id": "adoption",
            "claim_type": ClaimType.EXPERIENCE_ADOPTION,
            "statement": "The current artifact adopted the source experience.",
        }
    )
    positive_adopted = compiler.compile(
        adoption_claim,
        _request(protocol_ref, Operation.DISTILL_POSITIVE_ADOPTED),
    )

    assert diagnostic.required_receipts == set()
    assert diagnostic.require_protocol_compatibility is False
    assert {ReceiptType.METHOD_IDENTITY, ReceiptType.CODE_EXECUTION}.issubset(
        candidate.required_receipts
    )
    assert ReceiptType.STATIC_ACTUATION in candidate.required_receipts
    assert ReceiptType.RUNTIME_ACTUATION not in candidate.required_receipts
    assert ReceiptType.DERIVATION in positive_result.required_receipts
    assert ReceiptType.STATIC_ACTUATION not in positive_result.required_receipts
    assert ReceiptType.RUNTIME_ACTUATION not in positive_result.required_receipts
    assert {
        ReceiptType.DERIVATION,
        ReceiptType.STATIC_ACTUATION,
        ReceiptType.RUNTIME_ACTUATION,
    }.issubset(positive_adopted.required_receipts)
    assert (
        ReceiptType.COUNTERFACTUAL_ACTUATION
        not in positive_adopted.required_receipts
    )

    causal_claim = Claim(
        claim_id="causal",
        claim_type=ClaimType.CAUSAL_ATTRIBUTION,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task"},
        method_fingerprint="method-hash",
        protocol_ref=protocol_ref,
        statement="The admitted experience changed the resulting program.",
    )
    causal = compiler.compile(
        causal_claim,
        _request(protocol_ref, Operation.DISTILL_POSITIVE_ADOPTED),
    )
    assert ReceiptType.COUNTERFACTUAL_ACTUATION in causal.required_receipts
    assert causal.require_positive_effect is False

    effective_claim = Claim(
        **{
            **causal_claim.__dict__,
            "claim_id": "effective",
            "boundary": {"required_actuation_level": 5},
        }
    )
    effective = compiler.compile(
        effective_claim,
        _request(protocol_ref, Operation.DISTILL_POSITIVE_ADOPTED),
    )
    assert effective.require_positive_effect is True
    assert effective.required_payload_flags[
        ReceiptType.COUNTERFACTUAL_ACTUATION
    ]["effective"] is True
