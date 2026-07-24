from __future__ import annotations

import dataclasses

import pytest

from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer
from agents.memory.sop_visibility_gateway import SOPVisibilityGateway
from authority.authority_engine import AuthorityEngine
from authority.collectors import TrustedCollectorHost
from authority.adapters.mlevolve.retrieval_gate import (
    authorize_clause_for_visibility,
)
from authority.domain_scope import (
    DOMAIN_GENERAL,
    SAME_DOMAIN,
    audit_same_domain_task_heldout_exposures,
    canonical_domain,
    transfer_is_compatible,
)
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    Claim,
    ClaimType,
    GenerationStage,
    GovernanceStage,
    Operation,
    ProtocolRef,
    ProtocolSpec,
    ReceiptType,
    SOPClauseV1,
    TaskContext,
    VisibilityRequest,
)
from authority.protocol_registry import ProtocolRegistry
from tests.authority.clean_replay_helpers import trusted_replay_receipts


PROTOCOL = ProtocolRef("domain-transfer-test", "1", "d" * 64)


def _request(
    *,
    task_family: str,
    protocol: ProtocolRef = PROTOCOL,
    task_id: str = "unseen-target-task",
) -> VisibilityRequest:
    return VisibilityRequest(
        operation=Operation.GENERATE_CANDIDATE,
        generation_stage=GenerationStage.MODEL_DESIGN,
        governance_stage=GovernanceStage.RETRIEVAL,
        active_protocol=protocol,
        task_context=TaskContext(
            task_id=task_id,
            task_family=task_family,
        ),
        memory_bundle_version="same-domain-bundle-v1",
        token_budget=1000,
        requesting_component="tests.domain_transfer",
    )


def _clause(
    *,
    transfer_scope: str = SAME_DOMAIN,
    protocol: ProtocolRef = PROTOCOL,
    publication_class: str = "candidate",
    receipt_refs: tuple[str, ...] = (),
) -> SOPClauseV1:
    return SOPClauseV1(
        clause_id="clause-source-image",
        sop_id="sop-source-image",
        text="Use a source-task image augmentation tactic.",
        retrieval_text="image augmentation tactic",
        claim_refs=("claim-source-image",),
        claim_types=("method_hypothesis",),
        source_artifact_refs=("run::source-run::node::n1",),
        source_run_ids=("source-run",),
        source_task_ids=("source-image-task",),
        source_task_families=("General Image",),
        source_domains=("image",),
        transfer_scope=transfer_scope,
        protocol_scope=(protocol.key(),),
        task_scope={"task_ids": ["source-image-task"]},
        permitted_operations=(Operation.GENERATE_CANDIDATE.value,),
        permitted_generation_stages=(GenerationStage.MODEL_DESIGN.value,),
        permitted_governance_stages=(GovernanceStage.RETRIEVAL.value,),
        publication_class=publication_class,
        receipt_refs=receipt_refs,
    )


def _trusted_method_fixture(
    *,
    trusted: bool = True,
) -> tuple[AuthorityEngine, ProtocolRef, SOPClauseV1, Claim]:
    registry = ProtocolRegistry()
    ref = registry.register(
        ProtocolSpec(
            protocol_id="same-domain-transfer-test",
            version="1",
            task_profile={"family": "image"},
        )
    ).ref()
    claim = Claim(
        claim_id="claim-source-image",
        claim_type=ClaimType.METHOD_HYPOTHESIS,
        subject_artifact_id="clean-image-replay",
        task_scope={
            "task_ids": ["source-image-task"],
            "task_families": ["General Image"],
        },
        method_fingerprint="a" * 64,
        protocol_ref=ref,
        statement="Use a clean replayed image augmentation method.",
        boundary={"clean_replay": True},
    )
    receipts = trusted_replay_receipts(
        TrustedCollectorHost("same-domain-transfer-host"),
        artifact_id=claim.subject_artifact_id,
        protocol_ref=ref,
        method_fingerprint=claim.method_fingerprint,
        code_sha256="b" * 64,
    )
    if not trusted:
        receipts[0] = dataclasses.replace(
            receipts[0], trust_status="legacy_static_only"
        )
    graph = EvidenceGraph()
    graph.add_claim(claim)
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id="path-source-image",
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipts],
        )
    )
    clause = dataclasses.replace(
        _clause(
            protocol=ref,
            publication_class="certified",
            receipt_refs=tuple(receipt.receipt_id for receipt in receipts),
        ),
        source_artifact_refs=("historical-source-image-artifact",),
    )
    return AuthorityEngine(registry, graph=graph), ref, clause, claim


def test_domain_normalization_and_cross_domain_escape_are_explicit() -> None:
    assert canonical_domain("General Image") == "image"
    assert canonical_domain("image_binary_classification") == "image"
    assert canonical_domain("NLP") == "nlp"
    assert canonical_domain("Audio") == "audio"
    assert canonical_domain("Others") == "tabular"
    assert transfer_is_compatible(["image"], "image_classification", SAME_DOMAIN)
    assert not transfer_is_compatible(["image"], "text_classification", SAME_DOMAIN)
    assert not transfer_is_compatible(["image"], "image", "")
    assert transfer_is_compatible(["image"], "text_classification", DOMAIN_GENERAL)


def test_visibility_scope_allows_same_domain_but_rejects_cross_domain() -> None:
    same_domain = authorize_clause_for_visibility(
        _clause(), _request(task_family="image_binary_classification")
    )
    assert same_domain.reason == "cross_task_requires_trusted_method_evidence"

    cross_domain = authorize_clause_for_visibility(
        _clause(), _request(task_family="text_classification")
    )
    assert cross_domain.allowed is False
    assert cross_domain.reason == "declared_scope_mismatch"

    missing_scope = authorize_clause_for_visibility(
        _clause(transfer_scope=""),
        _request(task_family="image_binary_classification"),
    )
    assert missing_scope.allowed is False
    assert missing_scope.reason == "declared_scope_mismatch"

    domain_general = authorize_clause_for_visibility(
        _clause(transfer_scope=DOMAIN_GENERAL),
        _request(task_family="text_classification"),
    )
    assert domain_general.reason == "cross_task_requires_trusted_method_evidence"


def test_certified_trusted_method_authorizes_source_claim_for_image_target() -> None:
    engine, ref, clause, claim = _trusted_method_fixture()
    source_scope_before = dataclasses.asdict(claim)["task_scope"]

    same_domain = authorize_clause_for_visibility(
        clause,
        _request(
            protocol=ref,
            task_id="aerial-cactus-identification",
            task_family="image_binary_classification",
        ),
        authority_engine=engine,
    )

    assert same_domain.allowed is True
    assert same_domain.reason == "authority_allow"
    assert len(same_domain.authority_decision_refs) == 1
    decision = engine.decisions[same_domain.authority_decision_refs[0]]
    assert decision.permitted_scope is not None
    assert decision.permitted_scope.task_ids == ["source-image-task"]
    assert claim.task_scope == source_scope_before

    for target_family in ("text_classification", "audio_classification"):
        denied = authorize_clause_for_visibility(
            clause,
            _request(protocol=ref, task_family=target_family),
            authority_engine=engine,
        )
        assert denied.allowed is False
        assert denied.reason == "declared_scope_mismatch"

    missing_scope = authorize_clause_for_visibility(
        dataclasses.replace(clause, transfer_scope=""),
        _request(protocol=ref, task_family="image_binary_classification"),
        authority_engine=engine,
    )
    assert missing_scope.allowed is False
    assert missing_scope.reason == "declared_scope_mismatch"


def test_cross_task_provisional_method_requires_trusted_path_binding() -> None:
    engine, ref, clause, _claim = _trusted_method_fixture()
    candidate = authorize_clause_for_visibility(
        dataclasses.replace(clause, publication_class="candidate"),
        _request(protocol=ref, task_family="image_binary_classification"),
        authority_engine=engine,
    )
    assert candidate.allowed is True
    assert candidate.reason == "authority_allow"

    untrusted_engine, untrusted_ref, untrusted_clause, _claim = (
        _trusted_method_fixture(trusted=False)
    )
    untrusted = authorize_clause_for_visibility(
        untrusted_clause,
        _request(
            protocol=untrusted_ref,
            task_family="image_binary_classification",
        ),
        authority_engine=untrusted_engine,
    )
    assert untrusted.allowed is False
    assert untrusted.reason == "cross_task_requires_trusted_method_evidence"


def test_cross_task_provisional_method_needs_only_execution_and_method_identity() -> None:
    engine, ref, clause, claim = _trusted_method_fixture()
    minimal_graph = EvidenceGraph()
    minimal_graph.add_claim(
        dataclasses.replace(claim, boundary={"source_clear": True})
    )
    minimal_receipts = [
        receipt
        for receipt in engine.graph.receipts.values()
        if receipt.receipt_type
        in {ReceiptType.METHOD_IDENTITY, ReceiptType.CODE_EXECUTION}
    ]
    assert {receipt.receipt_type for receipt in minimal_receipts} == {
        ReceiptType.METHOD_IDENTITY,
        ReceiptType.CODE_EXECUTION,
    }
    for receipt in minimal_receipts:
        minimal_graph.add_receipt(receipt)
    minimal_graph.add_path(
        EvidencePath(
            path_id="path-source-image-provisional",
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in minimal_receipts],
        )
    )
    minimal_engine = AuthorityEngine(engine.registry, graph=minimal_graph)
    provisional = dataclasses.replace(
        clause,
        publication_class="provisional",
        receipt_refs=tuple(receipt.receipt_id for receipt in minimal_receipts),
    )

    decision = authorize_clause_for_visibility(
        provisional,
        _request(
            protocol=ref,
            task_id="aerial-cactus-identification",
            task_family="image_binary_classification",
        ),
        authority_engine=minimal_engine,
    )

    assert decision.allowed is True
    assert decision.warning is False
    assert decision.reason == "authority_allow"


def test_cross_task_provisional_method_cannot_inherit_source_score_authority() -> None:
    engine, ref, clause, _claim = _trusted_method_fixture()
    provisional = dataclasses.replace(clause, publication_class="provisional")

    generated = authorize_clause_for_visibility(
        provisional,
        _request(protocol=ref, task_family="image_binary_classification"),
        authority_engine=engine,
    )
    rank_request = dataclasses.replace(
        _request(protocol=ref, task_family="image_binary_classification"),
        operation=Operation.RANK,
        generation_stage=GenerationStage.IMPROVE,
        governance_stage=GovernanceStage.BRANCH_SELECTION,
    )
    ranked = authorize_clause_for_visibility(
        provisional,
        rank_request,
        authority_engine=engine,
    )

    assert generated.allowed is True
    assert ranked.allowed is False
    assert ranked.reason == "publication_class_incompatible"


def test_strict_runforest_sop_filter_fails_closed_before_ranking() -> None:
    layer = object.__new__(StageAwareHybridMemoryLayer)
    layer.domain_scope_required = True

    assert not layer._sop_task_compatible({}, "image_binary_classification")
    assert not layer._sop_task_compatible(
        {"task_families": ["general"]}, "image_binary_classification"
    )
    assert layer._sop_task_compatible(
        {
            "source_domains": ["image"],
            "transfer_scopes": [SAME_DOMAIN],
        },
        "image_binary_classification",
    )
    assert not layer._sop_task_compatible(
        {
            "source_domains": ["nlp"],
            "transfer_scopes": [SAME_DOMAIN],
        },
        "image_binary_classification",
    )
    assert layer._sop_task_compatible(
        {
            "source_domains": ["nlp"],
            "transfer_scopes": [DOMAIN_GENERAL],
        },
        "image_binary_classification",
    )


def test_explicit_runforest_clauses_do_not_gain_legacy_fallback_duplicates() -> None:
    clause = _clause()
    nodes = {
        clause.sop_id: {
            "id": clause.sop_id,
            "type": "SOP",
            "sop_id": clause.sop_id,
            "title": "formal image SOP",
            "clause_ids": [clause.clause_id],
            "source_domains": ["image"],
            "transfer_scopes": [SAME_DOMAIN],
        },
        clause.clause_id: {
            **dataclasses.asdict(clause),
            "id": clause.clause_id,
            "type": "SOPClause",
        },
    }
    gateway = SOPVisibilityGateway(nodes, mode="shadow")

    assert set(gateway.clauses) == {clause.clause_id}
    assert gateway.clause_ids_by_sop[clause.sop_id] == [clause.clause_id]
    loaded = gateway.clauses[clause.clause_id]
    assert loaded.source_run_ids == ("source-run",)
    assert loaded.source_domains == ("image",)
    assert loaded.transfer_scope == SAME_DOMAIN


def test_explicit_container_with_missing_clause_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing explicit clauses"):
        SOPVisibilityGateway(
            {
                "sop-missing": {
                    "id": "sop-missing",
                    "type": "SOP",
                    "clause_ids": ["clause-missing"],
                }
            },
            mode="shadow",
        )


def test_domain_scoped_container_cannot_fall_back_to_legacy_prose() -> None:
    with pytest.raises(ValueError, match="missing explicit clause_ids"):
        SOPVisibilityGateway(
            {
                "sop-domain-corrupt": {
                    "id": "sop-domain-corrupt",
                    "type": "SOP",
                    "title": "must not become a legacy fallback",
                    "source_domains": ["image"],
                    "transfer_scopes": [SAME_DOMAIN],
                    "domain_scope_complete": True,
                }
            },
            mode="shadow",
        )


def _exposure_event(
    clause: SOPClauseV1,
    *,
    operation: str,
    generation_stage: str,
) -> dict[str, object]:
    return {
        "schema": "experience_exposure_event_v2",
        "contract_id": f"contract::{clause.clause_id}",
        "clause_id": clause.clause_id,
        "source_task_ids": list(clause.source_task_ids),
        "source_domains": list(clause.source_domains),
        "transfer_scope": clause.transfer_scope,
        "target_scope": {
            "task_id": "aerial-cactus-identification",
            "task_family": "image_binary_classification",
            "domain": "image",
        },
        "operation": operation,
        "generation_stage": generation_stage,
        "governance_stage": GovernanceStage.RETRIEVAL.value,
        "publication_class": clause.publication_class,
        "prompt_sha256": "f" * 64,
    }


def test_task_heldout_exposure_audit_stratifies_certified_and_debug_views() -> None:
    _engine, ref, certified, _claim = _trusted_method_fixture()
    diagnostic = dataclasses.replace(
        _clause(protocol=ref, publication_class="diagnostic"),
        clause_id="clause-source-image-debug",
        sop_id="sop-source-image-debug",
        claim_refs=("claim-source-image-debug",),
        claim_types=(ClaimType.DEBUG_REPAIR.value,),
        source_artifact_refs=("run::debug-image::node::n1",),
        source_run_ids=("debug-image",),
        source_task_ids=("source-debug-image-task",),
        permitted_operations=(Operation.DEBUG_HYPOTHESIS.value,),
        permitted_generation_stages=(GenerationStage.DEBUG.value,),
        permitted_governance_stages=(GovernanceStage.RETRIEVAL.value,),
        receipt_refs=(),
    )
    report = audit_same_domain_task_heldout_exposures(
        [
            _exposure_event(
                certified,
                operation=Operation.GENERATE_CANDIDATE.value,
                generation_stage=GenerationStage.MODEL_DESIGN.value,
            ),
            _exposure_event(
                diagnostic,
                operation=Operation.DEBUG_HYPOTHESIS.value,
                generation_stage=GenerationStage.DEBUG.value,
            ),
        ],
        [dataclasses.asdict(certified), dataclasses.asdict(diagnostic)],
        target_task_id="aerial-cactus-identification",
        target_domain="image",
        certified_clause_id=certified.clause_id,
        certified_source_task_id="source-image-task",
    )

    assert report["valid"] is True
    assert report["exposure_event_count"] == 2
    assert report["classified_exposure_count"] == 2
    assert report["certified_method_exposure_count"] == 1
    assert report["diagnostic_debug_exposure_count"] == 1
    assert report["invalid_exposure_count"] == 0


def test_task_heldout_exposure_audit_rejects_unstratified_or_unbound_events() -> None:
    _engine, _ref, certified, _claim = _trusted_method_fixture()
    event = _exposure_event(
        certified,
        operation=Operation.GENERATE_CANDIDATE.value,
        generation_stage=GenerationStage.MODEL_DESIGN.value,
    )
    event["schema"] = "experience_exposure_event_v1"
    event["source_domains"] = ["nlp"]
    report = audit_same_domain_task_heldout_exposures(
        [event],
        [dataclasses.asdict(certified)],
        target_task_id="aerial-cactus-identification",
        target_domain="image",
        certified_clause_id=certified.clause_id,
        certified_source_task_id="source-image-task",
    )

    assert report["valid"] is False
    assert report["invalid_exposure_count"] == 1
    reasons = report["invalid_exposures"][0]["reasons"]
    assert "exposure_schema_not_v2" in reasons
    assert "source_domain_mismatch" in reasons
    assert "source_domain_binding_mismatch" in reasons


def test_task_heldout_exposure_audit_requires_positive_certified_transfer() -> None:
    _engine, ref, certified, _claim = _trusted_method_fixture()
    diagnostic = dataclasses.replace(
        _clause(protocol=ref, publication_class="diagnostic"),
        clause_id="clause-only-debug",
        sop_id="sop-only-debug",
        claim_refs=("claim-only-debug",),
        claim_types=(ClaimType.DEBUG_REPAIR.value,),
        source_task_ids=("source-debug-image-task",),
        permitted_operations=(Operation.DEBUG_HYPOTHESIS.value,),
        permitted_generation_stages=(GenerationStage.DEBUG.value,),
        permitted_governance_stages=(GovernanceStage.RETRIEVAL.value,),
        receipt_refs=(),
    )
    report = audit_same_domain_task_heldout_exposures(
        [
            _exposure_event(
                diagnostic,
                operation=Operation.DEBUG_HYPOTHESIS.value,
                generation_stage=GenerationStage.DEBUG.value,
            )
        ],
        [dataclasses.asdict(certified), dataclasses.asdict(diagnostic)],
        target_task_id="aerial-cactus-identification",
        target_domain="image",
        certified_clause_id=certified.clause_id,
        certified_source_task_id="source-image-task",
    )

    assert report["valid"] is False
    assert report["certified_method_exposure_count"] == 0
    assert report["invalid_exposures"][-1]["reasons"] == [
        "certified_method_was_never_exposed"
    ]
