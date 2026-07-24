from __future__ import annotations

import copy

import pytest

from authority.actuation import ExperienceContract, ExperienceContractCompiler
from authority.models import (
    GenerationStage,
    GovernanceStage,
    Operation,
    ProtocolRef,
    SOPClauseV1,
    TaskContext,
    VisibilityRequest,
)


PROTOCOL = ProtocolRef("test-protocol", "1", "a" * 64)


def _request() -> VisibilityRequest:
    return VisibilityRequest(
        operation=Operation.GENERATE_CANDIDATE,
        generation_stage=GenerationStage.MODEL_DESIGN,
        governance_stage=GovernanceStage.RETRIEVAL,
        active_protocol=PROTOCOL,
        task_context=TaskContext(task_id="task-a", task_family="tabular"),
        memory_bundle_version="bundle-v1",
        token_budget=1000,
        requesting_component="tests",
    )


def _clause() -> SOPClauseV1:
    return SOPClauseV1(
        clause_id="clause-a",
        sop_id="sop-a",
        text="Align OOF predictions by sample_id.",
        retrieval_text="Align OOF predictions by sample_id.",
        claim_refs=("claim-a",),
        claim_types=("method_hypothesis",),
        source_artifact_refs=("run-a/node-a",),
        source_transition_refs=("run::run-a::transition::t1",),
        source_run_ids=("run-a",),
        source_task_ids=("source-task",),
        source_task_families=("Others",),
        source_domains=("tabular",),
        transfer_scope="same_domain",
        protocol_scope=(PROTOCOL.key(),),
        task_scope={"task_ids": ["task-a"]},
        permitted_operations=(Operation.GENERATE_CANDIDATE.value,),
        permitted_generation_stages=(GenerationStage.MODEL_DESIGN.value,),
        permitted_governance_stages=(GovernanceStage.RETRIEVAL.value,),
        publication_class="candidate",
        applies_when=("OOF predictions are joined from multiple folds",),
        prevents=("duplicate or missing training sample IDs",),
        contract_spec={
            "must_preserve": [
                {"name": "model_family", "expected": "lightgbm"}
            ],
            "must_change": [
                {"name": "oof_join_key", "expected": "sample_id"}
            ],
            "must_not_use": [
                {"name": "test_labels_read", "expected": False}
            ],
            "expected_runtime_observations": [
                {"name": "one_prediction_per_training_sample", "expected": True}
            ],
        },
    )


def test_contract_compiler_is_deterministic_and_binds_scope() -> None:
    compiler = ExperienceContractCompiler()
    first = compiler.compile(_clause(), _request())
    second = compiler.compile(_clause(), _request())

    assert first.as_dict() == second.as_dict()
    first.verify()
    assert first.contract_hash == second.contract_hash
    assert first.contract_id.startswith("experience_contract::")
    assert first.clause_id == "clause-a"
    assert first.sop_id == "sop-a"
    assert first.active_protocol_ref == PROTOCOL.key()
    assert first.task_scope == {"task_ids": ["task-a"]}
    assert first.source_transition_refs == ["run::run-a::transition::t1"]
    assert first.source_run_ids == ["run-a"]
    assert first.source_task_ids == ["source-task"]
    assert first.source_task_families == ["Others"]
    assert first.source_domains == ["tabular"]
    assert first.transfer_scope == "same_domain"
    assert first.target_task_id == "task-a"
    assert first.target_task_family == "tabular"
    assert first.target_domain == "tabular"
    assert first.minimum_writeback_level == 2

    preserve = {item.name: item.expected for item in first.must_preserve}
    changes = {item.name: item.expected for item in first.must_change}
    forbidden = {item.name: item.expected for item in first.must_not_use}
    runtime = {
        item.name: item.expected for item in first.expected_runtime_observations
    }
    assert preserve["active_protocol_ref"] == PROTOCOL.key()
    assert preserve["task_id"] == "task-a"
    assert preserve["model_family"] == "lightgbm"
    assert changes["clause_applied::clause-a"] is True
    assert changes["oof_join_key"] == "sample_id"
    assert forbidden["forbidden_dependency_count"] == 0
    assert forbidden["holdout_used_for_selection"] is False
    assert forbidden["test_labels_read"] is False
    assert runtime["target_path_executed"] is True
    assert runtime["one_prediction_per_training_sample"] is True


def test_contract_tampering_is_detected() -> None:
    payload = ExperienceContractCompiler().compile(
        _clause(), _request()
    ).as_dict()
    tampered = copy.deepcopy(payload)
    tampered["must_change"][0]["expected"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        ExperienceContract.from_dict(tampered)


def test_clause_prose_does_not_self_satisfy_contract() -> None:
    contract = ExperienceContractCompiler().compile(_clause(), _request())
    assert contract.must_change
    assert contract.expected_runtime_observations
    # Descriptions survive for auditability, but every item still requires a
    # separate host observation under a stable predicate name.
    assert any(item.description for item in contract.must_change)
    assert all(item.name for item in contract.must_change)
