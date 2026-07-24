from __future__ import annotations

import pytest

from authority.models import (
    AuthorityRequest,
    DecisionStage,
    Operation,
    ProtocolRef,
    TaskContext,
)
from authority.stage_ontology import (
    GenerationStage,
    GovernanceStage,
    known_runtime_stage_mappings,
    resolve_stage_axes,
    runtime_stage_axes,
)


def _protocol() -> ProtocolRef:
    return ProtocolRef("test", "1", "hash-v1")


def test_every_runtime_stage_has_one_explicit_dual_axis_mapping() -> None:
    mappings = known_runtime_stage_mappings()
    assert set(mappings) == {
        "root",
        "draft",
        "model_design",
        "improve",
        "debug",
        "evolution",
        "fusion",
        "fusion_draft",
    }
    for runtime_stage, expected in mappings.items():
        assert runtime_stage_axes(runtime_stage) == expected
        assert isinstance(expected.generation_stage, GenerationStage)
        assert isinstance(expected.governance_stage, GovernanceStage)


def test_runtime_generation_and_governance_operation_are_orthogonal() -> None:
    axes = resolve_stage_axes(
        runtime_stage="draft",
        legacy_stage=DecisionStage.BRANCH_SELECTION,
    )
    assert axes.generation_stage == GenerationStage.DRAFT
    assert axes.governance_stage == GovernanceStage.BRANCH_SELECTION


def test_legacy_authority_request_is_migrated_but_preserved() -> None:
    request = AuthorityRequest(
        artifact_id="artifact",
        claim_id="claim",
        operation=Operation.RANK,
        decision_stage=DecisionStage.BRANCH_SELECTION,
        active_protocol=_protocol(),
        task_context=TaskContext("task"),
        requesting_component="test",
    )
    assert request.decision_stage == DecisionStage.BRANCH_SELECTION
    assert request.generation_stage == GenerationStage.IMPROVE
    assert request.governance_stage == GovernanceStage.BRANCH_SELECTION


def test_new_request_records_both_axes_and_emits_legacy_compatibility_stage() -> None:
    request = AuthorityRequest(
        artifact_id="artifact",
        claim_id="claim",
        operation=Operation.GENERATE_CANDIDATE,
        decision_stage=None,
        active_protocol=_protocol(),
        task_context=TaskContext("task"),
        requesting_component="test",
        generation_stage=GenerationStage.MODEL_DESIGN,
        governance_stage=GovernanceStage.RETRIEVAL,
    )
    assert request.generation_stage == GenerationStage.MODEL_DESIGN
    assert request.governance_stage == GovernanceStage.RETRIEVAL
    assert request.decision_stage == DecisionStage.RETRIEVAL


def test_unknown_or_missing_stage_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown runtime stage"):
        runtime_stage_axes("mystery")
    with pytest.raises(ValueError, match="generation_stage is required"):
        resolve_stage_axes()
