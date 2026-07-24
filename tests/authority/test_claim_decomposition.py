from __future__ import annotations

import hashlib
from types import SimpleNamespace

from authority.adapters.mlevolve.transition_adapter import decompose_transition_claims
from authority.claim_decomposer import ClaimBoundaryProposal, decompose_node_claims
from authority.models import ClaimType, ProtocolRef


def _protocol() -> ProtocolRef:
    return ProtocolRef("test", "1", "p" * 64)


def _mixed_node():
    code = """
oof = predictions.set_index(sample_id).reindex(train_ids)
best = scores.loc[test_labels].idxmax()
"""
    return SimpleNamespace(
        id="mixed",
        stage="debug",
        code=code,
        plan="Fix OOF sample_id index alignment while preserving the model.",
        analysis="",
        metric=SimpleNamespace(value=0.92, maximize=True),
        exec_time=1.0,
        method_fingerprint=hashlib.sha256(code.encode()).hexdigest(),
        leakage_audit={
            "status": "blocked",
            "issues": [{
                "issue_code": "TEST_LABEL_MODEL_SELECTION",
                "category": "target_leakage",
                "severity": "critical",
                "evidence": "test labels selected the model",
            }],
        },
        derived_from_refs=[],
        claim_refs=[],
    )


def test_mixed_node_splits_fact_claims_with_stable_ids_and_bindings() -> None:
    node = _mixed_node()
    first = decompose_node_claims(node, _protocol(), "task")
    second = decompose_node_claims(_mixed_node(), _protocol(), "task")
    types = {claim.claim_type for claim in first.claims}
    assert {
        ClaimType.EXECUTED,
        ClaimType.METHOD_HYPOTHESIS,
        ClaimType.DEBUG_REPAIR,
        ClaimType.AUDIT_FINDING,
        ClaimType.SCORE,
    }.issubset(types)
    assert [claim.claim_id for claim in first.claims] == [
        claim.claim_id for claim in second.claims
    ]
    assert first.claims_of_type(ClaimType.SCORE)[0].claim_id == "node:mixed:score"
    assert all(claim.source_artifact_refs == ["node:mixed"] for claim in first.claims)
    assert all(first.bindings[claim.claim_id]["evidence_refs"] for claim in first.claims)
    assert set(node.claim_refs) == {claim.claim_id for claim in first.claims}


def test_llm_can_reword_one_bound_fact_but_cannot_invent_authority() -> None:
    node = _mixed_node()
    baseline = decompose_node_claims(node, _protocol(), "task")
    repair = baseline.claims_of_type(ClaimType.DEBUG_REPAIR)[0]
    proposal = ClaimBoundaryProposal(
        claim_type="debug_repair",
        statement="Align OOF rows by immutable sample IDs.",
        source_refs=("node:mixed",),
        evidence_refs=tuple(repair.evidence_refs),
        boundary={"applies_when": "OOF order differs from row order"},
    )
    result = decompose_node_claims(
        _mixed_node(), _protocol(), "task", proposals=[proposal]
    )
    rebound = result.claims_of_type(ClaimType.DEBUG_REPAIR)[0]
    assert rebound.claim_id == repair.claim_id
    assert rebound.statement == proposal.statement
    assert rebound.boundary["fact"] == "debug_repair_candidate"
    assert rebound.boundary["llm_boundary"] == proposal.boundary

    forged = ClaimBoundaryProposal(
        claim_type="pairwise_superiority",
        statement="The method is better.",
        source_refs=("invented:source",),
        evidence_refs=("invented:evidence",),
    )
    rejected = decompose_node_claims(
        _mixed_node(), _protocol(), "task", proposals=[forged]
    )
    assert rejected.claims_of_type(ClaimType.PAIRWISE_SUPERIORITY) == []
    assert rejected.quarantined_proposals[0]["reason"] == "unbound_source_ref"


def test_legacy_records_are_explicitly_static_only() -> None:
    result = decompose_node_claims(
        _mixed_node(), _protocol(), "task", legacy_static_only=True
    )
    assert {claim.legacy_status for claim in result.claims} == {"legacy_static_only"}
    assert result.deterministic_facts["legacy_status"] == "legacy_static_only"


def test_transition_claims_bind_both_parent_and_child_sources() -> None:
    parent = SimpleNamespace(id="parent", claim_refs=["claim:parent"])
    child = _mixed_node()
    result = decompose_transition_claims(parent, child, _protocol(), "task")
    for claim in result.claims:
        assert "node:parent" in claim.source_artifact_refs
        assert "node:mixed" in claim.source_artifact_refs
        assert claim.parent_claims == ["claim:parent"]
        assert claim.claim_id in child.claim_refs
