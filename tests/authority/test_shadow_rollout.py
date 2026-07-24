from __future__ import annotations

import copy
from pathlib import Path

import pytest

from agents.memory.sop_visibility_gateway import SOPVisibilityGateway
from authority.models import (
    AuthorityDecision,
    DecisionOutcome,
    DecisionStage,
    GenerationStage,
    GovernanceStage,
    Operation,
)
from authority.rollout import (
    AuthorityRolloutController,
    RolloutVersionSet,
    build_shadow_review_packet,
    verify_shadow_review_packet,
)
from tests.authority.sop_visibility_helpers import (
    SCORE_CLAUSE_ID,
    build_mixed_authority,
    mixed_nodes,
    visibility_request,
)
from tests.authority.test_mlevolve_adapter import fake_agent, node


def _decision(decision_id: str = "decision-1") -> AuthorityDecision:
    return AuthorityDecision(
        decision_id=decision_id,
        outcome=DecisionOutcome.DENY,
        permitted_scope=None,
        satisfied_paths=[],
        missing_obligations=["receipt:evaluator"],
        blocking_receipts=[],
        required_action="satisfy evidence",
        policy_version="authority_v1",
        claim_id="claim-1",
        artifact_id="artifact-1",
        operation=Operation.RANK.value,
        decision_stage="branch_selection",
        generation_stage=GenerationStage.IMPROVE.value,
        governance_stage=GovernanceStage.BRANCH_SELECTION.value,
    )


def test_rollout_versions_freeze_on_first_comparison_and_do_not_duplicate() -> None:
    controller = AuthorityRolloutController(
        mode="shadow",
        versions=RolloutVersionSet(
            rollout_id="shadow-test",
            policy_version="authority_v1",
            protocol_ref="protocol@1#" + "a" * 64,
            collector_version="collector-7",
        ),
    )
    decision = _decision()
    first = controller.record(
        decision,
        legacy_allowed=True,
        effective_allowed=True,
        enforced=False,
    )
    repeated = controller.record(
        decision,
        legacy_allowed=False,
        effective_allowed=False,
        enforced=False,
    )

    assert controller.frozen is True
    assert repeated.record_hash == first.record_hash
    assert len(controller.records()) == 1
    report = controller.report()
    assert report["record_count"] == 1
    assert report["taxonomy_counts"] == {"legacy_allow_authority_deny": 1}
    with pytest.raises(RuntimeError, match="frozen"):
        controller.bind_bundle(bundle_id="bundle-v1", manifest_sha256="b" * 64)


def test_configured_bundle_version_must_match_loaded_current() -> None:
    controller = AuthorityRolloutController(
        mode="shadow",
        versions=RolloutVersionSet(
            rollout_id="bundle-pin-test",
            policy_version="authority_v1",
            protocol_ref="protocol@1#" + "a" * 64,
            collector_version="1",
            bundle_id="bundle-a",
            bundle_manifest_sha256="b" * 64,
        ),
    )
    with pytest.raises(ValueError, match="bundle ID"):
        controller.bind_bundle(
            bundle_id="bundle-b", manifest_sha256="b" * 64
        )
    with pytest.raises(ValueError, match="manifest"):
        controller.bind_bundle(
            bundle_id="bundle-a", manifest_sha256="c" * 64
        )


def test_adapter_shadow_preserves_legacy_and_records_full_decision(tmp_path: Path) -> None:
    agent = fake_agent(tmp_path, mode="shadow")
    agent.cfg.evaluation_authority.collector_version = "collector-7"
    agent.cfg.evaluation_authority.rollout_id = "adapter-shadow-test"
    from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter

    adapter = MLEvolveAuthorityAdapter(agent)
    agent.evaluation_authority = adapter
    contaminated = node("shadow-dirty", clean=False)

    assert adapter.gate_node(
        contaminated,
        Operation.RANK,
        DecisionStage.BRANCH_SELECTION,
        "test.shadow",
        legacy_allowed=True,
    ) is True
    assert adapter.collector_host.collector_version == "collector-7"
    records = adapter.rollout.records()
    assert len(records) == 1
    assert records[0].legacy_allowed is True
    assert records[0].authority_allowed is False
    assert records[0].effective_allowed is True
    assert records[0].enforced is False
    assert (tmp_path / "authority_rollout_report.json").is_file()


def test_visibility_shadow_records_legacy_and_full_clause_sets() -> None:
    engine, protocol_ref = build_mixed_authority()
    gateway = SOPVisibilityGateway(
        mixed_nodes(protocol_ref),
        mode="shadow",
        authority_engine=engine,
        decision_lookup=engine.decisions.get,
    )
    pack = gateway.evaluate(
        visibility_request(
            protocol_ref,
            Operation.RANK,
            generation_stage=GenerationStage.IMPROVE,
        )
    )
    trace = pack.visibility_trace

    assert trace["request_enforced"] is False
    assert SCORE_CLAUSE_ID in trace["legacy_visible_clause_ids"]
    assert SCORE_CLAUSE_ID not in trace["full_policy_visible_clause_ids"]
    assert SCORE_CLAUSE_ID in trace["effective_visible_clause_ids"]
    comparison = trace["visibility_comparison"]
    assert comparison["legacy_allow_authority_deny_count"] >= 1
    assert comparison["suppressed_count"] >= 1


def test_shadow_review_packet_requires_independent_completed_dispositions() -> None:
    controller = AuthorityRolloutController(
        mode="shadow",
        versions=RolloutVersionSet(
            rollout_id="review-packet-test",
            policy_version="authority_v1",
            protocol_ref="protocol@1#" + "a" * 64,
            collector_version="1",
        ),
    )
    controller.record(
        _decision("disagreement-1"),
        legacy_allowed=True,
        effective_allowed=True,
        enforced=False,
    )
    packet = build_shadow_review_packet(controller.records(), max_records=10)
    assert packet["population_count"] == 1
    assert packet["sample_count"] == 1
    with pytest.raises(ValueError, match="lacks reviewer"):
        verify_shadow_review_packet(packet, controller.records())

    reviewed = copy.deepcopy(packet)
    reviewed["sampled_records"][0]["review"] = {
        "reviewer": "independent-reviewer",
        "disposition": "confirmed_legacy_false_allow",
        "notes": "legacy allowed an evidence-incomplete rank action",
    }
    report = verify_shadow_review_packet(reviewed, controller.records())
    assert report["verified"] is True
    assert report["reviewed_sample_count"] == 1
    assert report["reviewers"] == ["independent-reviewer"]
    assert len(report["reviews_sha256"]) == 64
    assert report["disposition_counts"] == {
        "confirmed_legacy_false_allow": 1
    }
