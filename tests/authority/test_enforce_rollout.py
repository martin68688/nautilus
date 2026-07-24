from __future__ import annotations

from pathlib import Path

import pytest

from agents.memory.sop_visibility_gateway import SOPVisibilityGateway
from authority.adapters.mlevolve import runtime
from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from authority.models import (
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    GenerationStage,
    GovernanceStage,
    Operation,
)
from authority.memory_snapshot import MemorySnapshotLoader
from tests.authority.sop_visibility_helpers import (
    SCORE_CLAUSE_ID,
    build_mixed_authority,
    mixed_nodes,
    visibility_request,
)
from tests.authority.test_mlevolve_adapter import fake_agent, node
from tests.authority.test_actuation_pipeline import _contract, _observations
from tests.test_memory_snapshot_overlay import build_tiny_bundle, write_current


def _configure_scope(agent, *, operations, generation_stages, governance_stages):
    cfg = agent.cfg.evaluation_authority
    cfg.rollout_id = "staged-enforce-test"
    cfg.enforce_operations = list(operations)
    cfg.enforce_generation_stages = list(generation_stages)
    cfg.enforce_governance_stages = list(governance_stages)


def test_enforce_applies_only_inside_frozen_operation_and_stage_scope(
    tmp_path: Path,
) -> None:
    agent = fake_agent(tmp_path, mode="enforce")
    _configure_scope(
        agent,
        operations=[Operation.RANK.value],
        generation_stages=[GenerationStage.IMPROVE.value],
        governance_stages=[GovernanceStage.BRANCH_SELECTION.value],
    )
    adapter = MLEvolveAuthorityAdapter(agent)

    in_scope = node("in-scope", clean=False)
    in_scope_decision = adapter.authorize_node(
        in_scope, Operation.RANK, DecisionStage.BRANCH_SELECTION, "test.enforce"
    )
    assert in_scope_decision.outcome == DecisionOutcome.DENY
    assert adapter.permits(in_scope_decision, legacy_allowed=True) is False

    outside_scope = node("outside-scope", clean=False)
    outside_scope.stage = "evolution"
    outside_decision = adapter.authorize_node(
        outside_scope,
        Operation.RANK,
        DecisionStage.BRANCH_SELECTION,
        "test.enforce",
    )
    assert outside_decision.outcome == DecisionOutcome.DENY
    assert adapter.permits(outside_decision, legacy_allowed=True) is True

    records = {record.artifact_id: record for record in adapter.rollout.records()}
    assert records["in-scope"].enforced is True
    assert records["outside-scope"].enforced is False


def test_canary_cannot_start_without_hash_verified_bound_bundle(tmp_path: Path) -> None:
    agent = fake_agent(tmp_path, mode="enforce")
    agent.cfg.evaluation_authority.require_bound_bundle = True
    adapter = MLEvolveAuthorityAdapter(agent)
    with pytest.raises(RuntimeError, match="hash-verified CURRENT Base Bundle"):
        adapter.authorize_node(
            node("unbound", clean=True),
            Operation.RANK,
            DecisionStage.BRANCH_SELECTION,
            "test.enforce",
        )


def test_canary_freezes_the_hash_verified_current_bundle(tmp_path: Path) -> None:
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    agent = fake_agent(tmp_path / "logs", mode="enforce")
    cfg = agent.cfg.evaluation_authority
    cfg.require_bound_bundle = True
    cfg.expected_bundle_id = manifest["bundle_id"]
    cfg.expected_bundle_manifest_sha256 = manifest["manifest_sha256"]
    adapter = MLEvolveAuthorityAdapter(agent)
    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "overlay",
        active_protocol_ref=adapter.active_protocol.key(),
        authority_policy_version=adapter.engine.policy_version,
    )

    adapter.configure_memory_snapshot(snapshot)
    adapter.seal_rollout_versions()

    assert adapter.rollout.frozen is True
    assert adapter.rollout.versions.bundle_id == manifest["bundle_id"]
    assert (
        adapter.rollout.versions.bundle_manifest_sha256
        == manifest["manifest_sha256"]
    )


def test_unattributed_clean_result_appends_fact_without_derivation_edge(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    agent = fake_agent(tmp_path / "logs", mode="enforce")
    _configure_scope(
        agent,
        operations=[Operation.PROMOTE_RESULT.value],
        generation_stages=[GenerationStage.IMPROVE.value],
        governance_stages=[GovernanceStage.MEMORY_WRITEBACK.value],
    )
    adapter = MLEvolveAuthorityAdapter(agent)
    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "overlay",
        active_protocol_ref=adapter.active_protocol.key(),
        authority_policy_version=adapter.engine.policy_version,
    )
    adapter.configure_memory_snapshot(snapshot)
    candidate = node("independent-result", clean=True)

    decision = adapter.authorize_node(
        candidate,
        Operation.PROMOTE_RESULT,
        DecisionStage.MEMORY_WRITEBACK,
        "test.result_writeback",
    )

    assert decision.outcome == DecisionOutcome.ALLOW
    assert adapter.append_authorized_memory_overlay(candidate) is True
    events = snapshot.session_overlay.events()
    assert len(events) == 1
    payload = events[0].payload
    assert payload["artifact_id"] == "independent-result"
    assert payload["publication_class"] == "result_fact"
    assert payload["derived_from_refs"] == []
    assert payload["adoption_status"] == "not_exposed"
    assert payload["verified_adoption_report_refs"] == []
    assert payload["exposure_report_refs"] == []
    assert len(payload["code_sha256"]) == 64


def test_adoption_and_causal_edges_are_separate_overlay_events(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    agent = fake_agent(tmp_path / "logs", mode="enforce")
    _configure_scope(
        agent,
        operations=[
            Operation.PUBLISH_ADOPTION.value,
            Operation.PUBLISH_CAUSAL.value,
        ],
        generation_stages=[GenerationStage.IMPROVE.value],
        governance_stages=[GovernanceStage.MEMORY_WRITEBACK.value],
    )
    adapter = MLEvolveAuthorityAdapter(agent)
    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "overlay",
        active_protocol_ref=adapter.active_protocol.key(),
        authority_policy_version=adapter.engine.policy_version,
    )
    adapter.configure_memory_snapshot(snapshot)
    adapter.engine.graph.add_claim(
        Claim(
            claim_id="source-method-claim",
            claim_type=ClaimType.METHOD_HYPOTHESIS,
            subject_artifact_id="source-artifact",
            task_scope={"task_id": "source-image-task"},
            method_fingerprint="a" * 64,
            protocol_ref=adapter.active_protocol,
            statement="A source experience method.",
        )
    )
    candidate = node("adopted-result", clean=True)
    contract = _contract()
    contract.claim_refs = ["source-method-claim"]
    contract.finalize()
    adapter.actuation_tracker.record_exposure(
        artifact_id=candidate.id,
        contracts=[contract],
        request_id="request-adoption",
    )
    adapter.actuation_tracker.record_claimed_adoption(
        artifact_id=candidate.id,
        contract_id=contract.contract_id,
    )
    preconditions, static, runtime = _observations(contract)
    adapter.actuation_tracker.record_static_observation(
        artifact_id=candidate.id,
        contract_id=contract.contract_id,
        preconditions=preconditions,
        observations=static,
    )
    adapter.actuation_tracker.record_runtime_observation(
        artifact_id=candidate.id,
        contract_id=contract.contract_id,
        observations=runtime,
    )

    adoption = adapter.authorize_experience_link(
        candidate,
        contract_id=contract.contract_id,
    )
    causal_before_pair = adapter.authorize_experience_link(
        candidate,
        contract_id=contract.contract_id,
        causal=True,
    )
    assert adoption.outcome == DecisionOutcome.ALLOW, adoption.missing_obligations
    assert causal_before_pair.outcome != DecisionOutcome.ALLOW
    assert adapter.append_authorized_experience_link(
        candidate,
        contract_id=contract.contract_id,
    ) is True
    assert adapter.append_authorized_experience_link(
        candidate,
        contract_id=contract.contract_id,
        causal=True,
    ) is False

    adapter.actuation_tracker.record_counterfactual(
        artifact_id=candidate.id,
        contract_id=contract.contract_id,
        pair_result={
            "pair_id": "pair-adoption",
            "control_hash": "b" * 64,
            "memory_on_action_hash": "c" * 64,
            "memory_off_action_hash": "d" * 64,
            "memory_on_code_hash": "e" * 64,
            "memory_off_code_hash": "f" * 64,
            "influence_confirmed": True,
            "protocol_legal": True,
            "effective": False,
        },
    )
    causal_after_pair = adapter.authorize_experience_link(
        candidate,
        contract_id=contract.contract_id,
        causal=True,
    )
    assert causal_after_pair.outcome == DecisionOutcome.ALLOW
    assert adapter.append_authorized_experience_link(
        candidate,
        contract_id=contract.contract_id,
        causal=True,
    ) is True

    events = snapshot.session_overlay.events()
    assert [event.event_type for event in events] == [
        "memory_derivation_edge",
        "memory_derivation_edge",
    ]
    assert [event.payload["kind"] for event in events] == [
        "adoption",
        "causal",
    ]
    for event in events:
        assert event.payload["source_claim_refs"] == ["source-method-claim"]
        assert event.payload["contract_id"] == contract.contract_id
        assert len(event.payload["edge_hash"]) == 64
        assert event.payload["edge_id"].startswith("experience_edge::")
        assert event.payload["static_receipt_refs"]
        assert event.payload["runtime_receipt_refs"]
    assert events[0].payload["actuation_level"] == 3
    assert events[1].payload["actuation_level"] == 4
    assert events[0].payload["counterfactual_receipt_refs"] == []
    assert events[1].payload["counterfactual_receipt_refs"]


def test_low_risk_internal_error_is_navigation_only_or_abstain(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = fake_agent(tmp_path, mode="enforce")
    _configure_scope(
        agent,
        operations=[
            Operation.INSPECT.value,
            Operation.GENERATE_CANDIDATE.value,
        ],
        generation_stages=[
            GenerationStage.DEBUG.value,
            GenerationStage.DRAFT.value,
        ],
        governance_stages=[GovernanceStage.RETRIEVAL.value],
    )
    adapter = MLEvolveAuthorityAdapter(agent)

    def broken(*_args, **_kwargs):
        raise RuntimeError("simulated low-risk adapter failure")

    monkeypatch.setattr(runtime, "claims_for_node", broken)
    inspect_node = node("inspect-error", clean=True)
    inspect_node.stage = "debug"
    inspect_decision = adapter.authorize_node(
        inspect_node, Operation.INSPECT, DecisionStage.DEBUG, "test.enforce"
    )
    assert inspect_decision.outcome == DecisionOutcome.ALLOW_WITH_WARNING
    assert adapter.permits(inspect_decision, legacy_allowed=True) is True
    assert "navigation only" in str(inspect_decision.required_action)

    draft_node = node("draft-error", clean=True)
    draft_node.stage = "draft"
    draft_decision = adapter.authorize_node(
        draft_node,
        Operation.GENERATE_CANDIDATE,
        DecisionStage.DRAFT,
        "test.enforce",
    )
    assert draft_decision.outcome == DecisionOutcome.DENY
    assert adapter.permits(draft_decision, legacy_allowed=True) is False
    assert "abstain" in str(draft_decision.required_action)


def test_clause_visibility_enforces_only_configured_generation_stage() -> None:
    engine, protocol_ref = build_mixed_authority()
    gateway = SOPVisibilityGateway(
        mixed_nodes(protocol_ref),
        mode="enforce",
        authority_engine=engine,
        decision_lookup=engine.decisions.get,
        enforce_operations=[Operation.RANK.value],
        enforce_generation_stages=[GenerationStage.DEBUG.value],
        enforce_governance_stages=[GovernanceStage.RETRIEVAL.value],
    )
    improve = gateway.evaluate(
        visibility_request(
            protocol_ref,
            Operation.RANK,
            generation_stage=GenerationStage.IMPROVE,
        )
    )
    assert improve.visibility_trace["request_enforced"] is False
    assert SCORE_CLAUSE_ID in improve.effective_clause_ids

    debug = gateway.evaluate(
        visibility_request(
            protocol_ref,
            Operation.RANK,
            generation_stage=GenerationStage.DEBUG,
        )
    )
    assert debug.visibility_trace["request_enforced"] is True
    assert SCORE_CLAUSE_ID not in debug.effective_clause_ids


def test_off_shadow_and_canary_enforce_profiles_remain_selectable() -> None:
    from config import _load_config_tree

    config_dir = Path(__file__).resolve().parents[2] / "mlevolve" / "config"
    off = _load_config_tree(config_dir / "config_authority_off.yaml")
    shadow = _load_config_tree(config_dir / "config_authority_shadow.yaml")
    canary = _load_config_tree(
        config_dir / "config_authority_canary_enforce.yaml"
    )

    assert off.evaluation_authority.mode == "off"
    assert shadow.evaluation_authority.mode == "shadow"
    assert canary.evaluation_authority.mode == "enforce"
    assert canary.evaluation_authority.require_bound_bundle is True
    assert list(canary.evaluation_authority.enforce_operations)
    assert {
        Operation.PROMOTE_RESULT.value,
        Operation.PUBLISH_ADOPTION.value,
        Operation.PUBLISH_CAUSAL.value,
    } <= set(canary.evaluation_authority.enforce_operations)
    assert set(canary.evaluation_authority.enforce_generation_stages) == {
        "draft",
        "improve",
        "debug",
    }
