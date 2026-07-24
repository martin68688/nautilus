from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from authority.bundle_publisher import (
    PublicationValidationError,
    SleepTimePipeline,
    classify_writeback_events,
)
from authority.memory_snapshot import MemorySnapshotLoader
from authority.memory_snapshot import ImmutableBaseBundle, sha256_file, sha256_json, write_json_atomic
from authority.models import (
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    GenerationStage,
    GovernanceStage,
    Operation,
)
from authority.writeback_distillation import build_positive_writeback_plan
from tests.authority.test_actuation_pipeline import _contract, _observations
from tests.authority.test_enforce_rollout import _configure_scope
from tests.authority.test_mlevolve_adapter import fake_agent, node
from tests.test_memory_snapshot_overlay import build_tiny_bundle, write_current


def test_result_adoption_and_causal_objects_are_separately_materialized(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    log_dir = tmp_path / "logs"
    agent = fake_agent(log_dir, mode="enforce")
    _configure_scope(
        agent,
        operations=[
            Operation.PROMOTE_RESULT.value,
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
    candidate = node("target-result", clean=True)
    result_decision = adapter.authorize_node(
        candidate,
        Operation.PROMOTE_RESULT,
        DecisionStage.MEMORY_WRITEBACK,
        "tests.result_fact",
    )
    assert result_decision.outcome == DecisionOutcome.ALLOW
    assert adapter.append_authorized_memory_overlay(candidate) is True
    journal_path = log_dir / "journal.json"
    journal_path.write_text(
        json.dumps(
            {"nodes": [{"id": candidate.id, "code": candidate.code}]},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    snapshot.session_overlay.freeze_to(tmp_path / "result-only")
    result_only = classify_writeback_events(tmp_path / "result-only")
    assert result_only["result_fact_count"] == 1
    assert result_only["adoption_edge_count"] == 0
    assert result_only["causal_edge_count"] == 0
    assert (
        result_only["result_facts"][0]["payload"]["derived_from_refs"]
        == []
    )
    result_only_plan = build_positive_writeback_plan(result_only)
    assert result_only_plan["positive_result_candidate_count"] == 1
    assert result_only_plan["positive_adopted_candidate_count"] == 0

    adapter.engine.graph.add_claim(
        Claim(
            claim_id="source-method-claim",
            claim_type=ClaimType.METHOD_HYPOTHESIS,
            subject_artifact_id="source-artifact",
            task_scope={"task_id": "source-image-task"},
            method_fingerprint="a" * 64,
            protocol_ref=adapter.active_protocol,
            statement="A source method Claim.",
        )
    )
    contract = _contract()
    contract.claim_refs = ["source-method-claim"]
    contract.finalize()
    adapter.actuation_tracker.record_exposure(
        artifact_id=candidate.id,
        contracts=[contract],
        request_id="request-edge",
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
        candidate, contract_id=contract.contract_id
    )
    assert adoption.outcome == DecisionOutcome.ALLOW
    assert adapter.append_authorized_experience_link(
        candidate, contract_id=contract.contract_id
    ) is True
    assert adapter.append_authorized_experience_link(
        candidate, contract_id=contract.contract_id
    ) is False

    causal_without_pair = adapter.authorize_experience_link(
        candidate, contract_id=contract.contract_id, causal=True
    )
    assert causal_without_pair.outcome != DecisionOutcome.ALLOW
    assert "receipt:counterfactual_actuation" in (
        causal_without_pair.missing_obligations
    )
    adapter.actuation_tracker.record_counterfactual(
        artifact_id=candidate.id,
        contract_id=contract.contract_id,
        pair_result={
            "pair_id": "pair-edge",
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
    causal = adapter.authorize_experience_link(
        candidate, contract_id=contract.contract_id, causal=True
    )
    assert causal.outcome == DecisionOutcome.ALLOW
    assert adapter.append_authorized_experience_link(
        candidate, contract_id=contract.contract_id, causal=True
    ) is True
    snapshot.session_overlay.freeze_to(tmp_path / "all-writeback")
    inventory = classify_writeback_events(tmp_path / "all-writeback")
    assert inventory["result_fact_count"] == 1
    assert inventory["adoption_edge_count"] == 1
    assert inventory["causal_edge_count"] == 1
    adoption_payload = inventory["adoption_edges"][0]["payload"]
    causal_payload = inventory["causal_edges"][0]["payload"]
    assert adoption_payload["source_claim_refs"] == ["source-method-claim"]
    assert adoption_payload["counterfactual_receipt_refs"] == []
    assert causal_payload["counterfactual_receipt_refs"]

    plan = build_positive_writeback_plan(inventory)
    assert plan["positive_result_candidate_count"] == 1
    assert plan["positive_adopted_candidate_count"] == 1
    proposals = [
        {
            "candidate_id": item["candidate_id"],
            "title": (
                "Validated target result"
                if item["kind"] == "result"
                else "Verified adopted experience"
            ),
            "text": (
                "Reuse the target-node method after target-task validation."
                if item["kind"] == "result"
                else "Reuse the experience that was statically and dynamically adopted."
            ),
            "source_task_family": "image_binary_classification",
            "source_domain": "image",
        }
        for item in plan["items"]
    ]
    proposals_path = tmp_path / "positive-proposals.json"
    proposals_path.write_text(
        json.dumps({"proposals": proposals}, sort_keys=True),
        encoding="utf-8",
    )
    binder_root = Path(__file__).resolve().parents[2] / "paper-skills" / "memory_bundle"
    sys.path.insert(0, str(binder_root))
    try:
        from bind_positive_writeback import bind

        report = bind(
            tmp_path / "all-writeback",
            proposals_path,
            adapter.protocol_registry.registry_dir,
            tmp_path / "positive-bound",
            policy_version=adapter.engine.policy_version,
            collector_version=adapter.collector_version,
        )
    finally:
        sys.path.remove(str(binder_root))
    assert report["positive_result_count"] == 1
    assert report["positive_adopted_count"] == 1
    clauses = [
        json.loads(line)
        for line in (tmp_path / "positive-bound" / "clauses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["publication_class"] for row in clauses} == {
        "positive_result",
        "positive_adopted",
    }
    assert {tuple(row["claim_types"]) for row in clauses} == {
        (ClaimType.METHOD_HYPOTHESIS.value,)
    }

    def build_candidate(context):
        context["candidate_dir"].mkdir(parents=True, exist_ok=True)

    def passed(_context):
        return {"status": "passed"}

    def pipeline_with_distillation(report):
        return SleepTimePipeline(
            audit=passed,
            claim_decomposition=passed,
            distillation=lambda _context: report,
            build_candidate=build_candidate,
            derivation_validation=passed,
            visibility_validation=passed,
            bundle_validation=passed,
        )

    with pytest.raises(
        PublicationValidationError,
        match="typed writeback plan",
    ):
        pipeline_with_distillation({"status": "passed"})(
            snapshot.base_bundle,
            tmp_path / "all-writeback",
            tmp_path / "ignored-candidate",
        )
    pipeline_report = pipeline_with_distillation(report)(
        snapshot.base_bundle,
        tmp_path / "all-writeback",
        tmp_path / "accepted-candidate",
    )
    assert pipeline_report["distillation"]["positive_result_count"] == 1

    # Exercise the actual staging builder, not only its callback contract.
    from tests.authority.test_replay_authority_recovery import (
        _prepare_tiny_replay_clause_bundle,
    )
    pipeline_parent_root = tmp_path / "positive-publish-parent"
    publish_parent, publish_manifest = build_tiny_bundle(pipeline_parent_root)
    _prepare_tiny_replay_clause_bundle(
        publish_parent,
        publish_manifest,
        adapter.active_protocol.key(),
    )
    publish_manifest["artifact_hashes"] = {
        path.relative_to(publish_parent).as_posix(): sha256_file(path)
        for path in sorted(publish_parent.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    publish_manifest["manifest_sha256"] = sha256_json(
        {
            key: value
            for key, value in publish_manifest.items()
            if key != "manifest_sha256"
        }
    )
    write_json_atomic(publish_parent / "manifest.json", publish_manifest)
    immutable_parent = ImmutableBaseBundle.load(publish_parent)
    pipeline_module_root = Path(__file__).resolve().parents[2] / "paper-skills" / "memory_bundle"
    sys.path.insert(0, str(pipeline_module_root))
    try:
        from positive_writeback_pipeline import make_positive_writeback_pipeline

        production_pipeline = make_positive_writeback_pipeline(
            new_version="v2",
            proposals_path=proposals_path,
            protocol_registry_path=adapter.protocol_registry.registry_dir,
            policy_version=adapter.engine.policy_version,
            collector_version=adapter.collector_version,
        )
    finally:
        sys.path.remove(str(pipeline_module_root))
    staged = tmp_path / "positive-staged"
    reports = production_pipeline(
        immutable_parent,
        tmp_path / "all-writeback",
        staged,
    )
    assert reports["bundle_validation"]["valid"] is True
    staged_manifest = json.loads(
        (staged / "manifest.json").read_text(encoding="utf-8")
    )
    assert staged_manifest["bundle_version"] == "v2"
    assert staged_manifest["parent_bundle"] == publish_manifest["bundle_id"]
    assert staged_manifest["positive_writeback_counts"] == {
        "result": 1,
        "adopted": 1,
    }
