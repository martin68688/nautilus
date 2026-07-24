from __future__ import annotations

import copy

from authority.clean_replay import ReplayAuthorityRecovery
from authority.authority_engine import AuthorityEngine
from authority.collectors import TrustedCollectorHost
from authority.evidence_graph import EvidenceGraph
from authority.models import (
    AuthorityRequest,
    DecisionOutcome,
    DecisionStage,
    GenerationStage,
    GovernanceStage,
    Operation,
    TaskContext,
)
from authority.replay_certifier import (
    ProtocolRepairSurface,
    ReplayIdentity,
    verify_protocol_only_patch,
)
from tests.authority.clean_replay_helpers import (
    SOURCE_CODE,
    build_registry,
    historical_score_claim,
    trusted_replay_receipts,
)


def test_method_change_creates_successor_without_restoring_old_claim() -> None:
    registry, ref = build_registry()
    changed = SOURCE_CODE.replace("LogisticRegression", "RandomForestClassifier")
    verification = verify_protocol_only_patch(
        SOURCE_CODE,
        changed,
        ProtocolRepairSurface.from_protocol_spec(registry.resolve(ref)),
        source_artifact_id="historical-artifact",
        replay_artifact_id="successor-artifact",
    )
    assert verification.identity == ReplayIdentity.SUCCESSOR_METHOD
    graph = EvidenceGraph()
    original = historical_score_claim(
        "claim::historical",
        "historical-artifact",
        ref,
        verification.source_method_fingerprint,
    )
    graph.add_claim(original)
    before = copy.deepcopy(original)
    receipts = trusted_replay_receipts(
        TrustedCollectorHost("successor-host"),
        artifact_id="successor-artifact",
        protocol_ref=ref,
        method_fingerprint=verification.replay_method_fingerprint,
        code_sha256=verification.replay_code_sha256,
    )

    registration = ReplayAuthorityRecovery(graph, registry).register(
        original_claim_id=original.claim_id,
        verification=verification,
        receipts=receipts,
        protocol_ref=ref,
        statement="The successor method obtains a new clean replay score of 0.41.",
    )

    assert registration.identity == ReplayIdentity.SUCCESSOR_METHOD
    assert registration.replay_claim_id.startswith("claim::successor::")
    assert registration.replay_claim_id != original.claim_id
    assert registration.authority_recovered_for_old_claim is False
    assert graph.claims[original.claim_id] == before
    assert graph.claim_paths[original.claim_id] == []
    successor = graph.claims[registration.replay_claim_id]
    assert successor.boundary["predecessor_claim_id"] == original.claim_id
    assert successor.boundary["old_claim_authority_recovered"] is False
    assert successor.parent_claims == []
    assert graph.paths[registration.path_id].claim_id == successor.claim_id
    engine = AuthorityEngine(registry, graph=graph)

    def rank(claim):
        return engine.authorize(
            AuthorityRequest(
                artifact_id=claim.subject_artifact_id,
                claim_id=claim.claim_id,
                operation=Operation.RANK,
                decision_stage=DecisionStage.BRANCH_SELECTION,
                active_protocol=ref,
                task_context=TaskContext(task_id="task-a", task_family="tabular"),
                requesting_component="tests.successor",
                generation_stage=GenerationStage.IMPROVE,
                governance_stage=GovernanceStage.BRANCH_SELECTION,
            )
        )

    assert rank(successor).outcome == DecisionOutcome.ALLOW
    assert rank(original).outcome != DecisionOutcome.ALLOW


def test_human_review_result_does_not_mutate_graph() -> None:
    registry, ref = build_registry()
    verification = verify_protocol_only_patch(
        SOURCE_CODE,
        SOURCE_CODE + "\nunknown_side_effect()\n",
        ProtocolRepairSurface.from_protocol_spec(registry.resolve(ref)),
        source_artifact_id="historical-artifact",
        replay_artifact_id="review-artifact",
    )
    graph = EvidenceGraph()
    original = historical_score_claim(
        "claim::historical",
        "historical-artifact",
        ref,
        verification.source_method_fingerprint,
    )
    graph.add_claim(original)
    registration = ReplayAuthorityRecovery(graph, registry).register(
        original_claim_id=original.claim_id,
        verification=verification,
        receipts=[],
        protocol_ref=ref,
        statement="must not be registered",
    )
    assert registration.identity == ReplayIdentity.REQUIRE_HUMAN_REVIEW
    assert set(graph.claims) == {original.claim_id}
    assert graph.paths == {}
    assert graph.receipts == {}
