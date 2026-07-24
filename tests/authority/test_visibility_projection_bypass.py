from __future__ import annotations

import json

import pytest

from agents.memory.sop_visibility_gateway import SOPVisibilityGateway
from authority.models import GenerationStage, Operation
from tests.authority.sop_visibility_helpers import (
    MIXED_SOP_ID,
    SCORE_CLAUSE_ID,
    build_mixed_authority,
    decision_snapshot,
    visibility_request,
    write_stage_fixture,
)


def _layer_with_snapshot(
    tmp_path,
    *,
    edge_kind: str,
    edge_outcome: str,
):
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    _engine, ref = build_mixed_authority()
    graph_path, index_path = write_stage_fixture(
        tmp_path,
        ref,
        edge_kind=edge_kind,
        edge_outcome=edge_outcome,
    )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    sop = next(node for node in graph["nodes"] if node["id"] == MIXED_SOP_ID)
    sop["clauses"][2]["authority_decision_refs"] = ["decision::snapshot"]
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    nodes = {node["id"]: node for node in graph["nodes"]}
    snapshots = {"decision::snapshot": decision_snapshot(ref)}
    gateway = SOPVisibilityGateway(
        nodes,
        mode="enforce",
        decision_lookup=snapshots.get,
    )
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        mode="run_forest_stage_hybrid",
        enable_agentic=False,
        top_k=6,
        visibility_gateway=gateway,
        visibility_active_protocol=ref,
        visibility_policy_version="authority_v1",
        visibility_task_id="task-1",
        visibility_bundle_version="bundle-v1",
    )
    return layer, ref


@pytest.mark.parametrize(
    ("edge_kind", "edge_outcome"),
    [
        ("navigation_attached_to", "quarantine"),
        ("distills_to", "allow"),
        ("authorized_distills_to", "deny"),
    ],
)
def test_rejected_or_legacy_edge_cannot_be_revived_by_attached_ids(
    tmp_path, edge_kind, edge_outcome
) -> None:
    layer, ref = _layer_with_snapshot(
        tmp_path,
        edge_kind=edge_kind,
        edge_outcome=edge_outcome,
    )
    request = visibility_request(
        ref,
        Operation.RANK,
        generation_stage=GenerationStage.IMPROVE,
    )
    visibility = layer._prepare_visibility(
        stage="improve",
        task_id="task-1",
        task_desc="rank historical scores",
        request=request,
    )

    assert visibility.effective_clause_ids == [SCORE_CLAUSE_ID]
    assert layer._sop_edge_migration["navigation_edge_count"] == 1
    assert layer._sop_edge_migration["authorized_edge_count"] == 0
    assert layer.nodes["transition::1"]["attached_sop_ids"] == [MIXED_SOP_ID]
    assert layer._active_sops_for_transition("transition::1") == []
    assert layer._tree_sop_projection(["node::child"], {MIXED_SOP_ID}) == []
    assert layer._causal_attachment_rows(
        layer.nodes["transition::1"],
        stage="improve",
        task_family="general",
        allowed_sop_ids={MIXED_SOP_ID},
    ) == []

    ranking = layer.rank_sop_hybrid(
        stage="improve",
        task_id="task-1",
        task_desc="rank historical scores",
        query_text="historical score",
        limit=6,
        visibility_request=request,
    )
    assert ranking["direct_sop_candidates"]
    assert ranking["direct_clean_sop_ids"] == []
    assert ranking["tree_projected_sop_ids"] == []
    assert ranking["fused_sop_candidates"] == []


def test_explicit_allowed_authorized_edge_is_the_only_adoption_path(tmp_path) -> None:
    layer, ref = _layer_with_snapshot(
        tmp_path,
        edge_kind="authorized_distills_to",
        edge_outcome="allow",
    )
    request = visibility_request(
        ref,
        Operation.RANK,
        generation_stage=GenerationStage.IMPROVE,
    )
    layer._prepare_visibility(
        stage="improve",
        task_id="task-1",
        task_desc="rank historical scores",
        request=request,
    )
    assert layer._sop_edge_migration["authorized_edge_count"] == 1
    assert layer._active_sops_for_transition("transition::1") == [MIXED_SOP_ID]
    assert layer._tree_sop_projection(["node::child"], {MIXED_SOP_ID}) == [
        MIXED_SOP_ID
    ]
