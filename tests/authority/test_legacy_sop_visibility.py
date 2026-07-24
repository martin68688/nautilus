from __future__ import annotations

from pathlib import Path

import pytest

from authority.models import GenerationStage, Operation, ProtocolRef
from tests.authority.sop_visibility_helpers import visibility_request


REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
LEGACY_PROTOCOL = ProtocolRef("legacy-audit", "1", "a" * 64)


@pytest.fixture(scope="module")
def legacy_layer():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    return StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        mode="run_forest_stage_hybrid",
        enable_agentic=False,
        visibility_mode="enforce",
        visibility_active_protocol=LEGACY_PROTOCOL,
        visibility_policy_version="authority_v1",
        visibility_task_id="legacy-inspection",
        visibility_bundle_version="legacy-runforest-v1",
        visibility_token_budget=10_000_000,
    )


def test_all_281_legacy_sops_have_explicit_shadow_migration_coverage(
    legacy_layer,
) -> None:
    request = visibility_request(
        LEGACY_PROTOCOL,
        Operation.DEBUG_HYPOTHESIS,
        generation_stage=GenerationStage.DEBUG,
        token_budget=10_000_000,
        task_id="legacy-inspection",
        task_family="general",
    )
    report = legacy_layer.visibility_gateway.migration_report(request)
    pack = legacy_layer.visibility_gateway.evaluate(request)

    assert report["sop_count"] == 281
    assert report["legacy_sop_count"] == 281
    assert report["visible_sop_count"] == 281
    assert report["suppressed_sop_count"] == 0
    assert report["empty_sop_count"] == 0
    assert report["visible_clause_count"] == len(
        legacy_layer.visibility_gateway.clauses
    )
    assert report["rendered_token_count"] > 0
    assert pack.visibility_trace["legacy_clause_count"] == len(
        legacy_layer.visibility_gateway.clauses
    )
    assert len(pack.warning_clauses) == len(
        legacy_layer.visibility_gateway.clauses
    )


def test_2773_quarantine_edges_are_zero_consumption_for_high_risk_views(
    legacy_layer,
) -> None:
    migration = legacy_layer._sop_edge_migration
    assert migration["navigation_edge_count"] == 2773
    assert migration["authorized_edge_count"] == 0
    assert migration["authority_outcomes"] == {"quarantine": 2773}

    rank_request = visibility_request(
        LEGACY_PROTOCOL,
        Operation.RANK,
        generation_stage=GenerationStage.IMPROVE,
        token_budget=10_000_000,
        task_id="legacy-inspection",
        task_family="general",
    )
    rank_pack = legacy_layer._prepare_visibility(
        stage="improve",
        task_id="legacy-inspection",
        task_desc="rank historical evidence",
        request=rank_request,
    )
    rank_report = legacy_layer.visibility_gateway.migration_report(rank_request)
    assert rank_pack.effective_clause_ids == []
    assert rank_report["visible_sop_count"] == 0
    assert rank_report["suppressed_sop_count"] == 281
    assert rank_report["suppressed_clause_count"] == len(
        legacy_layer.visibility_gateway.clauses
    )
    assert sum(
        len(legacy_layer._active_transitions_for_sop(sop_id))
        for sop_id in legacy_layer._sops
    ) == 0

    debug_request = visibility_request(
        LEGACY_PROTOCOL,
        Operation.DEBUG_HYPOTHESIS,
        generation_stage=GenerationStage.DEBUG,
        token_budget=10_000_000,
        task_id="legacy-inspection",
        task_family="general",
    )
    debug_pack = legacy_layer._prepare_visibility(
        stage="debug",
        task_id="legacy-inspection",
        task_desc="debug legacy execution",
        request=debug_request,
    )
    assert debug_pack.effective_sop_ids
    assert debug_pack.warning_clauses
    assert sum(
        len(legacy_layer._active_transitions_for_sop(sop_id))
        for sop_id in legacy_layer._sops
    ) == 2773
